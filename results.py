"""Persistent storage of energy hub optimisation runs.

Every run is written into its own directory under ``results/``::

    results/
      index.csv                          one row per run, for comparison tables
      20260828-193100_base_d8b09ee1ff/
        run.json                         scenario, metrics, environment, summary
        parameters.json                  every component parameter, harvested
        costs.csv                        cost breakdown per component
        sizes.csv                        S_exist / S_new / S_total per component
        solution.nc                      all variable values (xarray/netCDF)

The run id encodes when the run happened, the scenario name and the scenario
fingerprint.  Together with the git commit and the parameter fingerprint stored
in ``run.json``, that is enough to tell whether two runs are comparable: the
scenario fingerprint covers the configured knobs, the parameter fingerprint
covers component data hardcoded in the model classes, and the commit covers
the formulation itself.

Reading back::

    from results import ResultStore
    store = ResultStore()
    store.index()                        # DataFrame of all runs
    run = store.load("20260828-193100_base_d8b09ee1ff")
    run["solution"]["ELYS_u_1"]          # xarray DataArray
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from benchmark import RunMetrics
from scenario import Scenario

COST_TERMS = ("C_inv", "C_mnt", "C_deg", "C_opr", "C_rev", "C_peak")
SIZE_TERMS = ("S_exist", "S_min", "S_max", "S_new", "S_total")

# modules whose objects are model artefacts / raw input data rather than
# parameters, and are therefore not recursed into when harvesting
_OPAQUE_MODULES = ("linopy", "pandas", "xarray", "numpy")


# --------------------------------------------------------------------------- #
# value extraction
# --------------------------------------------------------------------------- #
def _is_opaque(value: Any) -> bool:
    return type(value).__module__.split(".")[0] in _OPAQUE_MODULES


def evaluate(value: Any) -> float | None:
    """Reduce a model quantity to a single number, or ``None`` if it is not one.

    Handles plain numbers, solved :class:`linopy.Variable` /
    :class:`linopy.LinearExpression` objects, and scalar
    :class:`xarray.DataArray` objects.  Expressions are evaluated *including*
    their constant term, so cost expressions come out in euros directly.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    solution = getattr(value, "solution", None)
    if solution is not None:
        value = solution

    if isinstance(value, xr.DataArray):
        if value.size == 1:
            return float(value.values)
        return None
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(()))
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _sanitise(value: Any) -> Any:
    """Replace NaN / +-Inf by ``null`` so the output is strict JSON."""
    if isinstance(value, float):
        return (
            value
            if value == value and value not in (float("inf"), float("-inf"))
            else None
        )
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _sanitise(payload), indent=2, default=_json_default, allow_nan=False
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# harvesting the model
# --------------------------------------------------------------------------- #
def _summarise_array(array: xr.DataArray) -> Any:
    """Describe a time series parameter without storing all 8760 values."""
    values = np.asarray(array.values, dtype=float)
    if values.size == 0:
        return {"shape": list(values.shape)}
    first = values.flat[0]
    if values.size == 1 or bool(np.all(values == first)):
        return float(first)
    return {
        "shape": list(values.shape),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "sum": float(np.nansum(values)),
    }


def harvest_parameters(
    components: Mapping[str, Any], max_depth: int = 3
) -> dict[str, Any]:
    """Collect every numeric parameter of the component, recursively.

    Sub-assemblies (``PVP.PVI``, ``BESS.BATT``, ``RCS.WTNK``, ...) are walked as
    well and reported under dotted keys.  Decision variables, expressions and
    raw input dataframes are skipped; time varying parameters
    (:class:`xarray.DataArray`) are reduced to a constant if they are one, and
    to min/max/mean/sum otherwise.
    """
    harvested: dict[str, Any] = {}

    def walk(obj: Any, path: str, depth: int) -> None:
        if depth > max_depth or not hasattr(obj, "__dict__"):
            return
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            full_key = f"{path}.{key}"
            if isinstance(value, bool) or isinstance(
                value, (int, float, str, np.integer, np.floating)
            ):
                harvested[full_key] = (
                    value if isinstance(value, (bool, str)) else float(value)
                )
            elif isinstance(value, xr.DataArray):
                harvested[full_key] = _summarise_array(value)
            elif isinstance(value, pd.DataFrame):
                harvested[full_key] = {"dataframe_shape": list(value.shape)}
            elif isinstance(value, pd.Index):
                harvested[full_key] = {"index_name": value.name, "length": len(value)}
            elif isinstance(value, pd.Series):
                harvested[full_key] = {"series_length": len(value)}
            elif not _is_opaque(value) and hasattr(value, "__dict__"):
                walk(value, full_key, depth + 1)

    for name, component in components.items():
        walk(component, name, 1)
    return harvested


def parameters_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Hash of the harvested parameters.

    Component data is hardcoded in the model classes rather than in the
    scenario, so this is what tells you that a run used different numbers even
    though the scenario was nominally the same.
    """
    blob = json.dumps(parameters, sort_keys=True, default=_json_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def solution_dataset(model: Any) -> xr.Dataset:
    """All solved variables in one dataset, each keeping its own dimension.

    :attr:`linopy.Model.solution` broadcasts every variable onto the union of
    all dimensions.  Each component here carries its own time index
    (``k_SUPP``, ``k_PVP``, ...), so that union is over twenty dimensions and
    the broadcast blows up; assembling the dataset variable by variable does
    not.
    """
    data = {}
    for name in model.variables:
        try:
            data[name] = model.variables[name].solution
        except (AttributeError, ValueError):  # variable without a solution
            continue
    return xr.Dataset(data)


def timeseries(
    solution: xr.Dataset,
    scenario: Scenario,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Time varying variables as one dataframe on a datetime index.

    Every component has its own time axis ``k_<COMP>`` with its own time step,
    ``scenario.t_of(COMP)``.  The frame is indexed at the finest of these
    steps, and series on a coarser step are held over the finer steps inside
    each of their own -- a port value is the mean over its step, so holding
    keeps the amounts.  Variables on other dimensions, i.e. scalar sizing
    decisions (in ``sizes.csv``) and the monthly grid peaks (``omega_EGC``),
    are left out.
    """
    horizon = pd.Timedelta(days=scenario.N_days)
    series = {}
    for name in variables if variables is not None else solution.data_vars:
        array = solution[name]
        if array.ndim != 1 or not array.dims[0].startswith("k_"):
            continue
        step = pd.Timedelta(hours=scenario.t_of(array.dims[0][2:]))
        if step * array.size != horizon:
            raise ValueError(
                f"{name}: {array.size} steps of {step} do not span {horizon}"
            )
        index = pd.date_range(start=scenario.start_date, periods=array.size, freq=step)
        series[name] = pd.Series(np.asarray(array.values), index=index)
    if not series:
        return pd.DataFrame()
    step = min(s.index.freq for s in series.values())
    index = pd.date_range(
        start=scenario.start_date,
        end=scenario.start_date + horizon,
        freq=step,
        inclusive="left",
    )
    return pd.DataFrame(
        {name: s.reindex(index, method="ffill") for name, s in series.items()}
    )


def cost_breakdown(components: Mapping[str, Any]) -> pd.DataFrame:
    """Evaluated ``C_inv`` ... ``C_peak`` per component, at the solution.

    These are the raw cost terms in euros; the net present cost combines them
    as ``CAPEX + annuity * (OPEX - REV)`` and is recorded under ``objective``
    in ``run.json``, so the ``C_total`` column here does not add up to it.
    """
    rows = []
    for name, component in components.items():
        row = {"component": name}
        for term in COST_TERMS:
            row[term] = evaluate(getattr(component, term, None))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("component")
    frame["C_total"] = frame[list(COST_TERMS)].sum(axis=1, skipna=True)
    frame.loc["TOTAL"] = frame.sum(axis=0, skipna=True)
    return frame


def size_breakdown(components: Mapping[str, Any], max_depth: int = 3) -> pd.DataFrame:
    """Evaluated capacities of every component and sub-assembly."""
    rows: dict[str, dict[str, Any]] = {}

    def walk(obj: Any, path: str, depth: int) -> None:
        if depth > max_depth or not hasattr(obj, "__dict__"):
            return
        found = {
            term: evaluate(getattr(obj, term, None))
            for term in SIZE_TERMS
            if hasattr(obj, term)
        }
        if found:
            rows[path] = found
        for key, value in vars(obj).items():
            if key.startswith("_") or key in SIZE_TERMS:
                continue
            if not _is_opaque(value) and hasattr(value, "__dict__"):
                walk(value, f"{path}.{key}", depth + 1)

    for name, component in components.items():
        walk(component, name, 1)

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "component"
    return frame.reindex(columns=[c for c in SIZE_TERMS if c in frame.columns])


# --------------------------------------------------------------------------- #
# a completed run
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """Everything ``case_study.main`` produces, ready to be stored."""

    scenario: Scenario
    model: Any  # linopy.Model
    components: dict[str, Any]
    metrics: RunMetrics
    objective: dict[str, Any] = field(default_factory=dict)
    # set by the opt3s model, which knows these directly; when None they are
    # harvested from ``components`` as for the legacy model
    parameters: dict[str, Any] | None = None
    costs: pd.DataFrame | None = None
    sizes: pd.DataFrame | None = None
    amounts: pd.DataFrame | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. model name/version

    @property
    def solved(self) -> bool:
        return str(getattr(self.model, "status", "")) == "ok"


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
class ResultStore:
    """Writes and reads run directories under ``root`` (default ``results/``)."""

    INDEX_NAME = "index.csv"

    def __init__(self, root: str | Path = "results") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- writing -----------------------------------------------------------
    def run_id(self, scenario: Scenario, when: datetime | None = None) -> str:
        stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "-_=." else "-" for c in scenario.name
        )
        return f"{stamp}_{safe_name}_{scenario.fingerprint}"

    def save(
        self,
        result: RunResult,
        save_solution: bool = True,
        run_id: str | None = None,
    ) -> Path:
        """Write one run to disk and append it to the index.

        Returns the run directory.
        """
        run_id = run_id or self.run_id(result.scenario)
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        parameters = (
            result.parameters
            if result.parameters is not None
            else harvest_parameters(result.components)
        )
        param_hash = parameters_fingerprint(parameters)
        _write_json(run_dir / "parameters.json", parameters)

        costs = (
            result.costs if result.costs is not None else cost_breakdown(result.components)
        )
        costs.to_csv(run_dir / "costs.csv")
        sizes = (
            result.sizes if result.sizes is not None else size_breakdown(result.components)
        )
        sizes.to_csv(run_dir / "sizes.csv")
        amounts_name = None
        if result.amounts is not None:
            amounts_name = "amounts.csv"
            result.amounts.to_csv(run_dir / amounts_name)

        solution_path = None
        if save_solution and result.solved:
            solution_path = run_dir / "solution.nc"
            self._write_solution(result.model, solution_path)

        record = {
            "run_id": run_id,
            "scenario": result.scenario.to_dict(),
            "scenario_fingerprint": result.scenario.fingerprint,
            "parameters_fingerprint": param_hash,
            "objective": result.objective,
            "metrics": result.metrics.to_dict(),
            "model": result.extra.get("model", "legacy case_study.py"),
            "files": {
                "parameters": "parameters.json",
                "costs": "costs.csv",
                "sizes": "sizes.csv",
                "amounts": amounts_name,
                "solution": solution_path.name if solution_path else None,
            },
        }
        _write_json(run_dir / "run.json", record)

        self._append_index(record, costs)
        return run_dir

    @staticmethod
    def _write_solution(model: Any, path: Path) -> None:
        """Store all variable values as netCDF."""
        solution = solution_dataset(model)
        try:  # engines that support compression (netcdf4, h5netcdf)
            encoding = {
                name: {"zlib": True, "complevel": 4} for name in solution.data_vars
            }
            solution.to_netcdf(path, encoding=encoding)
        except (TypeError, ValueError):  # e.g. the scipy engine
            solution.to_netcdf(path)

    def _append_index(self, record: Mapping[str, Any], costs: pd.DataFrame) -> None:
        row: dict[str, Any] = {"run_id": record["run_id"]}
        row.update(
            {
                f"sc_{k}": v
                for k, v in record["scenario"].items()
                if k not in ("solver_options", "t_comp", "params")
            }
        )
        row["sc_solver_options"] = json.dumps(record["scenario"]["solver_options"])
        row["sc_t_comp"] = json.dumps(record["scenario"].get("t_comp", {}))
        row["sc_params"] = json.dumps(record["scenario"].get("params", {}))
        row["model"] = record.get("model")
        row["scenario_fingerprint"] = record["scenario_fingerprint"]
        row["parameters_fingerprint"] = record["parameters_fingerprint"]
        row.update({f"obj_{k}": v for k, v in record["objective"].items()})

        metrics = record["metrics"]
        for phase_name, phase in metrics["phases"].items():
            row[f"{phase_name}_wall_s"] = phase["wall_s"]
            row[f"{phase_name}_cpu_s"] = phase["cpu_s"]
            row[f"{phase_name}_rss_delta_mb"] = phase["rss_delta_mb"]
        row.update(metrics["totals"])
        for key in ("n_rows", "n_cols", "n_nonzeros", "nnz_per_row", "density"):
            row[key] = metrics["size"].get(key)
        for key in (
            "status",
            "termination_condition",
            "simplex_iterations",
            "ipm_iterations",
            "solver_runtime_s",
        ):
            row[key] = metrics["solver"].get(key)
        row["git_commit"] = metrics["environment"]["git"]["commit"]
        row["git_dirty"] = metrics["environment"]["git"]["dirty"]
        row["hostname"] = metrics["environment"]["hostname"]
        for term in COST_TERMS:
            row[f"total_{term}"] = costs.loc["TOTAL", term]

        index_path = self.root / self.INDEX_NAME
        frame = pd.DataFrame([row])
        if index_path.exists():
            existing = pd.read_csv(index_path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_csv(index_path, index=False)

    # --- reading -----------------------------------------------------------
    def index(self) -> pd.DataFrame:
        """All recorded runs as one table, newest last."""
        index_path = self.root / self.INDEX_NAME
        if not index_path.exists():
            return pd.DataFrame()
        return pd.read_csv(index_path)

    def runs(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "run.json").exists())

    def load(self, run_id: str, with_solution: bool = True) -> dict[str, Any]:
        """Read one stored run back into memory."""
        run_dir = self.root / run_id
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        loaded: dict[str, Any] = dict(record)
        loaded["scenario"] = Scenario.from_dict(record["scenario"])
        loaded["parameters"] = json.loads(
            (run_dir / "parameters.json").read_text(encoding="utf-8")
        )
        loaded["costs"] = pd.read_csv(run_dir / "costs.csv", index_col=0)
        loaded["sizes"] = pd.read_csv(run_dir / "sizes.csv", index_col=0)
        amounts_name = record["files"].get("amounts")
        loaded["amounts"] = (
            pd.read_csv(run_dir / amounts_name, index_col=0) if amounts_name else None
        )
        loaded["solution"] = None
        solution_name = record["files"].get("solution")
        if with_solution and solution_name and (run_dir / solution_name).exists():
            loaded["solution"] = xr.open_dataset(run_dir / solution_name)
        loaded["path"] = run_dir
        return loaded

    def compare(
        self,
        run_ids: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Side by side comparison table, runs as columns.

        With no arguments it reports the benchmark columns for every run, which
        is the table that goes into the paper.
        """
        frame = self.index()
        if frame.empty:
            return frame
        frame = frame.set_index("run_id")
        if run_ids is not None:
            frame = frame.loc[list(run_ids)]
        if columns is None:
            columns = [
                "sc_name",
                "n_rows",
                "n_cols",
                "n_nonzeros",
                "formulation_wall_s",
                "solving_wall_s",
                "total_wall_s",
                "peak_rss_mb",
                "simplex_iterations",
                "obj_value",
                "status",
            ]
        columns = [c for c in columns if c in frame.columns]
        return frame[columns].T
