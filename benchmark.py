"""Performance measurement for the energy hub optimisation runs.

Collects, for a single run:

* **problem size** -- rows, columns and nonzeros of the constraint matrix
* **timings** -- per phase (setup / formulation / solving / postprocessing),
  wall clock and CPU, plus the solver's own reported runtime
* **memory** -- process working set before/after each phase and the peak
  working set of the whole process (which includes the native allocations made
  by HiGHS, invisible to tracemalloc)
* **solver statistics** -- simplex / IPM / crossover iteration counts,
  termination status, objective value, infeasibility residuals

Nothing here depends on the model itself, so the same harness can be reused for
the other scripts in this directory.

Typical use::

    metrics = RunMetrics()
    with metrics.phase("formulation"):
        m = build_model()
    metrics.problem_size(m)                 # before solving: linopy counters
    with metrics.phase("solving"):
        m.solve(...)
    metrics.problem_size(m)                 # after solving: read back from HiGHS
    metrics.solver_stats(m)
    print(metrics.summary())
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator

MB = 1024.0**2

DEFAULT_PACKAGES = ("linopy", "highspy", "gurobipy", "numpy", "pandas", "xarray", "scipy")


# --------------------------------------------------------------------------- #
# memory probes
# --------------------------------------------------------------------------- #
class _WindowsMemoryProbe:
    """Reads the process working set through ``K32GetProcessMemoryInfo``.

    Preferred on Windows because it also accounts for memory allocated inside
    the HiGHS extension module, which ``tracemalloc`` cannot see.
    """

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def __init__(self) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.GetCurrentProcess.restype = ctypes.c_void_p
        self._fn = self._k32.K32GetProcessMemoryInfo
        self._fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._PROCESS_MEMORY_COUNTERS),
            ctypes.c_uint32,
        ]
        self._fn.restype = ctypes.c_int

    def __call__(self) -> tuple[float, float]:
        counters = self._PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = self._k32.GetCurrentProcess()
        if not self._fn(handle, ctypes.byref(counters), counters.cb):
            return float("nan"), float("nan")
        return counters.WorkingSetSize / MB, counters.PeakWorkingSetSize / MB


def _make_memory_probe():
    """Return a callable ``() -> (rss_mb, peak_rss_mb)``; NaNs if unavailable."""
    if sys.platform == "win32":
        try:
            probe = _WindowsMemoryProbe()
            probe()
            return probe
        except Exception:  # pragma: no cover - defensive
            pass
    try:  # any platform, if psutil happens to be installed
        import psutil

        process = psutil.Process()

        def probe_psutil() -> tuple[float, float]:
            info = process.memory_info()
            peak = getattr(info, "peak_wset", None)
            return info.rss / MB, (peak / MB if peak else float("nan"))

        return probe_psutil
    except Exception:
        pass
    try:  # POSIX fallback: peak only
        import resource

        def probe_resource() -> tuple[float, float]:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = peak / MB if sys.platform == "darwin" else peak / 1024.0
            return float("nan"), peak

        return probe_resource
    except Exception:  # pragma: no cover - defensive
        return lambda: (float("nan"), float("nan"))


memory_probe = _make_memory_probe()


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def _package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _git_revision(cwd: Path | None = None) -> dict[str, Any]:
    """Commit hash of the working tree, and whether it has uncommitted edits."""
    cwd = Path(cwd or Path(__file__).parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if commit.returncode != 0:
            return {"commit": None, "dirty": None}
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"commit": commit.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def environment_info(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Everything needed to make a timing comparable across machines."""
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "packages": _package_versions(packages),
        "git": _git_revision(),
    }


# --------------------------------------------------------------------------- #
# solver statistics
# --------------------------------------------------------------------------- #
# Gurobi model status codes, see the Gurobi reference manual
_GUROBI_STATUS = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
    6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}  # fmt: skip


def _gurobi_stats(grb) -> dict[str, Any]:
    """Iteration counts, runtime and solution quality of a solved gurobipy model.

    Every attribute is read on its own, since some exist only for some
    algorithms (e.g. the violations need a basic or barrier solution).
    """
    attributes = {
        "simplex_iterations": ("IterCount", int),
        "ipm_iterations": ("BarIterCount", int),
        "solver_runtime_s": ("Runtime", float),
        "solver_work": ("Work", float),
        "max_primal_infeasibility": ("ConstrVio", float),
        "max_bound_violation": ("BoundVio", float),
        "max_dual_infeasibility": ("DualVio", float),
    }
    stats: dict[str, Any] = {}
    for key, (attribute, cast) in attributes.items():
        try:
            stats[key] = cast(getattr(grb, attribute))
        except Exception:
            stats[key] = None
    try:
        stats["gurobi_status"] = _GUROBI_STATUS.get(int(grb.Status), str(grb.Status))
        stats["gurobi_method"] = int(grb.Params.Method)
        stats["gurobi_threads"] = int(grb.Params.Threads)
    except Exception as exc:  # pragma: no cover - solver API drift
        stats["solver_stats_error"] = repr(exc)
    return stats


# --------------------------------------------------------------------------- #
# phase timings
# --------------------------------------------------------------------------- #
def _isnan(value: float) -> bool:
    return value != value


@dataclass
class Phase:
    """Wall/CPU time and memory footprint of one stage of a run."""

    name: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    rss_start_mb: float = float("nan")
    rss_end_mb: float = float("nan")
    peak_rss_mb: float = float("nan")
    py_peak_mb: float = float("nan")  # tracemalloc, only when tracing is on

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_end_mb - self.rss_start_mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_s": self.wall_s,
            "cpu_s": self.cpu_s,
            "rss_start_mb": self.rss_start_mb,
            "rss_end_mb": self.rss_end_mb,
            "rss_delta_mb": self.rss_delta_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "py_peak_mb": self.py_peak_mb,
        }


@dataclass
class RunMetrics:
    """Accumulates the measurements of a single optimisation run.

    Parameters
    ----------
    trace_python_memory
        Additionally run :mod:`tracemalloc` around every phase.  This isolates
        the memory allocated by Python objects (the linopy model) from the
        solver's native allocations, at the cost of roughly a 2x slowdown of
        the formulation phase -- leave it off for runs whose timings you report.
    """

    trace_python_memory: bool = False
    phases: dict[str, Phase] = field(default_factory=dict)
    size: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=environment_info)

    # --- timing ------------------------------------------------------------
    @contextmanager
    def phase(self, name: str) -> Iterator[Phase]:
        """Time and profile the enclosed block, recording it as ``name``.

        Re-entering an existing phase name accumulates into that phase.
        """
        entry = self.phases.get(name)
        rss_start, _ = memory_probe()
        if entry is None:
            entry = Phase(name=name, rss_start_mb=rss_start)
            self.phases[name] = entry

        if self.trace_python_memory:
            tracemalloc.start()
            tracemalloc.reset_peak()

        wall0, cpu0 = time.perf_counter(), time.process_time()
        try:
            yield entry
        finally:
            entry.wall_s += time.perf_counter() - wall0
            entry.cpu_s += time.process_time() - cpu0
            if self.trace_python_memory:
                _, py_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                previous = 0.0 if _isnan(entry.py_peak_mb) else entry.py_peak_mb
                entry.py_peak_mb = max(previous, py_peak / MB)
            entry.rss_end_mb, entry.peak_rss_mb = memory_probe()

    @property
    def total_wall_s(self) -> float:
        return sum(phase.wall_s for phase in self.phases.values())

    @property
    def total_cpu_s(self) -> float:
        return sum(phase.cpu_s for phase in self.phases.values())

    @property
    def peak_rss_mb(self) -> float:
        """Peak working set of the process across all measured phases."""
        peaks = [
            phase.peak_rss_mb
            for phase in self.phases.values()
            if not _isnan(phase.peak_rss_mb)
        ]
        return max(peaks) if peaks else float("nan")

    # --- problem size ------------------------------------------------------
    def problem_size(self, model, exact_nonzeros: bool = True) -> dict[str, Any]:
        """Record rows / columns / nonzeros of the LP handed to the solver.

        After :meth:`linopy.Model.solve` with ``io_api="direct"`` the numbers
        are read straight from the solver (HiGHS or Gurobi) -- cheap and
        authoritative, and they refer to the model as loaded, before the
        solver's own presolve.  Called before
        solving, the row and column counts come from linopy's counters and the
        nonzero count from the flattened constraint terms
        (``exact_nonzeros=True``, which materialises a dataframe and is not
        free for large models).
        """
        size: dict[str, Any] = {
            "n_variables": int(model.nvars),
            "n_constraints": int(model.ncons),
            "source": "linopy",
        }

        solver_model = getattr(model, "solver_model", None)
        if solver_model is not None and hasattr(solver_model, "getNumNz"):
            try:
                size.update(
                    n_rows=int(solver_model.getNumRow()),
                    n_cols=int(solver_model.getNumCol()),
                    n_nonzeros=int(solver_model.getNumNz()),
                    source="highs",
                )
            except Exception:
                pass
        elif solver_model is not None and hasattr(solver_model, "NumNZs"):
            try:
                size.update(
                    n_rows=int(solver_model.NumConstrs),
                    n_cols=int(solver_model.NumVars),
                    n_nonzeros=int(solver_model.NumNZs),
                    source="gurobi",
                )
            except Exception:
                pass

        if size["source"] == "linopy":
            size["n_rows"] = size["n_constraints"]
            size["n_cols"] = size["n_variables"]
            size["n_nonzeros"] = None
            if exact_nonzeros:
                try:
                    size["n_nonzeros"] = int(len(model.constraints.flat))
                except Exception:
                    pass

        nonzeros, rows, cols = size["n_nonzeros"], size["n_rows"], size["n_cols"]
        if nonzeros and rows and cols:
            size["density"] = nonzeros / (rows * cols)
            size["nnz_per_row"] = nonzeros / rows

        self.size.update(size)
        return size

    # --- solver ------------------------------------------------------------
    def solver_stats(self, model) -> dict[str, Any]:
        """Record termination status, objective value and iteration counts."""
        stats: dict[str, Any] = {
            "solver_name": getattr(model, "solver_name", None),
            "status": str(getattr(model, "status", None)),
            "termination_condition": str(getattr(model, "termination_condition", None)),
        }
        try:
            stats["objective_solver"] = float(model.objective.value)
        except Exception:
            stats["objective_solver"] = None

        solver_model = getattr(model, "solver_model", None)
        if solver_model is not None and hasattr(solver_model, "getInfo"):
            try:
                info = solver_model.getInfo()
                stats.update(
                    simplex_iterations=int(info.simplex_iteration_count),
                    ipm_iterations=int(info.ipm_iteration_count),
                    crossover_iterations=int(info.crossover_iteration_count),
                    solver_runtime_s=float(solver_model.getRunTime()),
                    max_primal_infeasibility=float(info.max_primal_infeasibility),
                    max_dual_infeasibility=float(info.max_dual_infeasibility),
                    mip_gap=float(info.mip_gap),
                    mip_nodes=int(info.mip_node_count),
                    highs_model_status=str(
                        solver_model.modelStatusToString(solver_model.getModelStatus())
                    ),
                )
            except Exception as exc:  # pragma: no cover - solver API drift
                stats["solver_stats_error"] = repr(exc)
        elif solver_model is not None and hasattr(solver_model, "IterCount"):
            stats.update(_gurobi_stats(solver_model))

        self.solver.update(stats)
        return stats

    # --- output ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": {name: phase.to_dict() for name, phase in self.phases.items()},
            "totals": {
                "total_wall_s": self.total_wall_s,
                "total_cpu_s": self.total_cpu_s,
                "peak_rss_mb": self.peak_rss_mb,
            },
            "size": dict(self.size),
            "solver": dict(self.solver),
            "environment": self.environment,
        }

    def flat(self) -> dict[str, Any]:
        """One flat record per run, for the cross-run index / comparison table."""
        row: dict[str, Any] = {}
        for name, phase in self.phases.items():
            row[f"{name}_wall_s"] = phase.wall_s
            row[f"{name}_cpu_s"] = phase.cpu_s
            row[f"{name}_rss_delta_mb"] = phase.rss_delta_mb
        row["total_wall_s"] = self.total_wall_s
        row["total_cpu_s"] = self.total_cpu_s
        row["peak_rss_mb"] = self.peak_rss_mb
        for key in ("n_rows", "n_cols", "n_nonzeros", "nnz_per_row", "density"):
            row[key] = self.size.get(key)
        for key in (
            "status",
            "termination_condition",
            "simplex_iterations",
            "ipm_iterations",
            "solver_runtime_s",
            "objective_solver",
        ):
            row[key] = self.solver.get(key)
        return row

    def summary(self) -> str:
        """Human readable one-screen report, printed at the end of a run."""
        size, solver = self.size, self.solver

        def fmt(value, spec=",.0f"):
            if value is None or (isinstance(value, float) and _isnan(value)):
                return "n/a"
            return format(value, spec)

        lines = [
            "",
            "=" * 62,
            " problem size",
            "-" * 62,
            f"  rows (constraints)      {fmt(size.get('n_rows')):>14}",
            f"  columns (variables)     {fmt(size.get('n_cols')):>14}",
            f"  nonzeros                {fmt(size.get('n_nonzeros')):>14}",
            f"  nonzeros per row        {fmt(size.get('nnz_per_row'), '.2f'):>14}",
            "",
            " timings [s]                       wall            cpu",
            "-" * 62,
        ]
        for name, phase in self.phases.items():
            lines.append(f"  {name:<24}{phase.wall_s:>14.3f} {phase.cpu_s:>14.3f}")
        lines += [
            f"  {'TOTAL':<24}{self.total_wall_s:>14.3f} {self.total_cpu_s:>14.3f}",
            f"  {'solver internal':<24}"
            f"{fmt(solver.get('solver_runtime_s'), '.3f'):>14}",
            "",
            " memory [MB]",
            "-" * 62,
            f"  peak working set        {fmt(self.peak_rss_mb, ',.1f'):>14}",
        ]
        for name, phase in self.phases.items():
            lines.append(
                f"  {name + ' delta':<24}{fmt(phase.rss_delta_mb, ',.1f'):>14}"
            )
        lines += [
            "",
            " solver",
            "-" * 62,
            f"  status                  "
            f"{solver.get('status')} / {solver.get('termination_condition')}",
            f"  simplex iterations      {fmt(solver.get('simplex_iterations')):>14}",
            f"  ipm iterations          {fmt(solver.get('ipm_iterations')):>14}",
            f"  objective (solver)      "
            f"{fmt(solver.get('objective_solver'), ',.8g'):>14}",
            "=" * 62,
        ]
        return "\n".join(lines)
