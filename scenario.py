"""Scenario configuration for the energy hub case study.

A :class:`Scenario` holds every knob that is allowed to differ between runs of
``case_study.main``: the optimisation horizon, the economic parameters, the
model formulation variants, and the solver settings.  Everything else in the
model is fixed, so a scenario -- together with the code revision recorded by
:mod:`benchmark` -- fully identifies a run.

Defaults reproduce the model exactly as it was formulated before the scenario
system was introduced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from itertools import product
from typing import Any, Iterator


def _default_solver() -> str:
    """``OPT3S_SOLVER`` if set, else the first installed of gurobi and highs."""
    from opt3s.solve import resolve_solver  # imported late: linopy is heavy

    return resolve_solver()


def _jsonable(value: Any) -> Any:
    """Convert a scenario field into something ``json.dumps`` accepts."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class Scenario:
    """One configuration of the energy hub optimisation."""

    # --- identity -----------------------------------------------------------
    name: str = "base"
    notes: str = ""
    tags: tuple[str, ...] = ()

    # --- horizon ------------------------------------------------------------
    t: float = 1.0  # time step [h]
    # per component time step [h], keyed by acronym (e.g. {"ELYS": 2.0});
    # components not listed use ``t``
    t_comp: dict[str, float] = field(default_factory=dict)
    start_date: datetime = datetime(2025, 1, 1)
    end_date: datetime = datetime(2025, 12, 31)

    # --- economics ----------------------------------------------------------
    N_DL: int = 20  # design lifetime [a]
    r: float = 0.03  # discount rate [-]

    # --- model formulation variants ----------------------------------------
    scale_constraints: bool = True  # row-wise normalisation of constraints
    scale_objective: bool = True  # normalisation of the objective coefficients
    egc_equality: bool = True  # grid node: balance as == (True) or >= (False)
    # heating system (opt3s model only): "ineq" -- all inputs DO, balance
    # sum(u) >= y; "eq" -- waste-heat inputs BO, balance sum(u) == y
    hts_balance: str = "ineq"
    # electricity exchange with the grid (opt3s model only): "split" -- import
    # and export as two non-negative ports; "net" -- one signed net power as an
    # intermediate variable with a cost variable, as in previous work; "free" --
    # the same with the net power as a decision variable (depot.net_grid)
    grid_exchange: str = "split"

    # --- component data (opt3s model only) ----------------------------------
    # overrides of depot.params.PARAMS, e.g. {"HSLS.c_rev": 8.0}
    params: dict[str, Any] = field(default_factory=dict)

    # --- solver -------------------------------------------------------------
    # default: environment variable OPT3S_SOLVER, else gurobi, else highs
    solver_name: str = field(default_factory=lambda: _default_solver())
    io_api: str = "direct"
    solver_options: dict[str, Any] = field(default_factory=dict)
    progress: bool = True

    def __post_init__(self) -> None:
        # normalise numeric types so that e.g. t=1 and t=1.0 fingerprint alike
        self.t = float(self.t)
        self.t_comp = {str(k): float(v) for k, v in sorted(self.t_comp.items())}
        self.r = float(self.r)
        self.N_DL = int(self.N_DL)
        self.tags = tuple(self.tags)
        self.params = dict(sorted(self.params.items()))
        if self.hts_balance not in ("eq", "ineq"):
            raise ValueError(f"hts_balance must be 'eq' or 'ineq', not {self.hts_balance!r}")
        if self.grid_exchange not in ("split", "net", "free"):
            raise ValueError(
                f"grid_exchange must be 'split', 'net' or 'free', not {self.grid_exchange!r}"
            )

    # --- derived ------------------------------------------------------------
    def t_of(self, component: str) -> float:
        """Time step of ``component`` [h], falling back to the general ``t``."""
        return self.t_comp.get(component, self.t)

    @property
    def N_days(self) -> int:
        """Number of days in the horizon, endpoints included."""
        return (self.end_date - self.start_date).days + 1

    @property
    def N_h(self) -> int:
        """Number of time steps in the horizon at the general time step ``t``."""
        return int(24 * self.N_days / self.t)

    @property
    def lamda(self) -> float:
        """Annualisation factor scaling horizon costs to a full year."""
        return 365 / self.N_days

    # --- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = {k: _jsonable(v) for k, v in asdict(self).items()}
        d.update(N_days=self.N_days, N_h=self.N_h, lamda=self.lamda)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        for key in ("start_date", "end_date"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = datetime.fromisoformat(kwargs[key])
        if "tags" in kwargs:
            kwargs["tags"] = tuple(kwargs["tags"])
        return cls(**kwargs)

    def replace(self, **changes: Any) -> "Scenario":
        """Return a copy with ``changes`` applied (``dataclasses.replace``)."""
        return replace(self, **changes)

    # --- identity -----------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Short hash over everything that changes the problem being solved.

        ``name``, ``notes``, ``tags`` and ``progress`` are excluded, so two
        scenarios that differ only in labelling share a fingerprint.  An empty
        ``t_comp`` is left out as well, so runs from before it existed keep
        their fingerprints; the same holds for ``hts_balance``, ``grid_exchange``
        and ``params`` at their defaults.
        """
        payload = {
            k: v
            for k, v in self.to_dict().items()
            if k not in ("name", "notes", "tags", "progress")
        }
        if not payload["t_comp"]:
            del payload["t_comp"]
        if payload["hts_balance"] == "ineq":
            del payload["hts_balance"]
        if payload["grid_exchange"] == "split":
            del payload["grid_exchange"]
        if not payload["params"]:
            del payload["params"]
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def __str__(self) -> str:
        return f"{self.name} [{self.fingerprint}]"


def scenario_grid(base: Scenario, **axes: list) -> Iterator[Scenario]:
    """Yield the cartesian product of ``axes`` applied to ``base``.

    Each generated scenario is named ``<base.name>__<field>=<value>...``.

    >>> list(scenario_grid(Scenario(), t=[1, 2], egc_equality=[True, False]))
    """
    keys = list(axes)
    for combo in product(*(axes[k] for k in keys)):
        changes = dict(zip(keys, combo))
        suffix = "__".join(f"{k}={v}" for k, v in changes.items())
        yield base.replace(name=f"{base.name}__{suffix}", **changes)
