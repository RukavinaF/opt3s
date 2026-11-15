"""Solver selection and solving.

The solver is resolved in this order:

1. the ``solver`` argument;
2. the environment variable ``OPT3S_SOLVER``;
3. the first installed solver of :data:`PREFERRED`.

Any solver linopy supports can be named; :func:`available_solvers` lists the
installed ones.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import linopy

from .build import BuiltHub

PREFERRED = ("gurobi", "highs")
ENV_VAR = "OPT3S_SOLVER"


def available_solvers() -> list[str]:
    return list(linopy.available_solvers)


def resolve_solver(solver: str | None = None) -> str:
    name = solver or os.environ.get(ENV_VAR)
    installed = available_solvers()
    if name:
        if name not in installed:
            raise ValueError(f"solver {name!r} is not installed; available: {installed}")
        return name
    for candidate in PREFERRED:
        if candidate in installed:
            return candidate
    if installed:
        return installed[0]
    raise RuntimeError("no LP solver is installed")


def solve(
    built: BuiltHub,
    solver: str | None = None,
    options: Mapping[str, Any] | None = None,
    io_api: str = "direct",
    progress: bool = True,
) -> tuple[str, str]:
    """Solve the LP; returns linopy's ``(status, termination_condition)``."""
    name = resolve_solver(solver)
    return built.model.solve(
        solver_name=name, io_api=io_api, progress=progress, **dict(options or {})
    )
