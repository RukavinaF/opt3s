"""One run of the fleet depot EH with opt3s: formulate → solve → postprocess.

Every phase of a run is measured, and the run is stored in a
:class:`~results.ResultStore`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import opt3s
from benchmark import RunMetrics
from results import ResultStore, RunResult
from scenario import Scenario

from . import net_grid
from .hub import build_hub, horizon
from .params import flat, get_params

PYTHON_ROOT = Path(__file__).resolve().parent.parent

#: option profiles per solver; scenario options override them
SOLVER_DEFAULTS: dict[str, dict[str, Any]] = {
    # barrier with crossover; the homogeneous barrier is needed for t <= 0.5 h
    "gurobi": {"Method": 2, "BarHomogeneous": 1},
    # dual simplex: faster than IPM (IPX) + crossover in a pilot on R1-6h and R1-1h
    # (2026-11-03), both to the same optimum; HiPO is not in the highspy 1.15.1 wheel
    "highs": {"solver": "simplex", "simplex_strategy": 1},
}


def formulate(sc: Scenario) -> opt3s.BuiltHub:
    return opt3s.build(
        build_hub(sc),
        horizon(sc),
        n_dl=sc.N_DL,
        r=sc.r,
        scale_constraints=sc.scale_constraints,
        scale_objective=sc.scale_objective,
        extend=None if sc.grid_exchange == "split" else net_grid.extension(sc),
    )


def main(
    scenario: Scenario | None = None,
    metrics: RunMetrics | None = None,
    verbose: bool = True,
    exact_nonzeros_before_solve: bool = False,
) -> RunResult:
    """Formulate and solve one scenario, measuring every phase."""
    sc = scenario if scenario is not None else Scenario()
    # the options actually used are stored with the run, and so enter its fingerprint
    sc = sc.replace(
        solver_options={**SOLVER_DEFAULTS.get(sc.solver_name, {}), **sc.solver_options}
    )
    metrics = metrics if metrics is not None else RunMetrics()
    if verbose:
        print(f"\n>>> {sc}: {sc.N_days} days, t = {sc.t} h, solver {sc.solver_name}")
        if sc.t_comp:
            print(f"    component sampling times [h]: {sc.t_comp}")

    with metrics.phase("formulation"):
        built = formulate(sc)
    metrics.problem_size(built.model, exact_nonzeros=exact_nonzeros_before_solve)

    with metrics.phase("solving"):
        opt3s.solve(
            built,
            solver=sc.solver_name,
            options=sc.solver_options,
            io_api=sc.io_api,
            progress=sc.progress,
        )

    objective: dict[str, Any] = {}
    costs = sizes = amounts = None
    with metrics.phase("postprocessing"):
        metrics.problem_size(built.model)
        metrics.solver_stats(built.model)
        objective.update(
            annuity=built.annuity,
            objective_scale=built.objective_scale,
            solver_value=metrics.solver.get("objective_solver"),
        )
        if str(built.model.status) == "ok":
            res = opt3s.HubResult(built)
            eco = res.economics()
            objective.update(
                value=eco["NPC"], CAPEX=eco["CAPEX"], OPEX=eco["OPEX"], REV=eco["REV"],
                profit=eco["profit"], NPV=eco["NPV"],
            )  # fmt: skip
            costs = res.costs()
            if sc.grid_exchange != "split":  # import, export, OPEX and revenue as in "split"
                amounts = net_grid.split_results(built, sc, objective, costs, res.amounts())
            else:
                amounts = res.amounts()
            costs["C_total"] = costs[list(opt3s.COST_TERMS)].sum(axis=1)
            sizes = res.sizes()

    if verbose:
        print(metrics.summary())
        if "NPV" in objective:
            print(f"  NPV                     {objective['NPV']:>14,.0f} EUR")

    parameters = {
        **flat(get_params(sc.params)),
        "horizon.start": sc.start_date.isoformat(),
        "horizon.n_days": sc.N_days,
    }
    return RunResult(
        scenario=sc,
        model=built.model,
        components={},
        metrics=metrics,
        objective=objective,
        parameters={k: list(v) if isinstance(v, tuple) else v for k, v in parameters.items()},
        costs=costs,
        sizes=sizes,
        amounts=amounts,
        extra={"model": f"opt3s {opt3s.__version__}"},
    )


def run_scenarios(
    scenarios: Iterable[Scenario],
    store: ResultStore | None = None,
    save_solution: bool = True,
    isolate: bool = True,
    **main_kwargs: Any,
) -> pd.DataFrame:
    """Run scenarios back to back and store every result.

    ``isolate=True`` runs each scenario in a fresh process (``depot.run_one``),
    which the memory figures require; see the legacy ``run_scenarios``.
    """
    store = store if store is not None else ResultStore(PYTHON_ROOT / "results")
    for sc in scenarios:
        if not isolate:
            try:
                result = main(sc, **main_kwargs)
            except Exception as exc:  # keep a long sweep going
                print(f"!!! scenario {sc} failed: {exc!r}")
                continue
            print(f"  saved to {store.save(result, save_solution=save_solution)}")
            continue
        job = {
            "scenario": sc.to_dict(),
            "store_root": str(store.root),
            "save_solution": save_solution,
            **main_kwargs,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "depot.run_one"],
            input=json.dumps(job, default=str),
            text=True,
            cwd=PYTHON_ROOT,
        )
        if completed.returncode != 0:
            print(f"!!! scenario {sc} failed: exit code {completed.returncode}")
    return store.index()
