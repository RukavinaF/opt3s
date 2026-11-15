"""Command line of the case study, see :mod:`depot`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# the support modules (scenario, results, benchmark) live next to this package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import opt3s  # noqa: E402
from opt3s.solve import ENV_VAR  # noqa: E402
from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from . import scenarios  # noqa: E402
from .hub import build_hub  # noqa: E402
from .runner import PYTHON_ROOT, run_scenarios  # noqa: E402


def _cmd_list(args: argparse.Namespace) -> None:
    print("installed solvers:", ", ".join(opt3s.available_solvers()))
    print("default solver:   ", opt3s.resolve_solver(), f"(set {ENV_VAR} to change)")
    for name, f in scenarios.SETS.items():
        print(f"\n{name}:")
        for sc in f():
            print(f"  {sc.name:<34} {sc.fingerprint}")


def _cmd_run(args: argparse.Namespace) -> None:
    selected = scenarios.get(args.scenarios)
    changes = {}
    if args.solver:
        changes["solver_name"] = opt3s.resolve_solver(args.solver)
    if args.days:
        changes["end_date"] = selected[0].start_date + pd.Timedelta(days=args.days - 1)
    selected = [sc.replace(**changes) for sc in selected]
    store = ResultStore(PYTHON_ROOT / args.store)
    index = run_scenarios(
        selected, store=store, save_solution=not args.no_solution, isolate=not args.in_process
    )
    print(index.tail(len(selected))[["run_id", "obj_NPV", "total_wall_s", "status"]].to_string())


def _cmd_describe(args: argparse.Namespace) -> None:
    hub = build_hub(Scenario(hts_balance=args.hts_balance))
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print(opt3s.describe(hub).to_string())
        print()
        print(opt3s.describe(hub, custom_parts=True).to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="scenario sets and installed solvers")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("run", help="run a scenario set or a single scenario")
    p.add_argument("scenarios", help="R1, R2, R3, all, or a scenario name such as R1-1h")
    p.add_argument("--solver", help=f"solver name; default: ${ENV_VAR} or gurobi/highs")
    p.add_argument("--days", type=int, help="shorten the horizon to this many days (smoke tests)")
    p.add_argument("--store", default="results", help="result directory below the repository root")
    p.add_argument("--in-process", action="store_true", help="no subprocess per scenario")
    p.add_argument("--no-solution", action="store_true", help="do not write solution.nc")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("describe", help="declarative tables of the components (S5, S6)")
    p.add_argument("--hts-balance", default="ineq", choices=("ineq", "eq"))
    p.set_defaults(func=_cmd_describe)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
