"""R2 -- influence of mixed sampling times, solved with HiGHS.

The components are divided into fast, medium, and slow groups (Table
"Groups of components"); 21 combinations of the three sampling times.
Run from the repository root::

    python -m depot.studies.r2 run                # all 21 scenarios, coarse to fine
    python -m depot.studies.r2 run --only R2-1h-2h-6h
    python -m depot.studies.r2 tables             # tables/R2_*.tex for paper and SI

As in :mod:`depot.studies.r1`, every scenario is solved in its own process and
stored in ``results/``.  ``tables`` writes

* ``tables/R2_summary.tex``       short table for the paper;
* ``tables/S_R2_computation.tex`` LP metrics and performance (Supplementary Table S9);
* ``tables/S_R2_results.tex``     sizes, annual amounts, economics (Supplementary Table S10);
* ``tables/R2_highs.csv``         all numbers, one column per scenario.

``--days n`` shortens the horizon (smoke tests only; such runs have their own
fingerprints and never mix with the full-year runs).
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from .. import scenarios as sets  # noqa: E402
from ..report import MANUSCRIPT, _block_rows, _longtable, _table, fmt  # noqa: E402
from ..runner import PYTHON_ROOT, SOLVER_DEFAULTS, run_scenarios  # noqa: E402
from .r1 import SOLVER, SUMMARY_COLUMNS as R1_COLUMNS, Repeats, collect, csv  # noqa: E402


#: Scenarios solved with other options than :data:`depot.runner.SOLVER_DEFAULTS`.
#:
#: With dual simplex, R2-0.25h-2h-2h stalled for more than 18 h (2026-11-05) in the
#: primal clean-up after the dual simplex: the objective stopped changing while
#: about 8,000 dual infeasibilities remained.  The remaining 0.25 h scenarios were
#: then started with ``solver="hipo"`` (2026-11-07), but HiPO is not compiled into
#: the highspy 1.15.1 wheel ("features are unavailable: amd, blas, metis, rcm"), so
#: HiGHS fell back to dual simplex.  Four of them were solved to optimality that
#: way -- they are dual simplex runs, only stored with these options -- while
#: R2-0.25h-2h-2h stalled again, at the same objective, until the 12 h time limit.
#: It is therefore solved with the interior-point method IPX without crossover,
#: which matched dual simplex on R2-1h-2h-6h (NPV within 7e-7) in a pilot.
_HIPO_FALLBACK = {"solver": "hipo", "run_crossover": "off", "time_limit": 43200}
_IPX = {"solver": "ipm", "run_crossover": "off", "time_limit": 43200}
ALT_OPTIONS: dict[str, dict[str, object]] = {
    "R2-0.25h-1h-2h": _HIPO_FALLBACK,
    "R2-0.25h-1h-3h": _HIPO_FALLBACK,
    "R2-0.25h-1h-4h": _HIPO_FALLBACK,
    "R2-0.25h-1h-6h": _HIPO_FALLBACK,
    "R2-0.25h-2h-2h": _IPX,
}
#: scenarios actually solved with another method, marked in the tables
ALT_SCENARIOS = ("R2-0.25h-2h-2h",)
ALT_MARK = "$^\\dagger$"
ALT_NOTE = ("The scenario marked with $^\\dagger$ is solved with the interior-point method "
            "without crossover instead of the dual simplex method.")


def key(sc: Scenario) -> tuple[float, float, float]:
    """(t_fast, t_med, t_slow)."""
    return sc.t, sc.t_of("PVP"), sc.t_of("ELYS")


def scenarios(days: int | None = None) -> list[Scenario]:
    """The 21 R2 scenarios with HiGHS, ordered fine to coarse."""
    out = []
    for sc in sorted(sets.R2(), key=key):
        # as depot.runner.main merges them, so that the fingerprints match the stored runs
        options = {**SOLVER_DEFAULTS[SOLVER], **ALT_OPTIONS.get(sc.name, {})}
        sc = sc.replace(solver_name=SOLVER, solver_options=options)
        if days:
            sc = sc.replace(end_date=sc.start_date + timedelta(days=days - 1))
        out.append(sc)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(store: ResultStore, only: list[str] | None, days: int | None) -> None:
    # coarse fast sampling first (minutes), the 0.25 h ones last (hours)
    selected = sorted((sc for sc in scenarios(days) if not only or sc.name in only),
                      key=lambda sc: (-sc.t, -sc.t_of("PVP"), -sc.t_of("ELYS")))  # fmt: skip
    print(f"\n##### R2 with {SOLVER}: {len(selected)} scenarios")
    run_scenarios(selected, store=store)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def computation_table(runs: dict[str, Repeats], scs: list[Scenario]) -> str:
    cols = [
        ("Variables", "cols", 0),
        ("Constraints", "rows", 0),
        ("Nonzeros", "nonzeros", 0),
        ("\\makecell[tl]{Simplex \\\\ iterations}", "simplex", 0),
        ("\\makecell[tl]{IPM \\\\ iterations}", "barrier", 0),
        ("\\makecell[tl]{Formulation \\\\ time [s]}", "formulation", 1),
        ("\\makecell[tl]{Solving \\\\ time [s]}", "solving", 1),
        ("\\makecell[tl]{Total \\\\ time [s]}", "total", 1),
        ("\\makecell[tl]{Peak \\\\ memory [MB]}", "memory", 1),
    ]
    body, last = [], None
    for sc in scs:
        r = runs.get(sc.name)
        if r is None:
            continue
        if last is not None and sc.t != last:
            body.append("\\midrule")
        last = sc.t
        name = sc.name + (ALT_MARK if sc.name in ALT_SCENARIOS else "")
        body.append(" & ".join([name, *(fmt(r.perf(k), d) for _, k, d in cols)]) + " \\\\")
    header = [" & ".join(["Scenario", *(h for h, _, _ in cols)])]
    return _table("l" + "r" * len(cols),
                  "LP metrics and computational performance for mixed sampling times "
                  "(scenarios R2, solver HiGHS). " + ALT_NOTE,
                  "tab:S_R2_computation", header, body)  # fmt: skip


def results_table(runs: dict[str, Repeats], scs: list[Scenario]) -> str:
    fasts = sorted({sc.t for sc in scs})
    med_slow = sorted({(sc.t_of("PVP"), sc.t_of("ELYS")) for sc in scs})
    by_key = {key(sc): runs.get(sc.name) for sc in scs}
    n = len(med_slow) + 2
    header = [
        f" & & \\multicolumn{{{len(med_slow)}}}{{c}}{{$t_\\mathrm{{med}}$-$t_\\mathrm{{slow}}$}} \\\\\n"
        f"    \\cmidrule(l){{3-{n}}}\n    "
        + " & ".join(["Quantity", "Unit", *(f"{m:g}h-{s:g}h" for m, s in med_slow)])
    ]
    body = []
    for i, f in enumerate(fasts):
        if i:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{n}}}{{l}}{{\\textbf{{$t_\\mathrm{{fast}}$ = {f:g}~h}}}} \\\\")
        body.append("\\midrule")
        body += _block_rows([by_key.get((f, m, s)) for m, s in med_slow])
    return _longtable("l l " + "r " * len(med_slow),
                      "Optimal sizes, annual amounts, and economic indicators for mixed sampling "
                      "times (scenarios R2).",
                      "tab:S_R2_results", header, body, n)  # fmt: skip


#: the columns of Table 2 (R1 summary) with the battery added after the H2 tank
SUMMARY_COLUMNS = [
    *R1_COLUMNS[:5],
    ("\\makecell[tr]{Battery \\\\ {[}kWh{]}}", lambda r: r.size("BATT"), 1),
    *R1_COLUMNS[5:],
]


def summary_table(runs: dict[str, Repeats], scs: list[Scenario]) -> str:
    """One row per scenario, grouped by t_fast; the columns of Table 2 plus the battery."""
    body, last_fast = [], None
    for sc in scs:
        r = runs.get(sc.name)
        if r is None:
            continue
        if last_fast is not None and sc.t != last_fast:
            body.append("\\addlinespace")
        last_fast = sc.t
        cells = [fmt(get(r), d) for _, get, d in SUMMARY_COLUMNS]
        if sc.name in ALT_SCENARIOS:  # mark the solving time, without shifting the column
            cells[1] += f"\\rlap{{{ALT_MARK}}}"
        f, m, s = key(sc)
        body.append(" & ".join([fmt(f, 2), f"{m:g}", f"{s:g}", *cells]) + " \\\\")
    header = [" & ".join(["$t_\\mathrm{fast}$ [h]", "$t_\\mathrm{med}$ [h]", "$t_\\mathrm{slow}$ [h]",
                          *(h for h, _, _ in SUMMARY_COLUMNS)])]  # fmt: skip
    caption = (
        "Computational effort, main optimal sizes, and economic indicators for mixed sampling "
        "times (scenarios R2). Complete results are given in Supplementary Tables S9 and S10. "
        + ALT_NOTE
    )
    return _table("r" * (3 + len(SUMMARY_COLUMNS)), caption, "tab:R2_summary", header, body)


def tables(store: ResultStore, out: Path, days: int | None) -> None:
    scs = scenarios(days)
    runs = collect(store, scs)
    if not runs:
        raise SystemExit(f"no {SOLVER} runs of R2 in {store.root}")
    missing = [sc.name for sc in scs if sc.name not in runs]
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("R2_summary", summary_table(runs, scs)),
                       ("S_R2_computation", computation_table(runs, scs)),
                       ("S_R2_results", results_table(runs, scs))):  # fmt: skip
        (folder / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
        print("wrote", folder / f"{name}.tex")
    frame = csv({sc.name: runs[sc.name] for sc in scs if sc.name in runs})
    frame.to_csv(folder / "R2_highs.csv", lineterminator="\n")  # same file on every OS
    print("wrote", folder / "R2_highs.csv")
    if missing:
        print("MISSING:", ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot.studies.r2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--store", default="results")
        p.add_argument("--days", type=int, help="shorten the horizon (smoke tests only)")
        if name == "run":
            p.add_argument("--only", nargs="+", help="scenario names, e.g. R2-1h-2h-6h")
        else:
            p.add_argument("--out", default=str(MANUSCRIPT))
    args = parser.parse_args(argv)
    store = ResultStore(PYTHON_ROOT / args.store)
    if args.command == "run":
        run(store, args.only, args.days)
    else:
        tables(store, Path(args.out), args.days)


if __name__ == "__main__":
    main()
