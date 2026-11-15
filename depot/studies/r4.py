"""R4 -- two non-negative variables vs. a single signed variable, solved with HiGHS.

The electricity exchange with the grid is formulated in three ways:

* ``split``  import and export as two non-negative ports (the article);
* ``net``    one signed net power as an intermediate variable, with a cost
             variable bounded from below by two lines (previous work);
* ``free``   the same with the net power as a decision variable.

Every comparison is a group of three scenarios that differ only in this
formulation.  The ``split`` brother is an R2 scenario, whose stored run is used.
See :mod:`depot.net_grid` for the formulation.  Run from the repository root::

    python -m depot.studies.r4 run                  # the six new runs, net first
    python -m depot.studies.r4 run --variant net    # only one of the two variants
    python -m depot.studies.r4 compare              # brothers side by side, on screen
    python -m depot.studies.r4 simultaneous         # import and export at the same instant?
    python -m depot.studies.r4 presolve             # LP size after the presolve of HiGHS
    python -m depot.studies.r4 tables               # tables/R4_*.tex for paper and SI

``--days n`` (smoke tests) shortens the horizon and also solves the ``split``
brothers, since shortened R2 runs are not stored.

``tables`` writes

* ``tables/R4_summary.tex``       one row per scenario and formulation, for the paper;
* ``tables/S_R4_computation.tex`` LP metrics and performance of every run;
* ``tables/S_R4_results.tex``     sizes, annual amounts, economics of every run;
* ``tables/R4_highs.csv``         all numbers, one column per run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from .. import scenarios as sets  # noqa: E402
from ..report import MANUSCRIPT, NA, _block_rows, _longtable, _table, fmt  # noqa: E402
from ..runner import PYTHON_ROOT, formulate, run_scenarios  # noqa: E402
from . import r2  # noqa: E402
from .r1 import Repeats, collect, csv  # noqa: E402
from .r3 import (  # noqa: E402
    TIME_LIMIT,
    _small,
    _times,
    _unique,
    egc_release,
    max_size_difference,
    npv_difference,
)

VARIANTS = ("split", "net", "free")
#: formulation of the grid power, as named in the tables
LABEL = {
    "split": "two non-negative variables",
    "net": "signed intermediate variable",
    "free": "signed decision variable",
}
SHORT = {"split": "Two non-negative", "net": "Signed intermediate", "free": "Signed decision"}
TWO_ROWS = {"split": ("Two", "non-negative"), "net": ("Signed", "intermediate"), "free": ("Signed", "decision")}
#: powers below this value [kW] count as zero (solver feasibility tolerance)
POWER_TOL = 1e-4


@dataclass(frozen=True)
class Group:
    """Three scenarios that differ only in the formulation of the grid exchange."""

    times: str  # "1h-1h-4h"
    split: Scenario  # the R2 scenario, whose run is stored
    net: Scenario
    free: Scenario

    def of(self, variant: str) -> Scenario:
        return getattr(self, variant)

    @property
    def all(self) -> list[Scenario]:
        return [self.of(v) for v in VARIANTS]


def groups(days: int | None = None) -> list[Group]:
    """All groups, with the solver and the solver options of the R2 counterpart."""
    r2s = {sc.name: sc for sc in r2.scenarios(days)}
    found: dict[str, dict[str, Scenario]] = {}
    for sc in sets.R4():
        base = r2s["R2-" + _times(sc)]
        options = dict(base.solver_options)
        if "time_limit" in options:  # the 0.25 h scenarios
            options["time_limit"] = TIME_LIMIT
        sc = sc.replace(solver_name=base.solver_name, solver_options=options, end_date=base.end_date)
        found.setdefault(_times(sc), {"split": base})[sc.grid_exchange] = sc
    return [Group(times, **scs) for times, scs in found.items()]


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(store: ResultStore, variant: str, days: int | None) -> None:
    gs = groups(days)
    variants = [v for v in ("net", "free") if variant in ("all", v)]
    selected = [g.of(v) for v in variants for g in gs]  # one variant after the other
    if days:
        selected = [g.split for g in gs] + selected
    print(f"\n##### R4, variant {variant}: {len(selected)} scenarios")
    for sc in selected:
        print(f"  {sc.name:<24} {sc.fingerprint}  {sc.solver_options}")
    run_scenarios(selected, store=store)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
PERF_ROWS = [("variables", "cols", 0), ("constraints", "rows", 0), ("nonzeros", "nonzeros", 0),
             ("simplex iterations", "simplex", 0), ("formulation [s]", "formulation", 1),
             ("solving [s]", "solving", 1), ("total [s]", "total", 1),
             ("peak memory [MB]", "memory", 1)]  # fmt: skip


def compare(store: ResultStore, days: int | None) -> None:
    gs = groups(days)
    runs = collect(store, _unique([sc for g in gs for sc in g.all]))
    for g in gs:
        rs = {v: runs.get(g.of(v).name) for v in VARIANTS}
        print(f"\n===== {g.times}:  " + "   ".join(f"{v} {g.of(v).name}" for v in VARIANTS))
        missing = [g.of(v).name for v in VARIANTS if rs[v] is None]
        if missing:
            print("  missing:", ", ".join(missing))
        base = rs["split"]
        present = [v for v in VARIANTS if rs[v] is not None]
        print(f"  {'':<22}" + "".join(f"{v:>16}" for v in present)
              + "".join(f"{v + '/split':>12}" for v in present[1:] if base is not None))  # fmt: skip
        for label, key, dec in PERF_ROWS:
            values = [rs[v].perf(key) for v in present]
            ratios = [rs[v].perf(key) / base.perf(key) for v in present[1:]] if base is not None else []
            print(f"  {label:<22}" + "".join(f"{x:>16,.{dec}f}" for x in values)
                  + "".join(f"{x:>12.3f}" for x in ratios))  # fmt: skip
        for label in ("CAPEX", "OPEX", "REV", "profit", "NPV"):
            values = [rs[v].economics()[label] for v in present]
            print(f"  {label:<22}" + "".join(f"{x:>16,.0f}" for x in values))
        for label, port in (("import [MWh]", "ESRC.y1"), ("export [MWh]", "ESNK.u1")):
            print(f"  {label:<22}" + "".join(f"{rs[v].amount(port) / 1e3:>16,.2f}" for v in present))
        if base is None:
            continue
        for v in present[1:]:
            gap = rs[v].objective.get("net_cost_gap", float("nan"))
            print(f"  {v:<6} vs split: NPV {npv_difference(base, rs[v]):+.1e} (relative), "
                  f"largest size difference {max_size_difference(base, rs[v]):.1e}, "
                  f"cost variable above the cost by {gap:.2e} EUR/year")  # fmt: skip


# --------------------------------------------------------------------------- #
# import and export at the same instant, in the runs with two ports
# --------------------------------------------------------------------------- #
def simultaneous(store: ResultStore) -> None:
    gs = groups()
    runs = collect(store, [g.split for g in gs])
    for g in gs:
        r = runs.get(g.split.name)
        if r is None:
            print(f"\n{g.split.name}: no run")
            continue
        f = egc_release(store, r)
        both = np.minimum(f["import"], f["export"])
        print(f"\n===== {g.split.name}")
        print(f"  instants                          {len(f):>12,}")
        print(f"  instants with import              {int((f['import'] > POWER_TOL).sum()):>12,}")
        print(f"  instants with export              {int((f['export'] > POWER_TOL).sum()):>12,}")
        print(f"  instants with both                {int((both > POWER_TOL).sum()):>12,}")
        print(f"  largest min(import, export) [kW]  {float(both.max()):>12.2e}")


# --------------------------------------------------------------------------- #
# size of the LP after the presolve of HiGHS (nothing is solved)
# --------------------------------------------------------------------------- #
def presolve(days: int | None) -> None:
    """Variables, constraints and nonzeros that are left for the simplex method."""
    for g in groups(days):
        print(f"\n===== {g.times}")
        print(f"  {'':<8}{'variables':>12}{'constraints':>14}{'nonzeros':>12}   relative to split")
        base = None
        for v in VARIANTS:
            h = formulate(g.of(v)).model.to_highspy()
            h.setOptionValue("output_flag", False)
            h.presolve()
            lp = h.getPresolvedLp()
            size = np.array([lp.num_col_, lp.num_row_, len(lp.a_matrix_.value_)], float)
            base = size if base is None else base
            ratios = "  ".join(f"{x:.3f}" for x in size / base)
            print(f"  {v:<8}{size[0]:>12,.0f}{size[1]:>14,.0f}{size[2]:>12,.0f}   {ratios}")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _differences(base: Repeats | None, r: Repeats | None, variant: str) -> list[str]:
    if variant == "split" or base is None or r is None:
        return [NA, NA]
    return [_small(100 * npv_difference(base, r)), _small(100 * max_size_difference(base, r))]


def summary_table(gs: list[Group], runs: dict[str, Repeats]) -> str:
    """One row per scenario and formulation."""
    cols = [
        ("Variables", "cols", 0),
        ("Constraints", "rows", 0),
        ("Nonzeros", "nonzeros", 0),
        ("\\makecell[tr]{Simplex \\\\ iterations}", "simplex", 0),
        ("\\makecell[tr]{Solving \\\\ time [s]}", "solving", 0),
        ("\\makecell[tr]{Peak \\\\ memory [MB]}", "memory", 0),
    ]
    body = []
    for n, g in enumerate(gs):
        if n:
            body.append("\\addlinespace")
        base = runs.get(g.split.name)
        for i, v in enumerate(VARIANTS):
            r = runs.get(g.of(v).name)
            cells = [fmt(r.perf(k), d) if r is not None else NA for _, k, d in cols]
            body.append(" & ".join([g.times if i == 0 else "", LABEL[v], *cells,
                                    *_differences(base, r, v)]) + " \\\\")  # fmt: skip
    header = [" & ".join(["\\makecell[tl]{Sampling \\\\ times}", "Grid power", *(h for h, _, _ in cols),
                          "\\makecell[tr]{$\\Delta$NPV \\\\ {[\\%]}}",
                          "\\makecell[tr]{$\\Delta S_\\mathrm{max}$ \\\\ {[\\%]}}"])]  # fmt: skip
    caption = (
        "LP size, computational effort, and optimal solutions with the grid power as two "
        "non-negative variables and as a single signed variable (scenarios R4). "
        "$\\Delta$NPV is the relative difference of the NPV and $\\Delta S_\\mathrm{max}$ the "
        "largest relative difference of a component size, both with respect to the formulation "
        "with two non-negative variables. "
        "Complete results are given in Supplementary Tables S14 and S15."
    )
    return _table("ll" + "r" * (len(cols) + 2), caption, "tab:R4_summary", header, body)


def computation_table(gs: list[Group], runs: dict[str, Repeats]) -> str:
    cols = [
        ("Variables", "cols", 0),
        ("Constraints", "rows", 0),
        ("Nonzeros", "nonzeros", 0),
        ("\\makecell[tl]{Simplex \\\\ iterations}", "simplex", 0),
        ("\\makecell[tl]{Solving \\\\ time [s]}", "solving", 1),
        ("\\makecell[tl]{Peak \\\\ memory [MB]}", "memory", 1),
    ]
    body = []
    for n, g in enumerate(gs):
        if n:
            body.append("\\addlinespace")
        for v in VARIANTS:
            sc = g.of(v)
            r = runs.get(sc.name)
            cells = [fmt(r.perf(k), d) if r is not None else NA for _, k, d in cols]
            body.append(" & ".join([sc.name, SHORT[v], *cells]) + " \\\\")
    header = [" & ".join(["Scenario", "Grid power", *(h for h, _, _ in cols)])]
    formulation = [r.perf("formulation") for r in runs.values()]
    caption = (
        "LP metrics and computational performance with the grid power as two non-negative "
        "variables, as a signed intermediate variable, and as a signed decision variable "
        "(scenarios R4, solver HiGHS). "
        "Scenarios named R2 are those of Supplementary Table S9. "
        f"The formulation time is {fmt(min(formulation), 1)}--{fmt(max(formulation), 1)}~s in all scenarios."
    )
    return _table("ll" + "r" * len(cols), caption, "tab:S_R4_computation", header, body)


def results_table(gs: list[Group], runs: dict[str, Repeats], per_block: int = 2) -> str:
    """Sizes, amounts and economics; ``per_block`` groups side by side in every block."""
    nv = len(VARIANTS)
    n = nv * per_block + 2
    header = [f"Quantity & Unit & \\multicolumn{{{nv * per_block}}}{{c}}"
              "{Sampling times and formulation of the grid power}"]  # fmt: skip
    body = []
    for b in range(0, len(gs), per_block):
        block = gs[b : b + per_block]
        if b:
            body.append("\\newpage")  # every block on its own page
        pad = [""] * (nv * (per_block - len(block)))
        titles = [f"\\multicolumn{{{nv}}}{{c}}{{{g.times}}}" for g in block]
        body.append(" & ".join(["", "", *titles, *pad]) + " \\\\")
        for row in range(2):  # the formulation on two rows, to keep the columns narrow
            names = [TWO_ROWS[v][row] for v in VARIANTS] * len(block)
            body.append(" & ".join(["", "", *names, *pad]) + " \\\\")
        body.append("\\midrule")
        columns = [runs.get(g.of(v).name) for g in block for v in VARIANTS]
        rows = _block_rows(columns + [None] * len(pad))
        if pad:  # empty cells instead of "--" in the unused columns
            rows = [_blank_tail(line, len(pad)) for line in rows]
        body += [line.replace(f"\\multicolumn{{{len(columns) + len(pad) + 2}}}", f"\\multicolumn{{{n}}}")
                 for line in rows]  # fmt: skip
    caption = (
        "Optimal sizes, annual amounts, and economic indicators with the grid power as two "
        "non-negative variables, as a signed intermediate variable, and as a signed decision "
        "variable (scenarios R4). "
        "For the signed variable, the grid import and export are its positive and negative part, "
        "and the cost of the grid energy is divided into OPEX and revenue accordingly."
    )
    return _longtable("l l " + "r " * (nv * per_block), caption, "tab:S_R4_results", header, body, n)


def _blank_tail(line: str, n: int) -> str:
    if not line.endswith(" \\\\") or " & " not in line:
        return line
    cells = line[: -len(" \\\\")].split(" & ")
    return " & ".join(cells[:-n] + [""] * n) + " \\\\"


def tables(store: ResultStore, out: Path, days: int | None) -> None:
    gs = groups(days)
    scs = _unique([sc for g in gs for sc in g.all])
    runs = collect(store, scs)
    if not runs:
        raise SystemExit(f"no runs of R4 in {store.root}")
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("R4_summary", summary_table(gs, runs)),
                       ("S_R4_computation", computation_table(gs, runs)),
                       ("S_R4_results", results_table(gs, runs))):  # fmt: skip
        (folder / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
        print("wrote", folder / f"{name}.tex")
    frame = csv({sc.name: runs[sc.name] for sc in scs if sc.name in runs})
    frame.to_csv(folder / "R4_highs.csv", lineterminator="\n")  # same file on every OS
    print("wrote", folder / "R4_highs.csv")
    missing = [sc.name for sc in scs if sc.name not in runs]
    if missing:
        print("MISSING:", ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot.studies.r4", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare", "simultaneous", "presolve", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--store", default="results")
        if name != "simultaneous":
            p.add_argument("--days", type=int, help="shorten the horizon (smoke tests only)")
        if name == "run":
            p.add_argument("--variant", default="all", choices=("all", "net", "free"))
        if name == "tables":
            p.add_argument("--out", default=str(MANUSCRIPT))
    args = parser.parse_args(argv)
    store = ResultStore(PYTHON_ROOT / args.store)
    if args.command == "run":
        run(store, args.variant, args.days)
    elif args.command == "compare":
        compare(store, args.days)
    elif args.command == "simultaneous":
        simultaneous(store)
    elif args.command == "presolve":
        presolve(args.days)
    else:
        tables(store, Path(args.out), args.days)


if __name__ == "__main__":
    main()
