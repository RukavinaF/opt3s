"""R1 -- influence of the sampling time, solved with HiGHS.

The fleet depot EH is optimized over one year with one sampling time for all
components, from 0.25 h to 6 h.  Run from the repository root::

    python -m depot.studies.r1 run                # all seven scenarios, coarse to fine
    python -m depot.studies.r1 run --repeat 3 --only R1-6h R1-1h
    python -m depot.studies.r1 tables             # tables/R1_*.tex for paper and SI

``run`` solves every scenario in its own process (``depot.run_one``), so the
peak-memory figures are per run, and stores it in ``results/`` like every other
run.  With ``--repeat n`` each scenario is solved ``n`` times; the repeats are
interleaved (all scenarios once, then again) so that a slow phase of the
machine does not hit one scenario only.

``tables`` reads the HiGHS runs of the current scenario definitions (same
fingerprint, i.e. same solver options) and writes

* ``tables/R1_summary.tex``      short table for the paper (Table 2);
* ``tables/S_R1_computation.tex`` LP metrics and performance (Supplementary Table S7);
* ``tables/S_R1_results.tex``     sizes, annual amounts, economics (Supplementary Table S8);
* ``tables/R1_highs.csv``         all numbers, one column per scenario.

Timings and memory are the median over the repeats; sizes, amounts and
economics are those of the latest run (repeats solve the same LP).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from .. import scenarios as sets  # noqa: E402
from ..report import BLOCKS, MANUSCRIPT, Run, _block_rows, _longtable, _table, fmt  # noqa: E402
from ..runner import PYTHON_ROOT, SOLVER_DEFAULTS, run_scenarios  # noqa: E402

SOLVER = "highs"
TIMING_KEYS = ("formulation", "solving", "total", "memory")


def scenarios() -> list[Scenario]:
    """R1-0.25h ... R1-6h with HiGHS and its default options of ``depot.runner``."""
    return [
        sc.replace(solver_name=SOLVER, solver_options=dict(SOLVER_DEFAULTS[SOLVER]))
        for sc in sorted(sets.R1(), key=lambda s: s.t)
    ]


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(store: ResultStore, repeat: int, only: list[str] | None) -> None:
    # coarse to fine, so the quick scenarios are done first
    selected = [sc for sc in reversed(scenarios()) if not only or sc.name in only]
    for i in range(repeat):
        print(f"\n##### R1 with {SOLVER}, repeat {i + 1} of {repeat}")
        run_scenarios(selected, store=store)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
class Repeats:
    """All stored repeats of one scenario: values of the latest, median timings."""

    def __init__(self, runs: list[Run]) -> None:
        self.runs = runs
        self.latest = runs[-1]
        self.scenario = self.latest.scenario

    def __getattr__(self, name):  # sizes, amounts, economics: from the latest run
        return getattr(self.latest, name)

    def perf(self, key: str) -> float:
        values = [r.perf(key) for r in self.runs]
        values = [v for v in values if not math.isnan(v)]
        if not values:
            return math.nan
        return statistics.median(values) if key in TIMING_KEYS else values[-1]


def collect(store: ResultStore, scs: list[Scenario] | None = None) -> dict[str, Repeats]:
    """Stored runs of ``scs`` (default: R1) by scenario name, matched by fingerprint."""
    scs = scenarios() if scs is None else scs
    index = store.index()
    if index.empty:
        return {}
    # status "ok" with termination "time_limit" is a feasible, not an optimal point
    ok = index[(index["status"].astype(str) == "ok")
               & (index["termination_condition"].astype(str) == "optimal")
               & index.get("model", pd.Series("", index=index.index)).astype(str).str.startswith("opt3s")]  # fmt: skip
    out = {}
    for sc in scs:
        rows = ok[ok["scenario_fingerprint"] == sc.fingerprint].sort_values("run_id")
        runs = []
        for run_id in rows["run_id"]:
            if (store.root / run_id / "run.json").exists():
                record = store.load(run_id, with_solution=False)
                runs.append(Run(record["scenario"], record))
        if runs:
            out[sc.name] = Repeats(runs)
    return out


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _iteration_columns(runs: dict[str, Repeats]) -> list[tuple[str, str, int]]:
    """IPM and crossover iterations for an interior-point method, else simplex."""
    ipm = any(r.perf("barrier") > 0 for r in runs.values())
    if ipm:
        return [("\\makecell[tl]{IPM \\\\ iterations}", "barrier", 0),
                ("\\makecell[tl]{Crossover \\\\ iterations}", "crossover", 0)]  # fmt: skip
    return [("\\makecell[tl]{Simplex \\\\ iterations}", "simplex", 0)]


def computation_table(runs: dict[str, Repeats]) -> str:
    cols = [
        ("Variables", "cols", 0),
        ("Constraints", "rows", 0),
        ("Nonzeros", "nonzeros", 0),
        *_iteration_columns(runs),
        ("\\makecell[tl]{Formulation \\\\ time [s]}", "formulation", 1),
        ("\\makecell[tl]{Solving \\\\ time [s]}", "solving", 1),
        ("\\makecell[tl]{Total \\\\ time [s]}", "total", 1),
        ("\\makecell[tl]{Peak \\\\ memory [MB]}", "memory", 1),
    ]
    body = []
    for name, r in runs.items():
        cells = [fmt(r.perf(k), d) for _, k, d in cols]
        body.append(" & ".join([name, fmt(r.scenario.t, 2), fmt(r.scenario.N_h, 0), *cells]) + " \\\\")
    header = [" & ".join(["Scenario", "$t$ [h]", "$N_\\mathrm{h}$", *(h for h, _, _ in cols)])]
    n = max(len(r.runs) for r in runs.values())
    caption = (
        "LP metrics and computational performance for different sampling times "
        "(scenarios R1, solver HiGHS"
        + (f"; times and memory are the median of {n} runs" if n > 1 else "")
        + ")."
    )
    return _table("l" + "r" * (2 + len(cols)), caption, "tab:S_R1_computation", header, body)


def results_table(runs: dict[str, Repeats]) -> str:
    names = list(runs)
    header = [" & ".join(["Quantity", "Unit", *names])]
    return _longtable(
        "l l " + "r " * len(names),
        "Optimal sizes, annual amounts, and economic indicators for different sampling times "
        "(scenarios R1).",
        "tab:S_R1_results", header, _block_rows([runs[n] for n in names]), len(names) + 2,
    )  # fmt: skip


SUMMARY_COLUMNS = [  # (header, getter, decimals)
    ("\\makecell[tr]{Variables \\\\ {[}$10^3${]}}", lambda r: r.perf("cols") / 1e3, 1),
    ("\\makecell[tr]{Solving \\\\ time [s]}", lambda r: r.perf("solving"), 1),
    ("\\makecell[tr]{Peak \\\\ memory [MB]}", lambda r: r.perf("memory"), 0),
    ("\\makecell[tr]{Electrolyser \\\\ {[}kW{]}}", lambda r: r.size("ELYS"), 1),
    ("\\makecell[tr]{H$_2$ tank \\\\ {[}kg{]}}", lambda r: r.size("HTNK"), 1),
    ("\\makecell[tr]{Mean import \\\\ price [€/MWh]}", lambda r: r.mean_import_price(), 2),
    ("\\makecell[tr]{Annual profit \\\\ {[}M€{]}}", lambda r: r.economics()["profit"] / 1e6, 3),
    ("\\makecell[tr]{NPV \\\\ {[}M€{]}}", lambda r: r.economics()["NPV"] / 1e6, 3),
]


def summary_table(runs: dict[str, Repeats]) -> str:
    body = [
        " & ".join([fmt(r.scenario.t, 2), *(fmt(get(r), d) for _, get, d in SUMMARY_COLUMNS)])
        + " \\\\"
        for r in runs.values()
    ]
    header = [" & ".join(["$t$ [h]", *(h for h, _, _ in SUMMARY_COLUMNS)])]
    caption = (
        "Computational effort, main optimal sizes, and economic indicators for different "
        "sampling times $t$ (scenarios R1). Complete results are given in Supplementary "
        "Tables S7 and S8."
    )
    return _table("r" * (1 + len(SUMMARY_COLUMNS)), caption, "tab:R1_summary", header, body)


def csv(runs: dict[str, Repeats]) -> pd.DataFrame:
    data = {}
    for name, r in runs.items():
        col = {f"{label} {unit}": get(r) for _, rows in BLOCKS for label, unit, get, _ in rows}
        for k in ("cols", "rows", "nonzeros", "barrier", "crossover", "simplex", *TIMING_KEYS):
            col[f"perf {k}"] = r.perf(k)
        col["repeats"] = len(r.runs)
        col["run_ids"] = " ".join(x.record["run_id"] for x in r.runs)
        data[name] = col
    return pd.DataFrame(data)


def tables(store: ResultStore, out: Path) -> None:
    runs = collect(store)
    missing = [sc.name for sc in scenarios() if sc.name not in runs]
    if not runs:
        raise SystemExit(f"no {SOLVER} runs of R1 in {store.root}")
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("R1_summary", summary_table(runs)),
                       ("S_R1_computation", computation_table(runs)),
                       ("S_R1_results", results_table(runs))):  # fmt: skip
        (folder / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
        print("wrote", folder / f"{name}.tex")
    csv(runs).to_csv(folder / "R1_highs.csv")
    print("wrote", folder / "R1_highs.csv")
    print("repeats per scenario:", {n: len(r.runs) for n, r in runs.items()})
    if missing:
        print("MISSING:", ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot.studies.r1", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--only", nargs="+", help="scenario names, e.g. R1-6h R1-1h")
    p.add_argument("--store", default="results")
    p = sub.add_parser("tables")
    p.add_argument("--store", default="results")
    p.add_argument("--out", default=str(MANUSCRIPT))
    args = parser.parse_args(argv)
    store = ResultStore(PYTHON_ROOT / args.store)
    if args.command == "run":
        run(store, args.repeat, args.only)
    else:
        tables(store, Path(args.out))


if __name__ == "__main__":
    main()
