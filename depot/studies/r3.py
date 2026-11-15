"""R3 -- influence of bus equality vs. inequality, solved with HiGHS.

A bus balance is either the equality of eq. (bus_eq) or the inequality
``Σ u_i(k) >= Σ y_j(k)``, which releases a surplus of the inputs at the bus.
Every comparison is a pair of scenarios ("brothers") that differ in one balance:

* ``R3a``      heating system (HTS), day-ahead prices, four R2 scenarios.  The R2
               scenario is the inequality brother; the equality brother is new.
* ``R3b-DA``   grid connection (EGC), day-ahead prices.  The R2 scenario is the
               equality brother; the inequality brother may release electricity
               at the bus, which would pay at a negative cost of import.
* ``R3b-HEP``  grid connection (EGC), two-tariff prices without negative values;
               both brothers are new.

Run from the repository root::

    python -m depot.studies.r3 run --part a          # the four equality brothers of R3a
    python -m depot.studies.r3 run --part da         # R3b-DA
    python -m depot.studies.r3 run --part hep        # R3b-HEP, add --hts eq if R3a says so
    python -m depot.studies.r3 run --part control --store results_check
    python -m depot.studies.r3 run --part r2 --store results_check   # the four R2 brothers again
    python -m depot.studies.r3 compare               # brothers side by side, on screen
    python -m depot.studies.r3 release               # electricity released at the EGC per instant
    python -m depot.studies.r3 tables                # tables/R3_*.tex for paper and SI

``run`` solves only the brothers that are not R2 scenarios; ``--both`` solves the
R2 brothers again as well, and ``--days n`` (smoke tests) always does, since
shortened R2 runs are not stored.  ``--part control`` solves one R2 scenario
again, to check that the stored timings are still reproduced by the machine;
give it a separate ``--store`` so that the R2 tables keep their runs.

``tables`` writes (the ``.tex`` tables only with the pairs of ``TABLE_BUSES``)

* ``tables/R3_summary.tex``       one row per pair, for the paper;
* ``tables/S_R3_computation.tex`` LP metrics and performance of every run;
* ``tables/S_R3_results.tex``     sizes, annual amounts, economics of every run;
* ``tables/R3_highs.csv``         all numbers of every pair, one column per run.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from .. import scenarios as sets  # noqa: E402
from ..report import BLOCKS, MANUSCRIPT, NA, _block_rows, _longtable, _table, fmt  # noqa: E402
from ..runner import PYTHON_ROOT, formulate, run_scenarios  # noqa: E402
from . import r2  # noqa: E402
from .r1 import Repeats, collect, csv  # noqa: E402

#: time limit [s] of the new 0.25 h runs; one R2 scenario at 0.25 h stalled with dual simplex
TIME_LIMIT = 4 * 3600
#: R2 scenario solved again by ``run --part control``
CONTROL = "R2-1h-2h-6h"
#: a release below this value [kW] counts as none (solver feasibility tolerance)
RELEASE_TOL = 1e-4

#: buses whose pairs go into the LaTeX tables; the heating system pairs (R3a) were dropped
#: from the article and remain only in ``R3_highs.csv``
TABLE_BUSES = ("EGC",)

BUS_LABEL = {"HTS": "Heating system", "EGC": "El.\\ grid connection"}
BUS_NAME = {"HTS": "the heating system", "EGC": "the electrical grid connection"}
PRICE_LABEL = {"day-ahead": "day-ahead", "tariff": "two-tariff"}
#: rows added to the annual amounts of the results table: (label, bus)
RELEASED_ROWS = [("Heat released", "HTS"), ("Electricity released", "EGC")]
#: caption note of every row of RELEASED_ROWS, by bus
RELEASED_NOTE = {
    "HTS": "heat released is the surplus of heat released to the environment, either at the heating "
    "system ($\\ge$) or at its bounded inputs ($=$)",
    "EGC": "electricity released is the surplus of the inputs of the electrical grid connection",
}
EQ, INEQ = "$=$", "$\\ge$"


@dataclass(frozen=True)
class Pair:
    """Two scenarios that differ only in the balance of one bus."""

    part: str  # "a", "da" or "hep"
    bus: str  # "HTS" or "EGC"
    prices: str  # "day-ahead" or "tariff"
    times: str  # "1h-1h-4h"
    eq: Scenario
    ineq: Scenario
    stored: str | None  # the brother that is an R2 scenario: "eq", "ineq" or None

    @property
    def new(self) -> list[Scenario]:
        return [sc for side, sc in (("ineq", self.ineq), ("eq", self.eq)) if side != self.stored]

    @property
    def both(self) -> list[Scenario]:
        return [self.ineq, self.eq]


def pairs(hts_hep: str = "ineq", days: int | None = None) -> list[Pair]:
    """All pairs, with the solver and the solver options of the R2 counterpart.

    ``hts_hep`` is the heating system balance of both brothers of R3b-HEP.
    """
    r2s = {sc.name: sc for sc in r2.scenarios(days)}

    def like_r2(sc: Scenario) -> tuple[Scenario, Scenario]:
        """``sc`` with the solver settings of the R2 scenario it is derived from."""
        base = r2s["R2-" + _times(sc)]
        options = dict(base.solver_options)
        if "time_limit" in options:  # the 0.25 h scenarios
            options["time_limit"] = TIME_LIMIT
        return base, sc.replace(solver_name=base.solver_name, solver_options=options,
                                end_date=base.end_date)  # fmt: skip

    out = []
    for sc in sets.R3a():
        base, eq = like_r2(sc)
        out.append(Pair("a", "HTS", "day-ahead", _times(sc), eq, base, "ineq"))
    for sc in sets.R3b_DA():
        base, ineq = like_r2(sc)
        out.append(Pair("da", "EGC", "day-ahead", _times(sc), base, ineq, "eq"))
    eq, ineq = (like_r2(sc)[1] for sc in sets.R3b_HEP(hts_hep))
    out.append(Pair("hep", "EGC", "tariff", _times(eq), eq, ineq, None))
    return out


def _times(sc: Scenario) -> str:
    return f"{sc.t:g}h-{sc.t_of('PVP'):g}h-{sc.t_of('ELYS'):g}h"


def _unique(scs: list[Scenario]) -> list[Scenario]:
    seen, out = set(), []
    for sc in scs:
        if sc.fingerprint not in seen:
            seen.add(sc.fingerprint)
            out.append(sc)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(store: ResultStore, part: str, hts: str, days: int | None, both: bool) -> None:
    if part == "control":
        selected = [sc for sc in r2.scenarios(days) if sc.name == CONTROL]
    elif part == "r2":  # the brothers that are R2 scenarios, for timings of the same session
        selected = _unique([sc for p in pairs(hts, days) for sc in p.both if sc.name.startswith("R2-")])
    else:
        chosen = [p for p in pairs(hts, days) if part in ("all", p.part)]
        selected = _unique([sc for p in chosen for sc in (p.both if both or days else p.new)])
    print(f"\n##### R3, part {part}: {len(selected)} scenarios")
    for sc in selected:
        print(f"  {sc.name:<24} {sc.fingerprint}  {sc.solver_options}")
    run_scenarios(selected, store=store)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def sizes(r: Repeats) -> pd.Series:
    """Total size of every component, by its abbreviation."""
    s = r.record["sizes"]["S_total"]
    s.index = [str(i).split(".")[-1] for i in s.index]
    return s.astype(float)


def size_differences(eq: Repeats, ineq: Repeats) -> pd.DataFrame:
    """Sizes of both brothers and their difference, largest relative difference first."""
    frame = pd.DataFrame({"eq": sizes(eq), "ineq": sizes(ineq)})
    frame["diff"] = frame["ineq"] - frame["eq"]
    scale = frame[["eq", "ineq"]].abs().max(axis=1)
    frame["rel"] = (frame["diff"].abs() / scale).where(scale > 0, 0.0)
    return frame.sort_values("rel", ascending=False)


def max_size_difference(eq: Repeats, ineq: Repeats, min_size: float = 0.05) -> float:
    """Largest relative size difference [-] among components larger than ``min_size``."""
    d = size_differences(eq, ineq)
    d = d[d[["eq", "ineq"]].abs().max(axis=1) >= min_size]
    return float(d["rel"].max()) if len(d) else 0.0


def npv_difference(eq: Repeats, ineq: Repeats) -> float:
    """``(NPV_ineq - NPV_eq) / |NPV_eq|``."""
    a, b = eq.economics()["NPV"], ineq.economics()["NPV"]
    return (b - a) / abs(a)


def released(r: Repeats, bus: str) -> float:
    """Annual amount released to the environment: heat [kWh_th] or electricity [kWh].

    Heat: what the electrolyser, the boiler and the fuel cell produce minus the
    demand; this covers both the release at the bus (inequality) and at the
    bounded waste-heat ports (equality).  Electricity: inputs minus outputs of
    the grid connection.
    """
    a = r.record["amounts"]["amount"]
    if bus == "HTS":
        return float(a["ELYS.y3"] + a["BLR.y1"] + a["FCS.y2"] - a["FHC.u1"])
    ports = [k for k in a.index if k.startswith("EGC.")]
    return float(sum(a[k] if k.split(".")[1].startswith("u") else -a[k] for k in ports))


def compare(store: ResultStore, hts: str, days: int | None) -> None:
    ps = pairs(hts, days)
    runs = collect(store, _unique([sc for p in ps for sc in p.both]))
    for p in ps:
        eq, ineq = runs.get(p.eq.name), runs.get(p.ineq.name)
        print(f"\n===== {p.bus}, {p.prices} prices, {p.times}:  "
              f"'=' {p.eq.name}   '>=' {p.ineq.name}")  # fmt: skip
        if eq is None or ineq is None:
            print("  missing:", ", ".join(sc.name for sc, r in ((p.eq, eq), (p.ineq, ineq)) if r is None))
            continue
        rows = [("variables", "cols", 0), ("constraints", "rows", 0), ("nonzeros", "nonzeros", 0),
                ("simplex iterations", "simplex", 0), ("formulation [s]", "formulation", 1),
                ("solving [s]", "solving", 1), ("total [s]", "total", 1),
                ("peak memory [MB]", "memory", 1)]  # fmt: skip
        print(f"  {'':<22}{'=':>16}{'>=':>16}{'>= / =':>10}")
        for label, key, dec in rows:
            a, b = eq.perf(key), ineq.perf(key)
            print(f"  {label:<22}{a:>16,.{dec}f}{b:>16,.{dec}f}{b / a:>10.3f}")
        for label in ("CAPEX", "OPEX", "REV", "profit", "NPV"):
            a, b = eq.economics()[label], ineq.economics()[label]
            print(f"  {label:<22}{a:>16,.0f}{b:>16,.0f}{(b - a) / abs(a):>+10.1e}")
        a, b = released(eq, p.bus), released(ineq, p.bus)
        print(f"  {'released [MWh]':<22}{a / 1e3:>16,.2f}{b / 1e3:>16,.2f}")
        d = size_differences(eq, ineq)
        print(f"  largest size difference: {max_size_difference(eq, ineq):.1e} (relative)")
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 120):
            print(d.head(6).to_string().replace("\n", "\n    ").join(["    ", ""]))
        if p.part == "a":
            a = pd.DataFrame({"eq": eq.record["amounts"]["amount"],
                              "ineq": ineq.record["amounts"]["amount"]})  # fmt: skip
            a["rel"] = ((a["ineq"] - a["eq"]).abs() / a.abs().max(axis=1)).fillna(0.0)
            worst = a.sort_values("rel", ascending=False).head(4)
            with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 120):
                print("  largest differences of annual amounts:")
                print(worst.to_string().replace("\n", "\n    ").join(["    ", ""]))


# --------------------------------------------------------------------------- #
# electricity released at the grid connection, per instant
# --------------------------------------------------------------------------- #
def egc_release(store: ResultStore, r: Repeats) -> pd.DataFrame:
    """Per instant of the EGC: released power, import, export and the import price.

    The stored solution holds the decision variables only, so the hub is
    formulated again and the stored values are put back into its variables;
    every port is then evaluated as in :class:`opt3s.HubResult`.
    """
    from opt3s.backend import value_of
    from opt3s.timegrid import to_parameter

    sc = r.scenario
    built = formulate(sc)
    solution = xr.open_dataset(store.root / r.record["run_id"] / "solution.nc")
    for name in built.model.variables:
        built.model.variables[name].solution = solution[name]
    built.model.status = "ok"  # linopy evaluates expressions of solved models only
    grid = built.grids["EGC"]

    def port(key: str) -> np.ndarray:
        v = value_of(built.ports[key])
        return np.full(grid.n_h, v) if not isinstance(v, xr.DataArray) else np.asarray(v.values, float)

    egc = built.hub["EGC"]
    ins = sum(port(f"EGC.{p}") for p in egc.inputs)
    outs = sum(port(f"EGC.{p}") for p in egc.outputs)
    price = to_parameter(built.hub["ESRC"].ports["y1"].c_opr, built.grids["SUPP"])
    fee = to_parameter(egc.ports["u1"].c_opr, grid)
    return pd.DataFrame(
        {"release": ins - outs, "import": port("EGC.u1"), "export": port("EGC.y1"),
         "price": np.asarray(price), "fee": np.asarray(fee)},
        index=grid.timestamps,
    )  # fmt: skip


def release(store: ResultStore, hts: str, out: Path | None) -> None:
    ps = [p for p in pairs(hts) if p.bus == "EGC"]
    runs = collect(store, _unique([sc for p in ps for sc in p.both]))
    for p in ps:
        for side, sc in (("=", p.eq), (">=", p.ineq)):
            r = runs.get(sc.name)
            if r is None:
                print(f"\n{sc.name}: no run")
                continue
            f = egc_release(store, r)
            t = r.scenario.t
            cost = f["price"] + f["fee"]  # what one imported kWh costs [€/kWh]
            rel = f["release"] > RELEASE_TOL
            neg = cost < 0
            print(f"\n===== {sc.name} ('{side}', {p.prices} prices)")
            print(f"  instants                         {len(f):>12,}")
            print(f"  instants with import cost < 0    {int(neg.sum()):>12,}")
            print(f"  instants with a release          {int(rel.sum()):>12,}")
            print(f"    of which with import cost < 0  {int((rel & neg).sum()):>12,}")
            print(f"  released energy [MWh]            {t * f['release'].sum() / 1e3:>12,.1f}")
            print(f"    of which at import cost < 0    {t * f.loc[neg, 'release'].sum() / 1e3:>12,.1f}")
            print(f"  largest release [kW]             {f['release'].max():>12,.1f}")
            print(f"  largest import [kW]              {f['import'].max():>12,.1f}")
            print(f"  import at import cost < 0 [MWh]  {t * f.loc[neg, 'import'].sum() / 1e3:>12,.1f}")
            print(f"  export at import cost < 0 [MWh]  {t * f.loc[neg, 'export'].sum() / 1e3:>12,.1f}")
            if rel.any():
                print(f"  import cost at a release [€/MWh] {1e3 * cost[rel].min():>12,.2f} to "
                      f"{1e3 * cost[rel].max():,.2f}")  # fmt: skip
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                f.to_csv(out / f"{sc.name}_egc_release.csv")
                print("  wrote", out / f"{sc.name}_egc_release.csv")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _small(x: float, decimals: int = 3) -> str:
    """A percentage that may be far below the last printed decimal."""
    if math.isnan(x):
        return NA
    limit = 10.0**-decimals
    if abs(x) < limit / 2:
        return f"$<${limit:.{decimals}f}"
    return fmt(x, decimals)


def _pair_cells(p: Pair, first_of_bus: bool) -> list[str]:
    return [BUS_LABEL[p.bus] if first_of_bus else "", PRICE_LABEL[p.prices], p.times]


def _buses(ps: list[Pair]) -> list[str]:
    """Buses of ``ps`` in order of appearance."""
    return list(dict.fromkeys(p.bus for p in ps))


def _number(text: str) -> float:
    """The value of a number printed by ``fmt``."""
    return float(text.replace(",", ""))


def _bus_note(ps: list[Pair]) -> str:
    """Caption sentence explaining the bus abbreviations, e.g. 'EGC is the ... connection. '"""
    parts = [f"{b} {'is ' if i == 0 else ''}{BUS_NAME[b]}" for i, b in enumerate(_buses(ps))]
    return " and ".join(parts) + ". "


#: paired columns of the summary: (header, perf key, decimals)
SUMMARY_GROUPS = [
    ("Variables", "cols", 0),
    ("Simplex iterations", "simplex", 0),
    ("Solving time [s]", "solving", 0),
    ("Peak memory [MB]", "memory", 0),
]


def summary_table(ps: list[Pair], runs: dict[str, Repeats]) -> str:
    """One row per pair: effort of both brothers and how far their optima are apart."""
    ng = len(SUMMARY_GROUPS)
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h, _, _ in SUMMARY_GROUPS)
    rules = " ".join(f"\\cmidrule(lr){{{4 + 2 * g}-{5 + 2 * g}}}" for g in range(ng))
    header = [
        f" & & & {groups} & & \\\\\n        {rules}\n        "
        + " & ".join(["Bus", "Prices", "Sampling times", *([EQ, INEQ] * ng),
                      "$\\Delta$NPV [\\%]", "$\\Delta S_\\mathrm{max}$ [\\%]"])  # fmt: skip
    ]
    body, last_bus = [], None
    for p in ps:
        eq, ineq = runs.get(p.eq.name), runs.get(p.ineq.name)
        if last_bus is not None and p.bus != last_bus:
            body.append("\\addlinespace")
        cells = _pair_cells(p, p.bus != last_bus)
        last_bus = p.bus
        for _, key, dec in SUMMARY_GROUPS:
            cells += [fmt(r.perf(key), dec) if r is not None else NA for r in (eq, ineq)]
        if eq is not None and ineq is not None:
            cells += [_small(100 * npv_difference(eq, ineq)),
                      _small(100 * max_size_difference(eq, ineq))]  # fmt: skip
        else:
            cells += [NA, NA]
        body.append(" & ".join(cells) + " \\\\")
    caption = (
        "Computational effort and optimal solutions with the bus balance as equality ($=$) and "
        "as inequality ($\\ge$) (scenarios R3). "
        "$\\Delta$NPV is the relative difference of the NPV and $\\Delta S_\\mathrm{max}$ the "
        "largest relative difference of a component size, both of the inequality with respect "
        "to the equality. "
        "Complete results are given in Supplementary Tables S12 and S13."
    )
    return _table("lll" + "r" * (2 * ng + 2), caption, "tab:R3_summary", header, body)


def computation_table(ps: list[Pair], runs: dict[str, Repeats]) -> str:
    """Every run on its own row; narrow enough for the page, so no sampling time column.

    The bus column is left out when all pairs share one bus; the caption then names it.
    """
    cols = [
        ("Variables", "cols", 0),
        ("Constraints", "rows", 0),
        ("Nonzeros", "nonzeros", 0),
        ("\\makecell[tl]{Simplex \\\\ iterations}", "simplex", 0),
        ("\\makecell[tl]{Solving \\\\ time [s]}", "solving", 1),
        ("\\makecell[tl]{Peak \\\\ memory [MB]}", "memory", 1),
    ]
    several = len(_buses(ps)) > 1
    body, last_bus = [], None
    for p in ps:
        if last_bus is not None and p.bus != last_bus:
            body.append("\\midrule")
        for i, (sign, sc) in enumerate(((EQ, p.eq), (INEQ, p.ineq))):
            r = runs.get(sc.name)
            lead = [p.bus if p.bus != last_bus and i == 0 else ""] if several else []
            lead.append(PRICE_LABEL[p.prices] if i == 0 else "")
            cells = [fmt(r.perf(k), d) if r is not None else NA for _, k, d in cols]
            body.append(" & ".join([*lead, sign, sc.name, *cells]) + " \\\\")
        last_bus = p.bus
    lead_head = ["Bus"] if several else []
    header = [" & ".join([*lead_head, "Prices", "Balance", "Scenario", *(h for h, _, _ in cols)])]
    formulation = [fmt(r.perf("formulation"), 1) for r in runs.values()]
    lo, hi = min(formulation, key=_number), max(formulation, key=_number)
    if several:
        what = "the bus balance"
    else:
        what = f"the balance of {BUS_NAME[ps[0].bus]} ({ps[0].bus})"
    caption = (
        f"LP metrics and computational performance with {what} as equality ($=$) and "
        "as inequality ($\\ge$) (scenarios R3, solver HiGHS). "
        + (_bus_note(ps) if several else "")
        + "Scenarios named R2 are those of Supplementary Table S9. "
        f"The formulation time is {lo if lo == hi else f'{lo}--{hi}'}~s in all scenarios."
    )
    lead_cols = "l" * len(lead_head)
    return _table(lead_cols + "lll" + "r" * len(cols), caption, "tab:S_R3_computation", header, body)


def results_table(ps: list[Pair], runs: dict[str, Repeats], per_block: int = 3) -> str:
    """Sizes, amounts and economics; up to ``per_block`` pairs side by side in every block.

    Only the released amounts of the buses in ``ps`` are added to the annual amounts.
    """
    per_block = min(per_block, len(ps))
    buses = _buses(ps)
    released_rows = [(label, bus) for label, bus in RELEASED_ROWS if bus in buses]
    n = 2 * per_block + 2
    several = len(buses) > 1  # with one bus, the caption names it and the column titles do not
    header = [f"Quantity & Unit & \\multicolumn{{{2 * per_block}}}{{c}}"
              f"{{{'Bus, prices' if several else 'Prices'}, sampling times, and balance}}"]  # fmt: skip
    body = []
    for b in range(0, len(ps), per_block):
        block = ps[b : b + per_block]
        if b:
            body.append("\\newpage")  # every block on its own page
        pad = [""] * (2 * (per_block - len(block)))
        titles = [f"\\multicolumn{{2}}{{c}}{{{f'{p.bus}, ' if several else ''}{PRICE_LABEL[p.prices]}}}"
                  for p in block]  # fmt: skip
        times = [f"\\multicolumn{{2}}{{c}}{{{p.times}}}" for p in block]
        body.append(" & ".join(["", "", *titles, *pad]) + " \\\\")
        body.append(" & ".join(["", "", *times, *pad]) + " \\\\")
        body.append(" & ".join(["", "", *([EQ, INEQ] * len(block)), *pad]) + " \\\\")
        body.append("\\midrule")
        columns = [runs.get(sc.name) for p in block for sc in (p.eq, p.ineq)]
        rows = _block_rows(columns + [None] * len(pad))
        extra = [" & ".join([label, "[MWh]", *(fmt(released(r, bus) / 1e3, 1) if r is not None else NA
                                               for r in columns), *pad]) + " \\\\"
                 for label, bus in released_rows]  # fmt: skip
        at = next(i for i, line in enumerate(rows) if "\\textit{Economics}" in line) - 1
        rows[at:at] = extra  # the last rows of the annual amounts
        if pad:  # empty cells instead of "--" in the unused columns
            rows = [_blank_tail(line, len(pad)) for line in rows]
        body += [line.replace(f"\\multicolumn{{{len(columns) + len(pad) + 2}}}", f"\\multicolumn{{{n}}}")
                 for line in rows]  # fmt: skip
    if len(buses) > 1:
        what = "the bus balance"
    else:
        what = f"the balance of {BUS_NAME[buses[0]]} ({buses[0]})"
    released_note = ", and ".join(RELEASED_NOTE[bus] for _, bus in released_rows)
    caption = (
        f"Optimal sizes, annual amounts, and economic indicators with {what} as equality "
        "($=$) and as inequality ($\\ge$) (scenarios R3). "
        + (_bus_note(ps) if len(buses) > 1 else "")
        + released_note[:1].upper() + released_note[1:] + "."
    )
    return _longtable("l l " + "r " * (2 * per_block), caption, "tab:S_R3_results", header, body, n)


def _blank_tail(line: str, n: int) -> str:
    if not line.endswith(" \\\\") or " & " not in line:
        return line
    cells = line[: -len(" \\\\")].split(" & ")
    return " & ".join(cells[:-n] + [""] * n) + " \\\\"


def tables(store: ResultStore, out: Path, hts: str, days: int | None) -> None:
    ps = pairs(hts, days)
    scs = _unique([sc for p in ps for sc in p.both])
    runs = collect(store, scs)
    if not runs:
        raise SystemExit(f"no runs of R3 in {store.root}")
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    shown = [p for p in ps if p.bus in TABLE_BUSES]  # the CSV below keeps every pair
    shown_runs = {sc.name: runs[sc.name] for sc in _unique([sc for p in shown for sc in p.both])
                  if sc.name in runs}  # fmt: skip
    for name, text in (("R3_summary", summary_table(shown, shown_runs)),
                       ("S_R3_computation", computation_table(shown, shown_runs)),
                       ("S_R3_results", results_table(shown, shown_runs))):  # fmt: skip
        (folder / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
        print("wrote", folder / f"{name}.tex")
    frame = csv({sc.name: runs[sc.name] for sc in scs if sc.name in runs})
    for p in ps:
        for sc in p.both:
            if sc.name in runs:
                frame.loc[f"released at {p.bus} per year", sc.name] = released(runs[sc.name], p.bus)
    frame.to_csv(folder / "R3_highs.csv", lineterminator="\n")  # same file on every OS
    print("wrote", folder / "R3_highs.csv")
    missing = [sc.name for sc in scs if sc.name not in runs]
    if missing:
        print("MISSING:", ", ".join(missing))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot.studies.r3", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare", "release", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--store", default="results")
        p.add_argument("--hts", default="ineq", choices=("ineq", "eq"),
                       help="heating system balance of both brothers of R3b-HEP")  # fmt: skip
        if name != "release":
            p.add_argument("--days", type=int, help="shorten the horizon (smoke tests only)")
        if name == "run":
            p.add_argument("--part", default="all", choices=("all", "a", "da", "hep", "control", "r2"))
            p.add_argument("--both", action="store_true", help="also solve the R2 brothers again")
        if name == "tables":
            p.add_argument("--out", default=str(MANUSCRIPT))
        if name == "release":
            p.add_argument("--out", help="folder for the per-instant CSV files")
    args = parser.parse_args(argv)
    store = ResultStore(PYTHON_ROOT / args.store)
    if args.command == "run":
        run(store, args.part, args.hts, args.days, args.both)
    elif args.command == "compare":
        compare(store, args.hts, args.days)
    elif args.command == "release":
        release(store, args.hts, Path(args.out) if args.out else None)
    else:
        tables(store, Path(args.out), args.hts, args.days)


if __name__ == "__main__":
    main()
