"""Tables and figures for the manuscript and the Supplementary Information.

Reads the latest stored run of every scenario of R1 and R2 and writes::

    <out>/tables/R1_computation.tex   LP metrics and performance, uniform sampling
    <out>/tables/R1_results.tex       sizes, annual amounts, economics, uniform sampling
    <out>/tables/R2_computation.tex   the same for mixed sampling
    <out>/tables/R2_results.tex
    <out>/tables/*.csv                the same numbers, machine readable
    <out>/tables/S3_costs.tex         Supplementary Table S3, from the model definition
    <out>/tables/S5_components.tex    Supplementary Table S5, from the model definition
    <out>/tables/S6_custom.tex        Supplementary Table S6, from the model definition
    <out>/img/R1_sampling.pdf         figure: effort and accuracy vs. the sampling time
    <out>/img/R2_mixed.pdf            figure: mixed sampling times

``<out>`` is the repository root by default.  The .tex files
are fragments meant for ``\\input``; ``tables/preview.tex`` compiles them all.

Run from the repository root::

    python -m depot.report                 # opt3s runs only
    python -m depot.report --model any     # also legacy runs (no annual amounts)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import opt3s  # noqa: E402
from results import ResultStore  # noqa: E402
from scenario import Scenario  # noqa: E402

from . import scenarios  # noqa: E402
from .hub import build_hub  # noqa: E402
from .runner import PYTHON_ROOT  # noqa: E402

MANUSCRIPT = PYTHON_ROOT  # tables/ and img/ are written to the repository root
NA = "--"


# --------------------------------------------------------------------------- #
# reading runs
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """One stored run, reduced to what the tables and figures use."""

    scenario: Scenario
    record: dict[str, Any]

    @property
    def objective(self) -> dict[str, Any]:
        return self.record["objective"]

    @property
    def metrics(self) -> dict[str, Any]:
        return self.record["metrics"]

    def size(self, comp: str) -> float:
        sizes = self.record["sizes"]
        short = {str(i).split(".")[-1]: i for i in sizes.index}
        if comp not in short:
            return math.nan
        return float(sizes.loc[short[comp], "S_total"])

    def amount(self, port: str) -> float:
        amounts = self.record.get("amounts")
        if amounts is None or port not in amounts.index:
            return math.nan
        return float(amounts.loc[port, "amount"])

    def economics(self) -> dict[str, float]:
        o = self.objective
        npv = o["NPV"] if "NPV" in o else -o["value"]
        return {
            "CAPEX": o["CAPEX"],
            "OPEX": o["OPEX"],
            "REV": o["REV"],
            "profit": o["REV"] - o["OPEX"],
            "NPV": npv,
        }

    def mean_import_price(self) -> float:
        """Mean price of imported energy [€/MWh]: purchase cost / imported energy."""
        costs = self.record["costs"]
        row = "ESRC" if "ESRC" in costs.index else "SUPP"
        energy = self.amount("ESRC.y1")
        return float(costs.loc[row, "C_opr"]) / energy * 1000 if energy else math.nan

    def perf(self, key: str) -> float:
        m = self.metrics
        s = m["solver"]
        values = {
            "rows": m["size"].get("n_rows"),
            "cols": m["size"].get("n_cols"),
            "nonzeros": m["size"].get("n_nonzeros"),
            "barrier": s.get("ipm_iterations"),
            # HiGHS reports crossover separately; Gurobi's simplex count is its crossover
            "crossover": s.get("crossover_iterations", s.get("simplex_iterations")),
            "simplex": s.get("simplex_iterations"),
            "total": m["totals"].get("total_wall_s"),
            "solving": m["phases"].get("solving", {}).get("wall_s"),
            "formulation": m["phases"].get("formulation", {}).get("wall_s"),
            "memory": m["totals"].get("peak_rss_mb"),
        }
        v = values[key]
        return math.nan if v is None else float(v)


def latest_runs(store: ResultStore, names: Iterable[str], model: str) -> dict[str, Run]:
    """The newest successful run of every scenario name (``model``: opt3s / legacy / any)."""
    index = store.index()
    if index.empty:
        return {}
    index = index[index["status"].astype(str) == "ok"]
    model_col = index.get("model", pd.Series("legacy", index=index.index)).fillna("legacy")
    if model == "opt3s":
        index = index[model_col.astype(str).str.startswith("opt3s")]
    elif model == "legacy":
        index = index[~model_col.astype(str).str.startswith("opt3s")]
    runs = {}
    for name in names:
        rows = index[index["sc_name"] == name].sort_values("run_id")
        if rows.empty:
            continue
        run_id = rows["run_id"].iloc[-1]
        if not (store.root / run_id / "run.json").exists():
            continue
        record = store.load(run_id, with_solution=False)
        runs[name] = Run(record["scenario"], record)
    return runs


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #
def fmt(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return NA
    if abs(x) < 0.5 * 10 ** (-decimals):
        x = 0.0
    return f"{x:,.{decimals}f}"


def _longtable(columns: str, caption: str, label: str, header: list[str], body: list[str],
               n_cols: int, size: str = "\\footnotesize", tabcolsep: str = "4pt") -> str:  # fmt: skip
    head = "    \\toprule\n" + "".join(f"    {h} \\\\\n" for h in header) + "    \\midrule\n"
    return (
        "% generated by python/depot/report.py -- do not edit by hand\n"
        f"{{{size}\n\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
        f"\\begin{{longtable}}{{{columns}}}\n"
        f"    \\caption{{{caption}}}\n    \\label{{{label}}} \\\\\n"
        f"{head}    \\endfirsthead\n"
        f"    \\caption[]{{\\textit{{(continued)}}}} \\\\\n{head}    \\endhead\n"
        f"    \\midrule\n    \\multicolumn{{{n_cols}}}{{r}}{{\\footnotesize Continued on next page}} \\\\\n"
        "    \\endfoot\n    \\bottomrule\n    \\endlastfoot\n"
        + "".join(f"    {line}\n" for line in body)
        + "\\end{longtable}}\n"
    )


def _table(columns: str, caption: str, label: str, header: list[str], body: list[str]) -> str:
    return (
        "% generated by python/depot/report.py -- do not edit by hand\n"
        "\\begin{table}[ht]\n    \\centering\n"
        f"    \\caption{{{caption}}}\n    \\label{{{label}}}\n"
        "    \\footnotesize\n    \\setlength{\\tabcolsep}{3pt}\n"
        f"    \\begin{{tabular}}{{{columns}}}\n        \\toprule\n"
        + "".join(f"        {h} \\\\\n" for h in header)
        + "        \\midrule\n"
        + "".join(f"        {line}\n" for line in body)
        + "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------- #
# quantities of the results tables
# --------------------------------------------------------------------------- #
#: (label, unit, getter, decimals)
Row = tuple[str, str, Callable[[Run], float], int]

SIZES: list[Row] = [
    ("PV array 1", "[kWp]", lambda r: r.size("PVA1"), 1),
    ("PV array 2", "[kWp]", lambda r: r.size("PVA2"), 1),
    ("PV array 3", "[kWp]", lambda r: r.size("PVA3"), 1),
    ("PV inverter", "[kW]", lambda r: r.size("PVI"), 1),
    ("Battery", "[kWh]", lambda r: r.size("BATT"), 1),
    ("BESS power converter", "[kW]", lambda r: r.size("PC"), 1),
    ("Electrolyser", "[kW]", lambda r: r.size("ELYS"), 1),
    ("H$_2$ tank", "[kg]", lambda r: r.size("HTNK"), 1),
    ("H$_2$ compressor 2", "[kg/h]", lambda r: r.size("HCMP2"), 1),
    ("Gas grid injection", "[kg/h]", lambda r: r.size("GGI"), 1),
    ("Fuel cell system", "[kW]", lambda r: r.size("FCS"), 1),
    ("O$_2$ compressor", "[kg/h]", lambda r: r.size("OCMP"), 1),
    ("O$_2$ tank", "[kg]", lambda r: r.size("OTNK"), 1),
    ("Water tank", "[kg]", lambda r: r.size("WTNK"), 1),
    ("Rain collection area", "[m$^2$]", lambda r: r.size("RAIN"), 1),
    ("El. grid connection", "[kW]", lambda r: r.size("EGC"), 1),
]
AMOUNTS: list[Row] = [
    ("Grid import", "[MWh]", lambda r: r.amount("ESRC.y1") / 1000, 1),
    ("Grid export", "[MWh]", lambda r: r.amount("ESNK.u1") / 1000, 1),
    ("PV plant output", "[MWh]", lambda r: r.amount("PVI.y1") / 1000, 1),
    ("BESS charging", "[MWh]", lambda r: r.amount("PC.u1") / 1000, 1),
    ("BESS discharging", "[MWh]", lambda r: r.amount("PC.y1") / 1000, 1),
    ("Electrolyser el. input", "[MWh]", lambda r: r.amount("ELYS.u1") / 1000, 1),
    ("H$_2$ produced", "[kg]", lambda r: r.amount("ELYS.y1"), 0),
    ("H$_2$ sold", "[kg]", lambda r: r.amount("HSLS.u1"), 0),
    ("O$_2$ sold", "[kg]", lambda r: r.amount("OSLS.u1"), 0),
    ("Rain water collected", "[m$^3$]", lambda r: r.amount("RAIN.y1") / 1000, 0),
    ("Mean import price", "[€/MWh]", lambda r: r.mean_import_price(), 2),
]
ECONOMICS: list[Row] = [
    ("CAPEX", "[€]", lambda r: r.economics()["CAPEX"], 0),
    ("OPEX", "[€/year]", lambda r: r.economics()["OPEX"], 0),
    ("Revenue", "[€/year]", lambda r: r.economics()["REV"], 0),
    ("Annual profit", "[€/year]", lambda r: r.economics()["profit"], 0),
    ("NPV", "[€]", lambda r: r.economics()["NPV"], 0),
]
BLOCKS = [("Optimal sizes", SIZES), ("Annual amounts", AMOUNTS), ("Economics", ECONOMICS)]

PERF_COLUMNS = [  # (header, key, decimals)
    ("Variables", "cols", 0),
    ("Constraints", "rows", 0),
    ("Nonzeros", "nonzeros", 0),
    ("\\makecell[tl]{Barrier \\\\ iterations}", "barrier", 0),
    ("\\makecell[tl]{Crossover \\\\ iterations}", "crossover", 0),
    ("\\makecell[tl]{Formulation \\\\ time [s]}", "formulation", 1),
    ("\\makecell[tl]{Solving \\\\ time [s]}", "solving", 1),
    ("\\makecell[tl]{Total \\\\ time [s]}", "total", 1),
    ("\\makecell[tl]{Peak \\\\ memory [MB]}", "memory", 1),
]


def _block_rows(runs: list[Run | None]) -> list[str]:
    n = len(runs) + 2
    body = []
    for b, (title, rows) in enumerate(BLOCKS):
        if b:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for label, unit, get, dec in rows:
            cells = [fmt(get(r), dec) if r is not None else NA for r in runs]
            body.append(" & ".join([label, unit, *cells]) + " \\\\")
    return body


# --------------------------------------------------------------------------- #
# R1 and R2 tables
# --------------------------------------------------------------------------- #
def r1_tables(runs: dict[str, Run]) -> dict[str, str]:
    order = [sc.name for sc in sorted(scenarios.R1(), key=lambda s: s.t)]
    present = [runs.get(n) for n in order]

    body = []
    for name, r in zip(order, present):
        if r is None:
            continue
        cells = [fmt(r.perf(k), d) for _, k, d in PERF_COLUMNS]
        body.append(" & ".join([name, fmt(r.scenario.t, 2), fmt(r.scenario.N_h, 0), *cells]) + " \\\\")
    header = [" & ".join(["Scenario", "$t$ [h]", "$N_\\mathrm{h}$", *(h for h, _, _ in PERF_COLUMNS)])]
    comp = _table("l" + "r" * (2 + len(PERF_COLUMNS)),
                  "LP metrics and computational performance for different sampling times.",
                  "tab:R1_computation", header, body)  # fmt: skip

    header = [" & ".join(["Quantity", "Unit", *order])]
    res = _longtable("l l " + "r " * len(order),
                     "Optimal sizes, annual amounts, and economic indicators for different sampling times.",
                     "tab:R1_results", header, _block_rows(present), len(order) + 2)  # fmt: skip
    return {"R1_computation": comp, "R1_results": res}


def _r2_key(sc: Scenario) -> tuple[float, float, float]:
    return sc.t, sc.t_of("PVP"), sc.t_of("ELYS")


def r2_tables(runs: dict[str, Run]) -> dict[str, str]:
    grid = scenarios.R2()
    fasts = sorted({sc.t for sc in grid})
    med_slow = sorted({(sc.t_of("PVP"), sc.t_of("ELYS")) for sc in grid})

    body, last_fast = [], None
    for sc in grid:
        r = runs.get(sc.name)
        if r is None:
            continue
        if last_fast is not None and sc.t != last_fast:
            body.append("\\midrule")
        last_fast = sc.t
        cells = [fmt(r.perf(k), d) for _, k, d in PERF_COLUMNS]
        body.append(" & ".join([sc.name, *cells]) + " \\\\")
    header = [" & ".join(["Scenario", *(h for h, _, _ in PERF_COLUMNS)])]
    comp = _table("l" + "r" * len(PERF_COLUMNS),
                  "LP metrics and computational performance for mixed sampling times.",
                  "tab:R2_computation", header, body)  # fmt: skip

    by_key = {_r2_key(sc): runs.get(sc.name) for sc in grid}
    n = len(med_slow) + 2
    header = [
        f" & & \\multicolumn{{{len(med_slow)}}}{{c}}{{$t_\\mathrm{{med}}$-$t_\\mathrm{{slow}}$}}",
        " & ".join(["Quantity", "Unit", *(f"{m:g}h-{s:g}h" for m, s in med_slow)]),
    ]
    body = []
    for i, f in enumerate(fasts):
        if i:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{n}}}{{l}}{{\\textbf{{$t_\\mathrm{{fast}}$ = {f:g}~h}}}} \\\\")
        body.append("\\midrule")
        body += _block_rows([by_key.get((f, m, s)) for m, s in med_slow])
    res = _longtable("l l " + "r " * len(med_slow),
                     "Optimal sizes, annual amounts, and economic indicators for mixed sampling times.",
                     "tab:R2_results", header, body, n)  # fmt: skip
    # the header of the R2 results table has a cmidrule under the group label
    res = res.replace(
        f"{{$t_\\mathrm{{med}}$-$t_\\mathrm{{slow}}$}} \\\\\n",
        f"{{$t_\\mathrm{{med}}$-$t_\\mathrm{{slow}}$}} \\\\\n    \\cmidrule(l){{3-{n}}}\n",
    )
    return {"R2_computation": comp, "R2_results": res}


def csv_tables(runs: dict[str, Run]) -> dict[str, pd.DataFrame]:
    out = {}
    for set_name, scs in (("R1", scenarios.R1()), ("R2", scenarios.R2())):
        data = {}
        for sc in scs:
            r = runs.get(sc.name)
            if r is None:
                continue
            col = {f"{label} {unit}": get(r) for _, rows in BLOCKS for label, unit, get, _ in rows}
            col.update({f"perf {k}": r.perf(k) for _, k, _ in PERF_COLUMNS})
            col["run_id"] = r.record["run_id"]
            data[sc.name] = col
        if data:
            out[set_name] = pd.DataFrame(data)
    return out


# --------------------------------------------------------------------------- #
# Supplementary Tables S3, S5, S6 from the model definition
# --------------------------------------------------------------------------- #
SIZE_UNITS = {
    "PVA1": "kW\\textsubscript{p}", "PVI": "kW", "BATT": "kWh", "PC": "kW", "ELYS": "kW",
    "HCMP1": "kg/h", "HTNK": "kg", "HCMP2": "kg/h", "GGI": "kg/h", "FCS": "kW",
    "OCMP": "kg/h", "OTNK": "kg", "DMN": "kW", "WPMP": "kW", "WTNK": "kg",
    "RAIN": "m\\textsuperscript{2}", "WGS": "kg/h", "BLR": "kW\\textsubscript{th}",
    "HTS": "kW\\textsubscript{th}", "EGC": "kW",
}  # fmt: skip


def _num(x: Any) -> str:
    if x is None:
        return "$\\infty$"
    return f"{x:,.10g}"


def s3_table(hub: opt3s.Hub) -> str:
    body = []
    for c in hub.standards():
        if c.size is None or c.name in ("PVA2", "PVA3"):
            continue
        k = c.costs
        name = "PV array (PVA$a$)" if c.name == "PVA1" else f"{c.label} ({c.name})"
        name = name.replace("H2", "H\\textsubscript{2}").replace("O2", "O\\textsubscript{2}")
        cells = [
            name, SIZE_UNITS.get(c.name, ""), _num(c.size.s_exist), _num(c.size.s_max),
            _num(k.c_inv) if k.c_inv else NA, _num(k.c_mnt) if k.c_mnt else NA,
            _num(k.c_rpl) if k.c_rpl else NA, _num(k.n_lt) if k.n_lt else NA,
        ]  # fmt: skip
        body.append(" & ".join(cells) + " \\\\")
    header = [
        "Component & \\makecell[tl]{Unit of \\\\ size} & $S_\\mathrm{exist}$ & $S_\\mathrm{max}$ & "
        "\\makecell[tr]{$c_\\mathrm{inv}$ \\\\ {[}€/unit{]}} & "
        "\\makecell[tr]{$c_\\mathrm{mnt}$ \\\\ {[}€/(unit\\,a){]}} & "
        "\\makecell[tr]{$c_\\mathrm{rpl}$ \\\\ {[}€/unit{]}} & \\makecell[tr]{$N_\\mathrm{LT}$ \\\\ {[}a{]}}"
    ]
    caption = (
        "Sizes and costs of the components. $S_\\mathrm{exist}$ is the existing size, "
        "$S_\\mathrm{max}$ the upper bound of the total size, $c_\\mathrm{inv}$ the unit investment "
        "price, $c_\\mathrm{mnt}$ the yearly unit maintenance price, $c_\\mathrm{rpl}$ the unit "
        "replacement price, and $N_\\mathrm{LT}$ the lifetime. All prices are per unit of size given "
        "in the second column. A dash denotes that the parameter is not used, and $\\infty$ that "
        "the size is not bounded from above."
    )
    return _table("l l r r r r r r", caption, "tab:S_costs", header, body).replace(
        "\\footnotesize\n    \\setlength{\\tabcolsep}{3pt}", "\\small"
    )


def _port_tex(token: str) -> str:
    """``u1`` → ``$u_1$``, ``u1 = PC.u1`` → ``$u_1 = u^\\mathrm{PC}_1$``."""
    def one(p: str, comp: str | None = None) -> str:
        sup = f"^\\mathrm{{{comp}}}" if comp else ""
        return f"{p[0]}{sup}_{{{p[1:]}}}"

    if " = " in token:
        left, right = token.split(" = ")
        comp, port = right.split(".")
        return f"${one(left)} = {one(port, comp)}$"
    port, _, kind = token.partition(" as ")
    return f"${one(port)}$ as {kind}"


def _grouped_ports(tokens: list[str]) -> list[str]:
    """Ports of one direction and type in one entry: ``$u_{\\{1,\\dots,4\\}}$ as DO``."""
    if any(" = " in t for t in tokens):
        return [_port_tex(t) for t in tokens]
    groups: dict[tuple[str, str], list[int]] = {}
    for t in tokens:
        port, _, kind = t.partition(" as ")
        groups.setdefault((port[0], kind), []).append(int(port[1:]))
    out = []
    for (direction, kind), nums in groups.items():
        nums = sorted(nums)
        if len(nums) == 1:
            idx = str(nums[0])
        elif len(nums) >= 4 and nums == list(range(nums[0], nums[-1] + 1)):
            idx = f"\\{{{nums[0]},\\dots,{nums[-1]}\\}}"
        else:
            idx = "\\{" + ",".join(map(str, nums)) + "\\}"
        out.append(f"${direction}_{{{idx}}}$ as {kind}")
    return out


def _size_tex(s: str) -> str:
    if s in ("-", "x"):
        return {"-": NA, "x": "$\\bullet$"}[s]
    if s == "S_exist":
        return "$S_\\mathrm{exist}$"
    s = s.replace("max(", "\\max(").replace("max ", "\\max ")
    for p in sorted({w for w in s.replace("(", " ").replace(")", " ").replace(",", " ").split()
                     if w[:1] in "uyx" and w[1:].isdigit()}, key=len, reverse=True):  # fmt: skip
        s = s.replace(p, f"{p[0]}_{{{p[1:]}}}")
    return f"${s}$"


def _component_rows(frame: pd.DataFrame, indent: set[str] = frozenset()) -> list[str]:
    body = []
    for abbrev, row in frame.iterrows():
        ports = _grouped_ports(row["ports"].split(", "))
        ports_tex = ports[0] if len(ports) == 1 else "\\makecell[tl]{" + ", \\\\ ".join(ports) + "}"
        lines = [
            x.replace("H2", "H\\textsubscript{2}").replace("O2", "O\\textsubscript{2}")
            for x in _wrap(row["label"])
        ]
        if abbrev in indent:
            lines = ["\\: " + x for x in lines]
        elif row["type"] == "custom" and indent:
            lines = [f"\\textit{{{x}}}" for x in lines]
        label = lines[0] if len(lines) == 1 else "\\makecell[tl]{" + " \\\\ ".join(lines) + "}"
        states = NA if row["states"] == "-" else f"${row['states'][0]}_{{{row['states'][1:]}}}$"
        costs = [{"x": "$\\bullet$", "-": NA}.get(row[t], row[t]) for t in
                 ("C_inv", "C_mnt", "C_deg", "C_peak", "C_opr", "C_rev")]  # fmt: skip
        body.append(" & ".join([label, str(abbrev), row["type"], ports_tex, states,
                                _size_tex(row["size"]), *costs]) + " \\\\")  # fmt: skip
        body.append("\\addlinespace")
    return body[:-1]


def _wrap(label: str, width: int = 18) -> list[str]:
    """Split a long component name into two lines at the space nearest its middle."""
    if len(label) <= width or " " not in label:
        return [label]
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return [label[:cut], label[cut + 1:]]


COMP_HEADER = [
    "Component & Abbrev. & Type & Ports & States & Size & $C_\\mathrm{inv}$ & $C_\\mathrm{mnt}$ & "
    "$C_\\mathrm{deg}$ & $C_\\mathrm{peak}$ & $C_\\mathrm{opr}$ & $C_\\mathrm{rev}$"
]


def s5_s6_tables(hub: opt3s.Hub) -> dict[str, str]:
    s5 = _longtable("l l l l l l l l l l l l", "Declarative definition of the EH components.",
                    "tab:S_components", COMP_HEADER, _component_rows(opt3s.describe(hub)), 12,
                    size="\\small", tabcolsep="3pt")  # fmt: skip
    frame = opt3s.describe(hub, custom_parts=True)
    customs = {c.name for c in hub.components if isinstance(c, opt3s.Custom)}
    body = []
    for name in customs_in_order(frame, customs):
        group = frame.loc[[name] + [s.name for s in hub[name].walk()]]
        if body:
            body.append("\\cmidrule(lr){1-12}")
        body += _component_rows(group, indent=set(group.index) - {name})
    s6 = _longtable("l l l l l l l l l l l l",
                    "Declarative definition of the custom components, i.e., of their subcomponents.",
                    "tab:S_custom_comps", COMP_HEADER, body, 12, size="\\small", tabcolsep="3pt")  # fmt: skip
    return {"S5_components": s5, "S6_custom": s6}


def customs_in_order(frame: pd.DataFrame, customs: set[str]) -> list[str]:
    return [i for i in frame.index if i in customs]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
# categorical slots 1-3 of the reference palette (dataviz skill), validated for
# all pairs in light mode; every series also has its own marker and a label
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MM = 1 / 25.4


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })  # fmt: skip


def _line(ax, x, y, i, label):
    ax.plot(x, y, color=SERIES[i], marker=MARKERS[i], label=label,
            markeredgecolor="white", markeredgewidth=0.6)  # fmt: skip


def _t_axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def _panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left")


def r1_figure(runs: dict[str, Run], path: Path) -> bool:
    import matplotlib.pyplot as plt

    scs = [sc for sc in sorted(scenarios.R1(), key=lambda s: s.t) if sc.name in runs]
    if len(scs) < 2 or "R1-1h" not in runs:
        return False
    _style()
    t = np.array([sc.t for sc in scs])
    rr = [runs[sc.name] for sc in scs]
    ref = runs["R1-1h"]
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), constrained_layout=True)

    ax = axes[0]
    _line(ax, t, [r.perf("total") for r in rr], 0, "Total")
    _line(ax, t, [r.perf("solving") for r in rr], 1, "Solving")
    _line(ax, t, [r.perf("formulation") for r in rr], 2, "Formulation")
    ax.set_yscale("log")
    _panel_title(ax, "a", "Computation time [s]")
    ax.legend(loc="upper right")

    ax = axes[1]
    _line(ax, t, [r.perf("memory") for r in rr], 0, "Peak memory")
    ax.set_ylim(bottom=0)
    _panel_title(ax, "b", "Peak memory [MB]")

    ax = axes[2]
    def rel(f):
        base = f(ref)
        return [100 * (f(r) - base) / abs(base) for r in rr]
    _line(ax, t, rel(lambda r: r.economics()["NPV"]), 0, "NPV")
    _line(ax, t, rel(lambda r: r.size("ELYS")), 1, "Electrolyser size")
    price = rel(lambda r: r.mean_import_price())
    if not all(math.isnan(p) for p in price):
        _line(ax, t, price, 2, "Mean import price")
    ax.axhline(0, color=INK_2, linewidth=0.6)
    _panel_title(ax, "c", "Change relative to 1 h [%]")
    ax.legend(loc="lower left")

    for ax in axes:
        _t_axis(ax, t)
        ax.set_xlabel("Sampling time $t$ [h]")
    fig.savefig(path)
    plt.close(fig)
    return True


def r2_figure(runs: dict[str, Run], path: Path) -> bool:
    import matplotlib.pyplot as plt

    grid = [sc for sc in scenarios.R2() if sc.name in runs]
    if len(grid) < 2:
        return False
    _style()
    fasts = sorted({sc.t for sc in grid})
    slows = sorted({sc.t_of("ELYS") for sc in grid})

    def series(t_fast, t_med, get):
        pts = sorted((sc.t_of("ELYS"), get(runs[sc.name])) for sc in grid
                     if sc.t == t_fast and sc.t_of("PVP") == t_med)  # fmt: skip
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 112 * MM), constrained_layout=True)
    (a, b), (c, d) = axes

    for i, t_med in enumerate((1.0, 2.0)):  # NPV at the finest fast sampling
        x, y = series(fasts[0], t_med, lambda r: r.economics()["NPV"] / 1e6)
        if x:
            _line(a, x, y, i, f"$t_\\mathrm{{med}}$ = {t_med:g} h")
    _panel_title(a, "a", f"NPV [million €], $t_\\mathrm{{fast}}$ = {fasts[0]:g} h")
    a.legend(loc="lower left")

    for ax, t_med, letter in ((b, 1.0, "b"), (c, 2.0, "c")):
        for i, f in enumerate(fasts):
            x, y = series(f, t_med, lambda r: r.size("BATT"))
            if x:
                _line(ax, x, y, i, f"$t_\\mathrm{{fast}}$ = {f:g} h")
        _panel_title(ax, letter, f"Battery size [kWh], $t_\\mathrm{{med}}$ = {t_med:g} h")
        ax.legend(loc="upper left")
    c.sharey(b)

    for i, f in enumerate(fasts):
        x, y = series(f, 1.0, lambda r: r.perf("solving"))
        if x:
            _line(d, x, y, i, f"$t_\\mathrm{{fast}}$ = {f:g} h")
    d.set_ylim(bottom=0)
    _panel_title(d, "d", "Solving time [s], $t_\\mathrm{med}$ = 1 h")
    d.legend(loc="upper left")

    for ax in axes.flat:
        ax.set_xticks(slows)
        ax.set_xticklabels([f"{s:g}" for s in slows])
        ax.set_xlabel("Slow sampling time $t_\\mathrm{slow}$ [h]")
    fig.savefig(path)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# preview document
# --------------------------------------------------------------------------- #
PREVIEW = r"""%% generated by python/depot/report.py: compiles every generated table and figure
\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage[a4paper, margin=2cm]{geometry}
\usepackage{booktabs, longtable, makecell, graphicx, amsmath, amssymb}
\usepackage{caption}
\begin{document}
%(body)s
\end{document}
"""


def write_all(store: ResultStore, out: Path, model: str) -> list[Path]:
    names = [sc.name for sc in scenarios.R1() + scenarios.R2()]
    runs = latest_runs(store, names, model)
    tables, img = out / "tables", out / "img"
    tables.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(name: str, text: str) -> None:
        p = tables / f"{name}.tex"
        p.write_text(text, encoding="utf-8", newline="\n")
        written.append(p)

    if any(n in runs for n in (sc.name for sc in scenarios.R1())):
        for k, v in r1_tables(runs).items():
            save(k, v)
    if any(n in runs for n in (sc.name for sc in scenarios.R2())):
        for k, v in r2_tables(runs).items():
            save(k, v)
    for k, frame in csv_tables(runs).items():
        p = tables / f"{k}.csv"
        frame.to_csv(p)
        written.append(p)

    hub = build_hub(Scenario())
    save("S3_costs", s3_table(hub))
    for k, v in s5_s6_tables(hub).items():
        save(k, v)

    for fig, f in (("R1_sampling", r1_figure), ("R2_mixed", r2_figure)):
        p = img / f"{fig}.pdf"
        if f(runs, p):
            written.append(p)

    parts = [f"\\input{{{p.stem}}}\n\\clearpage" for p in written
             if p.suffix == ".tex" and p.parent == tables]  # fmt: skip
    parts += [
        f"\\begin{{figure}}[ht]\\centering\\includegraphics[width=\\textwidth]{{../img/{p.stem}}}"
        f"\\caption{{{p.stem.replace('_', chr(92) + '_')}}}\\end{{figure}}"
        for p in written if p.suffix == ".pdf"
    ]  # fmt: skip
    preview = tables / "preview.tex"
    preview.write_text(PREVIEW % {"body": "\n".join(parts)}, encoding="utf-8", newline="\n")

    missing = [n for n in names if n not in runs]
    print(f"runs used: {len(runs)} of {len(names)} ({model})")
    if missing:
        print("missing:", ", ".join(missing))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m depot.report", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--store", default="results", help="result directory below the repository root")
    parser.add_argument("--out", default=str(MANUSCRIPT), help="output folder (default: the repository root)")
    parser.add_argument("--model", default="opt3s", choices=("opt3s", "legacy", "any"))
    args = parser.parse_args(argv)
    for p in write_all(ResultStore(PYTHON_ROOT / args.store), Path(args.out), args.model):
        print("wrote", p)


if __name__ == "__main__":
    main()
