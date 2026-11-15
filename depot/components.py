"""The components of the fleet depot EH, one factory per row of Supplementary
Tables S5 and S6.  Interconnections are the ``Other`` ports, as drawn in
Fig. 1 and Supplementary Fig. S1.

Every factory takes the parameter group of :mod:`depot.params`, the sampling
time ``t`` and, where needed, the input data context ``ctx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from opt3s import (
    FC, UC, Amount, Costs, Custom, LinComb, Other, PeakBilling, Profile, Size, Standard,
    State,
)  # fmt: skip

from . import inputs
from .params import H2_HHV

E, HEAT, H2, O2, W = "electricity", "heat", "hydrogen", "oxygen", "water"


@dataclass(frozen=True)
class Context:
    """What the input data depend on: the horizon."""

    start: datetime
    n_days: int


def _costs(p: dict[str, Any], degradation: str = "cal", **kw: Any) -> Costs:
    return Costs(
        c_inv=p.get("c_inv", 0.0),
        c_mnt=p.get("c_mnt", 0.0),
        c_rpl=p.get("c_rpl", 0.0),
        n_lt=p.get("n_lt"),
        degradation=degradation,
        **kw,
    )


# --- electricity ------------------------------------------------------------
def supply_prices(P: dict, ctx: Context):
    """Purchase and sale price of electricity [€/kWh] as time series."""
    p = P["SUPP"]
    if p["price"] == "day-ahead":
        return inputs.prices(ctx.start, p["margin"])
    if p["price"] == "tariff":
        e = P["EGC"]
        c_opr = inputs.tariff_prices(ctx.start, ctx.n_days, p["tariff"],
                                     e["high_start"], e["high_end"])  # fmt: skip
        return c_opr, (p["tariff_sale"] * c_opr).rename("c_rev")
    raise ValueError(f"SUPP.price must be 'day-ahead' or 'tariff', not {p['price']!r}")


def import_fee(P: dict, ctx: Context):
    """Grid fee of imported electricity [€/kWh] as a time series."""
    p = P["EGC"]
    return inputs.grid_fee(ctx.start, ctx.n_days, p["fee_high"], p["fee_low"],
                           p["high_start"], p["high_end"])  # fmt: skip


def supp(P: dict, t: float, ctx: Context) -> Custom:
    """Electricity supply: purchase (ESRC) and sale (ESNK).

    ``SUPP.price`` selects the prices: ``"day-ahead"`` (market price ± margin) or
    ``"tariff"`` (two-tariff purchase price, sale price ``tariff_sale`` of it).
    """
    c_opr, c_rev = supply_prices(P, ctx)
    esrc = Standard("ESRC", "source", {"y1": FC(E, c_opr=c_opr)}, label="Electricity source")
    esnk = Standard("ESNK", "sink", {"u1": FC(E, c_rev=c_rev)}, label="Electricity sink")
    return Custom("SUPP", [esrc, esnk], ports={"y1": "ESRC.y1", "u1": "ESNK.u1"},
                  t=t, label="Electricity supply")  # fmt: skip


def pvp(P: dict, t: float, ctx: Context) -> Custom:
    """PV plant: three 1 kWp-normalised arrays (DP) feeding one inverter (BLC)."""
    pa, pi = P["PVA"], P["PVI"]
    arrays = [
        Standard(
            f"PVA{a + 1}", "source",
            {"y1": Profile(E, inputs.pv_profile(a, ctx.start))},
            size=Size.by_profile(s_max=pa["s_max"]),
            costs=_costs(pa), label=f"PV array {a + 1}",
        )  # fmt: skip
        for a in range(3)
    ]
    pvi = Standard(
        "PVI", "process",
        {"u1": Other("PVA1.y1"), "u2": Other("PVA2.y1"), "u3": Other("PVA3.y1"),
         "y1": LinComb(E, {"u1": pi["eta"], "u2": pi["eta"], "u3": pi["eta"]},
                       bounded=True)},
        size=Size.max_of("y1", s_max=pi["s_max"]), costs=_costs(pi), label="PV inverter",
    )  # fmt: skip
    return Custom("PVP", arrays + [pvi], ports={"y1": "PVI.y1"}, t=t, label="PV plant")


def bess(P: dict, t: float) -> Custom:
    """Battery energy storage system, the worked example of section Case study."""
    pb, pc_ = P["BATT"], P["PC"]
    pc = Standard(
        "PC", "process",
        {"u1": FC(E), "y1": FC(E),
         "u2": LinComb(E, {"y1": 1 / pc_["eta_dch"]}),
         "y2": LinComb(E, {"u1": pc_["eta_chg"]})},
        size=Size.max_of("u1", "y1"), costs=_costs(pc_), label="Power converter",
    )  # fmt: skip
    c_deg_use = pb["c_rpl"] / (2 * pb["n_cyc"] * pb["dod"])  # eq. (batt_c_deg_use)
    batt = Standard(
        "BATT", "storage",
        {"u1": Other("PC.y2"), "y1": Other("PC.u2")},
        states={"x1": State({"u1": 1.0, "y1": 1.0}, gamma=1 - pb["dod"])},
        size=Size.max_of_state("x1"),
        costs=_costs(pb, "cmb", c_deg_use=c_deg_use, deg_ports=("u1", "y1")),
        label="Battery",
    )  # fmt: skip
    return Custom(
        "BESS", [batt, pc], ports={"u1": "PC.u1", "y1": "PC.y1"},
        ratios=[("PC", "<=", P["BESS"]["beta_PC_BATT"], "BATT")],
        t=t, label="Battery energy storage system",
    )  # fmt: skip


def evc(P: dict, t: float, ctx: Context) -> Custom:
    """EV chargers: one sink per charger (DA), joined by the EV bus."""
    pe = P["EVC"]
    chargers = [
        Standard(
            f"EV{e + 1}", "sink",
            {"u1": Amount(E, inputs.ev_windows(s, ctx.start, ctx.n_days),
                          rounding="overlap", p_max=pe["p_max"])},
            label=f"EV charger {e + 1}",
        )  # fmt: skip
        for e, s in enumerate(pe["stations"])
    ]
    n = len(chargers)
    evb = Standard(
        "EVB", "bus",
        {**{f"y{e + 1}": Other(f"EV{e + 1}.u1") for e in range(n)},
         "u1": LinComb(E, {f"y{e + 1}": 1.0 for e in range(n)})},
        label="EV bus",
    )  # fmt: skip
    return Custom("EVC", chargers + [evb], ports={"u1": "EVB.u1"}, t=t, label="EV chargers")


def fec(P: dict, t: float, ctx: Context) -> Standard:
    profile = inputs.building_profile("LargeOffice.csv", ctx.start, P["FEC"]["annual_energy"])
    return Standard("FEC", "sink", {"u1": UC(E, profile)}, t=t, label="Fixed el. consumption")


def egc(P: dict, t: float, ctx: Context, balance: str = "eq",
        grid_ports: bool = True) -> Standard:  # fmt: skip
    """Electrical grid connection, the electricity bus of the hub.

    ``grid_ports=False`` leaves out the ports to the electricity supply, the
    balance and the size rule; :mod:`depot.net_grid` then closes the bus with a
    signed net grid power.
    """
    p = P["EGC"]
    ports = {"u1": Other("SUPP.y1", c_opr=import_fee(P, ctx), peak=PeakBilling(p["c_peak_import"])),
             "u2": Other("PVP.y1"),
             "u3": Other("BESS.y1"),
             "u4": Other("FCS.y1"),
             "y1": Other("SUPP.u1", c_opr=p["fee_export"], peak=PeakBilling(p["c_peak_export"])),
             **_egc_loads()}  # fmt: skip
    if grid_ports:
        size = Size.max_of("u1", "y1", s_exist=p["s_exist"], s_max=p["s_max"])
    else:
        del ports["u1"], ports["y1"]
        size, balance = Size.by_profile(s_exist=p["s_exist"], s_max=p["s_max"]), "open"
    return Standard("EGC", "bus", ports, size=size, costs=_costs(p, "none"), balance=balance,
                    t=t, label="El. grid connection")  # fmt: skip


def _egc_loads() -> dict:
    return {
        "y2": Other("BESS.u1"),
        "y3": Other("EVC.u1"),
        "y4": Other("FEC.u1"),
        "y5": Other("ELYS.u1"),
        "y6": Other("HCMP1.u2"),
        "y7": Other("HCMP2.u2"),
        "y8": Other("OCMP.u2"),
        "y9": Other("DMN.u2"),
        "y10": Other("RCS.u1"),
    }


# --- hydrogen -----------------------------------------------------------------
def elys(P: dict, t: float) -> Standard:
    p = P["ELYS"]
    h2 = p["eta_el"] / H2_HHV  # kg of hydrogen per kWh of electricity
    return Standard(
        "ELYS", "process",
        {"u1": FC(E),
         "u2": LinComb(W, {"u1": p["water_per_h2"] * h2}),
         "y1": LinComb(H2, {"u1": h2}),
         "y2": LinComb(O2, {"u1": p["o2_per_h2"] * h2}),
         "y3": LinComb(HEAT, {"u1": p["eta_ht"] * (1 - p["eta_el"])})},
        size=Size.max_of("u1"), costs=_costs(p), t=t, label="Electrolyser",
    )  # fmt: skip


def hcmp1(P: dict, t: float) -> Standard:
    p = P["HCMP1"]
    return Standard(
        "HCMP1", "process",
        {"u1": Other("ELYS.y1"),
         "u2": LinComb(E, {"u1": p["e_per_kg"]}),
         "y1": LinComb(H2, {"u1": 1.0})},
        size=Size.max_of("u1"), costs=_costs(p), t=t, label="H2 compressor 1",
    )  # fmt: skip


def htnk(P: dict, t: float) -> Standard:
    return Standard(
        "HTNK", "storage",
        {"u1": Other("HCMP1.y1"), "y1": Other("GGI.u1"),
         "y2": Other("HCMP2.u1"), "y3": Other("FCS.u1")},
        states={"x1": State({"u1": 1.0, "y1": 1.0, "y2": 1.0, "y3": 1.0})},
        size=Size.max_of_state("x1"), costs=_costs(P["HTNK"]), t=t, label="H2 tank",
    )  # fmt: skip


def hcmp2(P: dict, t: float) -> Standard:
    p = P["HCMP2"]
    return Standard(
        "HCMP2", "process",
        {"y1": Other("HSLS.u1"),
         "u1": LinComb(H2, {"y1": 1.0}),
         "u2": LinComb(E, {"u1": p["e_per_kg"]})},
        size=Size.max_of("u1"), costs=_costs(p), t=t, label="H2 compressor 2",
    )  # fmt: skip


def hsls(P: dict, t: float, ctx: Context) -> Standard:
    p = P["HSLS"]
    windows = inputs.daily_windows(p["windows"], ctx.start, ctx.n_days)
    return Standard(
        "HSLS", "sink", {"u1": Amount(H2, windows, bounded=True, c_rev=p["c_rev"])},
        t=t, label="H2 sales",
    )  # fmt: skip


def ggi(P: dict, t: float) -> Standard:
    p = P["GGI"]
    return Standard(
        "GGI", "sink", {"u1": FC(H2, c_rev=p["c_rev"])},
        size=Size.max_of("u1", s_max=p["s_max"]), costs=_costs(p, "none"),
        t=t, label="Gas grid injection",
    )  # fmt: skip


def fcs(P: dict, t: float) -> Standard:
    p = P["FCS"]
    return Standard(
        "FCS", "process",
        {"y1": FC(E),
         "u1": LinComb(H2, {"y1": 1 / p["eta_el"] / H2_HHV}),
         "y2": LinComb(HEAT, {"y1": p["eta_ht"] * (1 - p["eta_el"]) / p["eta_el"]}),
         "y3": LinComb(W, {"y1": p["water_per_h2"] * p["water_recovered"]
                                 / p["eta_el"] / H2_HHV})},
        size=Size.max_of("y1"), costs=_costs(p), t=t, label="Fuel cell system",
    )  # fmt: skip


# --- oxygen -------------------------------------------------------------------
def ocmp(P: dict, t: float) -> Standard:
    p = P["OCMP"]
    return Standard(
        "OCMP", "process",
        {"u1": Other("ELYS.y2", bounded=True),
         "u2": LinComb(E, {"u1": p["e_per_kg"]}),
         "y1": LinComb(O2, {"u1": 1.0})},
        size=Size.max_of("u1"), costs=_costs(p), t=t, label="O2 compressor",
    )  # fmt: skip


def otnk(P: dict, t: float) -> Standard:
    return Standard(
        "OTNK", "storage",
        {"u1": Other("OCMP.y1"), "y1": Other("OSLS.u1")},
        states={"x1": State({"u1": 1.0, "y1": 1.0})},
        size=Size.max_of_state("x1"), costs=_costs(P["OTNK"]), t=t, label="O2 tank",
    )  # fmt: skip


def osls(P: dict, t: float, ctx: Context) -> Standard:
    p = P["OSLS"]
    windows = inputs.daily_windows(p["windows"], ctx.start, ctx.n_days)
    return Standard(
        "OSLS", "sink", {"u1": Amount(O2, windows, bounded=True, c_rev=p["c_rev"])},
        t=t, label="O2 sales",
    )  # fmt: skip


# --- water --------------------------------------------------------------------
def dmn(P: dict, t: float) -> Standard:
    p = P["DMN"]
    return Standard(
        "DMN", "process",
        {"y1": Other("ELYS.u2"),
         "u1": LinComb(W, {"y1": p["water_per_product"]}),
         "u2": LinComb(E, {"u1": p["e_per_kg"]})},
        size=Size.max_of("u2"), costs=_costs(p), t=t, label="Demineralizer",
    )  # fmt: skip


def wb(P: dict, t: float) -> Standard:
    """Water bus: the grid supplies what the rain collection does not (u1 = y1 − u2)."""
    return Standard(
        "WB", "bus",
        {"u2": Other("RCS.y1"), "y1": Other("DMN.u1"),
         "u1": LinComb(W, {"y1": 1.0, "u2": -1.0})},
        t=t, label="Water bus",
    )  # fmt: skip


def wgs(P: dict, t: float) -> Standard:
    p = P["WGS"]
    return Standard(
        "WGS", "source", {"y1": Other("WB.u1", c_opr=p["c_opr"])},
        size=Size.max_of("y1", s_exist=p["s_exist"], s_max=p["s_max"]),
        costs=_costs(p, "none"), t=t, label="Water grid supply",
    )  # fmt: skip


def rcs(P: dict, t: float, ctx: Context) -> Custom:
    """Rain collection system: collection area (BP), water tank and pump."""
    pr, pp, pt = P["RAIN"], P["WPMP"], P["WTNK"]
    rain = Standard(
        "RAIN", "source", {"y1": Profile(W, inputs.rainfall(ctx.start), bounded=True)},
        size=Size.by_profile(s_max=pr["s_max"]), costs=_costs(pr), label="Rain",
    )  # fmt: skip
    wpmp = Standard(
        "WPMP", "process",
        {"y1": FC(W), "u1": LinComb(W, {"y1": 1.0}),
         "u2": LinComb(E, {"y1": pp["e_per_kg"]})},
        size=Size.max_of("u2"), costs=_costs(pp), label="Water pump",
    )  # fmt: skip
    wtnk = Standard(
        "WTNK", "storage",
        {"u1": Other("RAIN.y1"), "u2": Other("FCS.y3", bounded=True), "y1": Other("WPMP.u1")},
        states={"x1": State({"u1": 1.0, "u2": 1.0, "y1": 1.0})},
        size=Size.max_of_state("x1"), costs=_costs(pt), label="Water tank",
    )  # fmt: skip
    return Custom(
        "RCS", [rain, wpmp, wtnk],
        ports={"u1": "WPMP.u2", "u2": "WTNK.u2", "y1": "WPMP.y1"},
        t=t, label="Rain collection system",
    )  # fmt: skip


# --- heat ---------------------------------------------------------------------
def blr(P: dict, t: float) -> Standard:
    p = P["BLR"]
    return Standard(
        "BLR", "source", {"y1": FC(HEAT, c_opr=p["c_opr"])},
        size=Size.max_of("y1", s_exist=p["s_exist"], s_max=p["s_max"]),
        costs=_costs(p), t=t, label="Boiler",
    )  # fmt: skip


def hts(P: dict, t: float, balance: str = "ineq") -> Standard:
    """Heating system, the heat bus of the hub.

    ``balance="ineq"``: all ports DO and ``Σ u_i >= y_1``, i.e. surplus waste heat
    of the electrolyser and the fuel cell is released at the bus.
    ``balance="eq"``: the two waste-heat inputs are BO and the balance is
    eq. (bus_eq), i.e. the surplus is released at the ports instead.
    """
    p = P["HTS"]
    waste_heat_bounded = balance == "eq"
    return Standard(
        "HTS", "bus",
        {"u1": Other("ELYS.y3", bounded=waste_heat_bounded),
         "u2": Other("BLR.y1"),
         "u3": Other("FCS.y2", bounded=waste_heat_bounded),
         "y1": Other("FHC.u1")},
        size=Size.max_of_sum("u1", "u2", "u3", s_exist=p["s_exist"]),
        costs=_costs(p), balance=balance, t=t, label="Heating system",
    )  # fmt: skip


def fhc(P: dict, t: float, ctx: Context) -> Standard:
    profile = inputs.building_profile(
        "LargeOffice_heat.csv", ctx.start, P["FHC"]["annual_energy"]
    )
    return Standard("FHC", "sink", {"u1": UC(HEAT, profile)}, t=t, label="Fixed heat consumption")
