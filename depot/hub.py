"""Assembly of the fleet depot EH (Fig. 1) for one scenario."""

from __future__ import annotations

from scenario import Scenario

from opt3s import Horizon, Hub

from . import components as cmp
from .params import get_params

#: top-level components in the order of Supplementary Table S5
COMPONENT_NAMES = (
    "SUPP", "PVP", "BESS", "EVC", "FEC", "ELYS", "HCMP1", "HTNK", "HCMP2", "HSLS",
    "GGI", "FCS", "OCMP", "OTNK", "OSLS", "DMN", "WB", "WGS", "RCS", "BLR", "HTS",
    "FHC", "EGC",
)  # fmt: skip


def horizon(sc: Scenario) -> Horizon:
    return Horizon(sc.start_date, sc.N_days)


def build_hub(sc: Scenario) -> Hub:
    """The hub of scenario ``sc``: parameters with overrides, sampling times, variants."""
    unknown = set(sc.t_comp) - set(COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"t_comp names unknown components: {sorted(unknown)}")
    P = get_params(sc.params)
    ctx = cmp.Context(sc.start_date, sc.N_days)
    t = sc.t_of
    egc_balance = "eq" if sc.egc_equality else "ineq"
    split = sc.grid_exchange == "split"  # else: signed net grid power, depot.net_grid
    if not split and not sc.egc_equality:
        raise ValueError("grid_exchange 'net' and 'free' need egc_equality=True")

    parts = {
        "SUPP": cmp.supp(P, t("SUPP"), ctx),
        "PVP": cmp.pvp(P, t("PVP"), ctx),
        "BESS": cmp.bess(P, t("BESS")),
        "EVC": cmp.evc(P, t("EVC"), ctx),
        "FEC": cmp.fec(P, t("FEC"), ctx),
        "ELYS": cmp.elys(P, t("ELYS")),
        "HCMP1": cmp.hcmp1(P, t("HCMP1")),
        "HTNK": cmp.htnk(P, t("HTNK")),
        "HCMP2": cmp.hcmp2(P, t("HCMP2")),
        "HSLS": cmp.hsls(P, t("HSLS"), ctx),
        "GGI": cmp.ggi(P, t("GGI")),
        "FCS": cmp.fcs(P, t("FCS")),
        "OCMP": cmp.ocmp(P, t("OCMP")),
        "OTNK": cmp.otnk(P, t("OTNK")),
        "OSLS": cmp.osls(P, t("OSLS"), ctx),
        "DMN": cmp.dmn(P, t("DMN")),
        "WB": cmp.wb(P, t("WB")),
        "WGS": cmp.wgs(P, t("WGS")),
        "RCS": cmp.rcs(P, t("RCS"), ctx),
        "BLR": cmp.blr(P, t("BLR")),
        "HTS": cmp.hts(P, t("HTS"), balance=sc.hts_balance),
        "FHC": cmp.fhc(P, t("FHC"), ctx),
        "EGC": cmp.egc(P, t("EGC"), ctx, balance=egc_balance, grid_ports=split),
    }
    assert tuple(parts) == COMPONENT_NAMES
    if not split:
        del parts["SUPP"]
    return Hub("depot", list(parts.values()), label="Fleet depot energy hub")
