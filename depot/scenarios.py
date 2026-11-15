"""Named scenario sets of section Results.

* ``R1``: one sampling time for all components, 0.25 h to 6 h.
* ``R2``: fast / medium / slow component groups (Table "Groups of components").
* ``R3``: bus equality vs. inequality.  ``R3a``: the heating system (HTS) with
  the equality balance, for four R2 scenarios, which are themselves the
  inequality brothers.  ``R3b``: the grid connection (EGC) with the inequality
  balance and day-ahead prices (the R2 scenario is the equality brother), and
  with both balances and two-tariff prices.
* ``R4``: electricity exchange with the grid as one signed net power instead of
  two non-negative ports, for three R2 scenarios, which are themselves the
  brothers with two ports.  ``-net``: the net power is an intermediate variable
  (previous work); ``-free``: it is a decision variable.  See ``depot.net_grid``.

Solver and solver options are not set here; they come from
``Scenario.solver_name`` (environment variable ``OPT3S_SOLVER``) and
:data:`depot.runner.SOLVER_DEFAULTS`, or from the command line.
"""

from __future__ import annotations

from itertools import product

from scenario import Scenario

FAST = ("SUPP", "BESS", "EVC", "FCS", "HTNK", "EGC")
MEDIUM = ("PVP", "FEC", "FHC", "HTS", "BLR", "RCS")
SLOW = ("ELYS", "HCMP1", "GGI", "HCMP2", "HSLS", "OCMP", "OTNK", "OSLS", "DMN", "WB", "WGS")  # fmt: skip


def r1(t: float, **kw) -> Scenario:
    return Scenario(name=f"R1-{t:g}h", notes=f"same sampling for all comps - {t:g} h", t=t, **kw)


def r2(t_fast: float, t_medium: float, t_slow: float, **kw) -> Scenario:
    """Fast components use the general ``t``, the others are set per component."""
    return Scenario(
        name=f"R2-{t_fast:g}h-{t_medium:g}h-{t_slow:g}h",
        notes=f"sampling: fast {t_fast:g} h, medium {t_medium:g} h, slow {t_slow:g} h",
        t=t_fast,
        t_comp={**{c: t_medium for c in MEDIUM}, **{c: t_slow for c in SLOW}},
        **kw,
    )


def R1() -> list[Scenario]:
    return [r1(t) for t in (6, 4, 3, 2, 1, 0.5, 0.25)]


def R2() -> list[Scenario]:
    # 3 h slow with 2 h medium is not an integer ratio (HTS ← ELYS, WB ← RCS)
    return [
        r2(f, m, s)
        for f, m, s in product((0.25, 0.5, 1), (1, 2), (2, 3, 4, 6))
        if not (m == 2 and s == 3)
    ]


#: R2 scenarios (t_fast, t_med, t_slow) whose heating system balance is compared in R3a
R3A_BASES = ((1, 2, 6), (1, 1, 4), (0.5, 1, 4), (0.25, 1, 4))
#: R2 scenario whose grid connection balance is compared in R3b
R3B_BASE = (1, 1, 4)
#: parameter override that replaces the day-ahead prices by the two-tariff prices
TARIFF = {"SUPP.price": "tariff"}


def _tag(base: Scenario) -> str:
    return base.name[3:]  # "R2-1h-1h-4h" -> "1h-1h-4h"


def R3a() -> list[Scenario]:
    """Heating system with the equality balance; the brothers are the R2 scenarios."""
    out = []
    for times in R3A_BASES:
        base = r2(*times)
        out.append(
            base.replace(
                name=f"R3a-{_tag(base)}",
                notes=f"{base.notes}; HTS balance as equality",
                hts_balance="eq",
            )
        )
    return out


def R3b_DA() -> list[Scenario]:
    """Grid connection with the inequality balance and day-ahead prices."""
    base = r2(*R3B_BASE)
    return [
        base.replace(
            name=f"R3b-{_tag(base)}-DA",
            notes=f"{base.notes}; EGC balance as inequality, day-ahead prices",
            egc_equality=False,
        )
    ]


def R3b_HEP(hts_balance: str = "ineq") -> list[Scenario]:
    """Grid connection with the equality and the inequality balance, two-tariff prices."""
    base = r2(*R3B_BASE)
    return [
        base.replace(
            name=f"R3b-{_tag(base)}-HEP-{'eq' if egc_eq else 'ineq'}",
            notes=f"{base.notes}; EGC balance as {'equality' if egc_eq else 'inequality'}, "
            "two-tariff prices",
            egc_equality=egc_eq,
            hts_balance=hts_balance,
            params=dict(TARIFF),
        )
        for egc_eq in (True, False)
    ]


def R3() -> list[Scenario]:
    return R3a() + R3b_DA() + R3b_HEP()


#: R2 scenarios (t_fast, t_med, t_slow) solved again with a signed net grid power
R4_BASES = ((1, 1, 4), (0.5, 1, 4), (0.25, 1, 4))
R4_VARIANTS = ("net", "free")


def R4() -> list[Scenario]:
    """Signed net grid power; the brothers with two ports are the R2 scenarios."""
    words = {"net": "an intermediate variable", "free": "a decision variable"}
    return [
        base.replace(
            name=f"R4-{_tag(base)}-{variant}",
            notes=f"{base.notes}; net grid power as {words[variant]}",
            grid_exchange=variant,
        )
        for variant in R4_VARIANTS
        for base in (r2(*times) for times in R4_BASES)
    ]


SETS = {"R1": R1, "R2": R2, "R3": R3, "R4": R4}


def get(name: str) -> list[Scenario]:
    """A named set (``R1`` ... ``R4``, ``all``) or one scenario by its name."""
    if name == "all":
        return [sc for f in SETS.values() for sc in f()]
    if name in SETS:
        return SETS[name]()
    for f in SETS.values():
        for sc in f():
            if sc.name == name:
                return [sc]
    raise KeyError(f"no scenario set or scenario named {name!r}")
