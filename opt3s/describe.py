"""Declarative description of a hub, in the layout of Table 1 and Supplementary
Tables S5 and S6: type of every port, states, the quantity defining the size,
and the active cost terms."""

from __future__ import annotations

import pandas as pd

from .components import Component, Custom, Hub, Standard

ACTIVE, INACTIVE = "x", "-"  # bullet and dash of the tables, kept ASCII for consoles
COLUMNS = ["label", "type", "ports", "states", "size",
           "C_inv", "C_mnt", "C_deg", "C_peak", "C_opr", "C_rev"]  # fmt: skip


def _size(c: Standard) -> str:
    s = c.size
    if s is None:
        return INACTIVE
    if s.s_new_max == 0:
        return "S_exist"
    if s.rule == "profile":
        return ACTIVE
    items = list(s.items)
    if s.rule in ("ports", "states"):
        return f"max {items[0]}" if len(items) == 1 else f"max({', '.join(items)})"
    return "max " + " + ".join(items)


def _costs(c: Standard) -> dict[str, str]:
    k = c.costs
    expandable = c.size is not None and c.size.s_new_max > 0
    deg = {"none": INACTIVE, "cal": "cal", "use": "use", "cmb": "cmb"}[k.degradation]
    return {
        "C_inv": ACTIVE if k.c_inv and expandable else INACTIVE,
        "C_mnt": ACTIVE if k.c_mnt else INACTIVE,
        "C_deg": deg,
        "C_peak": ACTIVE if any(p.peak for p in c.ports.values()) else INACTIVE,
        "C_opr": ACTIVE if any(p.c_opr is not None for p in c.ports.values()) else INACTIVE,
        "C_rev": ACTIVE if any(p.c_rev is not None for p in c.ports.values()) else INACTIVE,
    }


def _row(c: Component) -> dict[str, str]:
    if isinstance(c, Standard):
        return {
            "label": c.label,
            "type": c.kind,
            "ports": ", ".join(f"{p} as {s.abbrev}" for p, s in c.ports.items()),
            "states": ", ".join(c.states) or INACTIVE,
            "size": _size(c),
            **_costs(c),
        }
    subs = [_row(s) for s in c.components]
    costs = {
        term: ACTIVE if any(r[term] != INACTIVE for r in subs) else INACTIVE
        for term in COLUMNS[5:]
    }
    return {
        "label": c.label,
        "type": "custom",
        "ports": ", ".join(f"{p} = {t}" for p, t in c.ports.items()),
        "states": INACTIVE,
        "size": INACTIVE,
        **costs,
    }


def describe(hub: Hub, custom_parts: bool = False) -> pd.DataFrame:
    """One row per top-level component (Supplementary Table S5), or, with
    ``custom_parts=True``, per custom component and its subcomponents (Table S6)."""
    rows: dict[str, dict[str, str]] = {}
    if not custom_parts:
        for c in hub.components:
            rows[c.name] = _row(c)
    else:
        for c in hub.components:
            if isinstance(c, Custom):
                rows[c.name] = _row(c)
                for s in c.walk():
                    rows[s.name] = _row(s)
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)
    frame.index.name = "abbrev"
    return frame
