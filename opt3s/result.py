"""Reading a solved hub: economics, sizes, cost terms, amounts and time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .backend import scalar, value_of
from .build import COST_TERMS, BuiltHub
from .components import Custom
from .ports import UNITS

#: unit of an amount ``t Σ P`` for each port unit
AMOUNT_UNITS = {"kW": "kWh", "kW_th": "kWh_th", "kg/h": "kg"}


class HubResult:
    """The solution of a :class:`~opt3s.build.BuiltHub`, in physical units and euros."""

    def __init__(self, built: BuiltHub) -> None:
        self.built = built
        if str(built.model.status) != "ok":
            raise RuntimeError(f"model not solved: {built.model.status}")

    # --- economics -----------------------------------------------------------
    def economics(self) -> dict[str, float]:
        """CAPEX, OPEX, REV, annual profit, NPV (= −objective) and the annuity factor."""
        b = self.built
        capex, opex, rev = (scalar(b.CAPEX), scalar(b.OPEX), scalar(b.REV))
        npc = capex + b.annuity * (opex - rev)
        return {
            "CAPEX": capex,
            "OPEX": opex,
            "REV": rev,
            "profit": rev - opex,
            "NPV": -npc,
            "NPC": npc,
            "annuity": b.annuity,
        }

    # --- per component -------------------------------------------------------
    def costs(self) -> pd.DataFrame:
        """Cost terms of every (sub)component in euros (per year except C_inv)."""
        rows = {
            name: {term: scalar(terms[term]) for term in COST_TERMS}
            for name, terms in self.built.costs.items()
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "component"
        top = [c.name for c in self.built.hub.components]
        frame.loc["TOTAL"] = frame.loc[top].sum()
        return frame

    def sizes(self) -> pd.DataFrame:
        rows = {}
        for name, s in self.built.sizes.items():
            rows[name] = {
                "S_exist": s["S_exist"],
                "S_max": np.nan if s["S_max"] is None else s["S_max"],
                "S_new": scalar(s["S_new"]),
                "S_total": scalar(s["S_total"]),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "component"
        return frame

    # --- ports ---------------------------------------------------------------
    def port(self, key: str) -> xr.DataArray:
        """Values of port ``"COMP.port"`` over the instants of its component."""
        value = value_of(self.built.ports[key])
        if not isinstance(value, xr.DataArray):
            comp = key.split(".")[0]
            grid = self.built.grids[comp]
            value = xr.DataArray(np.full(grid.n_h, value), coords=[grid.index])
        return value

    def amounts(self, annual: bool = True) -> pd.DataFrame:
        """Amount exchanged through every port, ``t Σ_k P(k)`` (times λ if ``annual``)."""
        b = self.built
        rows = {}
        for key in b.ports:
            comp = key.split(".")[0]
            if isinstance(b.hub[comp], Custom):
                continue  # same as the subcomponent port it extends
            grid = b.grids[comp]
            total = grid.t * float(self.port(key).sum())
            vector = b.vectors[key]
            rows[key] = {
                "vector": vector,
                "unit": AMOUNT_UNITS.get(UNITS.get(vector, ""), ""),
                "amount": total * (b.lamda if annual else 1.0),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "port"
        return frame

    def timeseries(self, keys: list[str] | None = None) -> pd.DataFrame:
        """Ports on a common datetime index at the finest sampling time (held values)."""
        b = self.built
        keys = keys or list(b.ports)
        series = {}
        for key in keys:
            grid = b.grids[key.split(".")[0]]
            series[key] = pd.Series(np.asarray(self.port(key).values), index=grid.timestamps)
        step = min(b.grids[k.split(".")[0]].t for k in keys)
        index = pd.date_range(
            start=b.horizon.start,
            periods=round(24 * b.horizon.n_days / step),
            freq=pd.Timedelta(hours=step),
        )
        return pd.DataFrame({k: s.reindex(index, method="ffill") for k, s in series.items()})
