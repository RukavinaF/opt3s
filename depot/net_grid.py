"""Electricity exchange with the grid as one signed net power (comparison only).

The hub of the article exchanges electricity with the grid through two
non-negative ports, import ``ESRC.y1`` and export ``ESNK.u1`` ("split").  Previous
work (Rukavina et al. 2025, eqs. 37, 51a, 54; Rukavina et al. 2026, eqs. 16-18)
uses one signed quantity instead.  This module adds that formulation to a hub
built without the electricity supply and without the grid ports of the EGC:

* net grid power, positive for import and negative for export::

      P_grid(k) = Σ_{j>=2} y_j(k) − Σ_{i>=2} u_i(k)

  ``"net"``: an intermediate variable (substituted wherever it appears);
  ``"free"``: a decision variable without sign restriction, defined by this
  equality;
* energy cost, a decision variable without sign restriction::

      C_grid(k) >= t c_buy(k) P_grid(k),     C_grid(k) >= t c_sell(k) P_grid(k)

  with ``c_buy`` = purchase price + import grid fee and ``c_sell`` = sale price
  − export grid fee; it describes the cost only if ``c_buy(k) >= c_sell(k)``;
* billed peaks ``P_imp,peak(ω) >= P_grid(k)`` and ``P_exp,peak(ω) >= −P_grid(k)``;
* size ``S_total >= P_grid(k)`` and ``S_total >= −P_grid(k)``.

It breaks the rule that every port is non-negative, so it is not a port type
of :mod:`opt3s`; scenarios select it with ``grid_exchange="net"`` or ``"free"``.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from opt3s.backend import lsum, value_of
from opt3s.build import BuiltHub
from opt3s.timegrid import PERIODS_PER_YEAR, billing_periods, to_parameter
from scenario import Scenario

from . import components as cmp
from .params import get_params

EGC = "EGC"


def _prices(b: BuiltHub, sc: Scenario) -> dict[str, Any]:
    """Purchase and sale price and the grid fees over the instants of the EGC [€/kWh]."""
    P = get_params(sc.params)
    ctx = cmp.Context(sc.start_date, sc.N_days)
    grid = b.grids[EGC]
    purchase, sale = cmp.supply_prices(P, ctx)
    return {
        "purchase": to_parameter(purchase, grid),
        "sale": to_parameter(sale, grid),
        "fee_import": to_parameter(cmp.import_fee(P, ctx), grid),
        "fee_export": float(P[EGC]["fee_export"]),
        "c_peak_import": float(P[EGC]["c_peak_import"]),
        "c_peak_export": float(P[EGC]["c_peak_export"]),
    }


def extension(sc: Scenario) -> Callable[[BuiltHub], None]:
    """The ``extend`` function of :func:`opt3s.build` for scenario ``sc``."""
    if sc.grid_exchange not in ("net", "free"):
        raise ValueError(f"no extension for grid_exchange={sc.grid_exchange!r}")
    if not math.isclose(sc.t_of("SUPP"), sc.t_of(EGC)):
        raise ValueError("SUPP and EGC need the same sampling time")

    def extend(b: BuiltHub) -> None:
        be, grid, egc = b.backend, b.grids[EGC], b.hub[EGC]
        c = _prices(b, sc)
        c_buy = c["purchase"] + c["fee_import"]
        c_sell = c["sale"] - c["fee_export"]
        if float((c_buy - c_sell).min()) < 0:
            raise ValueError("c_buy < c_sell at some instant: the cost is not convex")

        loads = lsum(*(b.ports[f"{EGC}.{p}"] for p in egc.outputs))
        sources = lsum(*(b.ports[f"{EGC}.{p}"] for p in egc.inputs))
        net = lsum(loads, -1 * sources)  # P_grid, positive for import
        if sc.grid_exchange == "free":
            p_grid = be.decision(f"{EGC}_net", coords=[grid.index], lower=-math.inf)
            be.constrain(p_grid - net == 0, f"{EGC}_balance")
            net = 1 * p_grid

        cost = be.decision(f"{EGC}_net_cost", coords=[grid.index], lower=-math.inf)
        be.constrain(cost - net * (grid.t * c_buy) >= 0, f"{EGC}_net_cost_buy")
        be.constrain(cost - net * (grid.t * c_sell) >= 0, f"{EGC}_net_cost_sell")

        omega = billing_periods(grid, "month")
        n = int(omega.max()) + 1
        dim = f"omega_{EGC}"
        indexer = xr.DataArray(omega, coords=[grid.index])
        lamda_omega = PERIODS_PER_YEAR["month"] / n
        peak_cost = []
        for name, sign, price in (("import", 1, c["c_peak_import"]), ("export", -1, c["c_peak_export"])):
            peak = be.decision(f"{EGC}_{name}_peak", coords=[(dim, np.arange(n))])
            be.constrain(sign * net - peak.isel({dim: indexer}) <= 0, f"{EGC}_{name}_peak")
            b.peaks[f"{EGC}.{name}"] = peak
            peak_cost.append(lamda_omega * (price * peak).sum())

        s_total = b.sizes[EGC]["S_total"]
        be.constrain(s_total - net >= 0, f"{EGC}_S_total_import")
        be.constrain(s_total + net >= 0, f"{EGC}_S_total_export")

        # the net cost holds purchase, sale and both grid fees; it enters the OPEX
        b.costs[EGC]["C_opr"] = lsum(b.costs[EGC]["C_opr"], b.lamda * cost.sum())
        b.costs[EGC]["C_peak"] = lsum(b.costs[EGC]["C_peak"], *peak_cost)
        b.net_grid = {"net": net, "cost": cost}  # read by split_results

    return extend


def split_results(b: BuiltHub, sc: Scenario, objective: dict[str, Any], costs: pd.DataFrame,
                  amounts: pd.DataFrame) -> pd.DataFrame:  # fmt: skip
    """Import, export, OPEX and revenue of a solved hub, reported as for ``"split"``.

    The net power is split into its positive part (import) and negative part
    (export), and the net cost into purchase cost, grid fees and sale revenue.
    ``objective`` and ``costs`` are changed in place: the revenue of the export
    is added to both OPEX and REV, so the annual profit and the NPV keep the
    values of the solved LP.  Returns ``amounts`` with the four grid ports.
    """
    grid = b.grids[EGC]
    c = _prices(b, sc)
    net = np.asarray(value_of(b.net_grid["net"]).values, float)
    imp, exp = np.maximum(net, 0.0), np.maximum(-net, 0.0)
    year = b.lamda * grid.t  # € per year from €/kWh times kW

    def total(price: Any, power: np.ndarray) -> float:
        return year * float((np.asarray(price) * power).sum())

    purchase = total(c["purchase"], imp)
    sale = total(c["sale"], exp)
    fees = total(c["fee_import"], imp) + total(c["fee_export"], exp)
    net_cost = b.lamda * float(value_of(b.net_grid["cost"]).sum())
    objective["net_cost_gap"] = net_cost - (purchase + fees - sale)  # 0 if the cost is tight

    costs.loc["ESRC"] = 0.0
    costs.loc["ESNK"] = 0.0
    costs.loc["ESRC", "C_opr"] = purchase
    costs.loc["ESNK", "C_rev"] = sale
    costs.loc["SUPP"] = costs.loc["ESRC"] + costs.loc["ESNK"]
    costs.loc[EGC, "C_opr"] = fees
    top = [comp.name for comp in b.hub.components] + ["SUPP"]
    costs.loc["TOTAL"] = costs.loc[top].sum()
    objective["OPEX"] += sale
    objective["REV"] += sale

    for key, power in (("ESRC.y1", imp), ("EGC.u1", imp), ("ESNK.u1", exp), ("EGC.y1", exp)):
        amounts.loc[key] = {"vector": "electricity", "unit": "kWh",
                            "amount": year * float(power.sum())}  # fmt: skip
    return amounts
