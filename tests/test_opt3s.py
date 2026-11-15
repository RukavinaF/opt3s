"""Unit tests of opt3s on small hubs with hand-computable results.

Run from the repository root with ``python -m unittest discover tests`` (or pytest).
Solved with HiGHS, so no license is needed.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opt3s import (  # noqa: E402
    FC, UC, Amount, Costs, Custom, Horizon, Hub, HubResult, LinComb, Other, PeakBilling,
    Profile, Size, Standard, State, Window, annuity, build, resample_port, solve,
)  # fmt: skip
from opt3s.timegrid import window_index  # noqa: E402

START = datetime(2025, 1, 1)
E = "electricity"


def hourly(values, days=1) -> pd.Series:
    idx = pd.date_range(START, periods=24 * days, freq="1h")
    return pd.Series(np.resize(np.asarray(values, float), 24 * days), index=idx)


def run(hub: Hub, days: int = 1, **kw) -> HubResult:
    built = build(hub, Horizon(START, days), **kw)
    solve(built, solver="highs", progress=False)
    return HubResult(built)


def grid_load_hub(t_grid=1.0, t_load=1.0, t_bus=1.0, price=0.2, load=5.0, balance="eq"):
    grid = Standard("GRID", "source", {"y1": FC(E, c_opr=price)}, t=t_grid)
    load_ = Standard("LOAD", "sink", {"u1": UC(E, hourly(load))}, t=t_load)
    third = Standard("SPILL", "sink", {"u1": FC(E)}, t=t_bus)
    bus = Standard(
        "BUS", "bus",
        {"u1": Other("GRID.y1"), "y1": Other("LOAD.u1"), "y2": Other("SPILL.u1")},
        balance=balance, t=t_bus,
    )  # fmt: skip
    return Hub("h", [grid, load_, third, bus])


class TestTime(unittest.TestCase):
    def test_hold_and_mean_preserve_amounts(self):
        h = Horizon(START, 1)
        coarse, fine = h.grid(2.0, "k_c"), h.grid(0.5, "k_f")
        x = xr.DataArray(np.arange(12.0), coords=[coarse.index])
        held = resample_port(x, coarse, fine)
        self.assertEqual(held.sizes["k_f"], 48)
        self.assertAlmostEqual(0.5 * float(held.sum()), 2.0 * float(x.sum()))
        back = resample_port(held, fine, coarse)
        np.testing.assert_allclose(back.values, x.values)

    def test_non_integer_ratio(self):
        h = Horizon(START, 1)
        with self.assertRaises(ValueError):
            resample_port(xr.DataArray(np.zeros(12), coords=[h.grid(2, "a").index]),
                          h.grid(2, "a"), h.grid(3, "b"))  # fmt: skip

    def test_windows_floor_overlap_merge(self):
        g = Horizon(START, 1).grid(4.0, "k")
        w = [Window(datetime(2025, 1, 1, 6), datetime(2025, 1, 1, 21), 10)]
        omega, E_ = window_index(w, g, "floor")  # 6:00-21:00 → 4:00-20:00 at 4 h
        self.assertEqual(list(np.flatnonzero(omega >= 0)), [1, 2, 3, 4])
        omega, _ = window_index(w, g, "overlap")  # → 4:00-24:00
        self.assertEqual(list(np.flatnonzero(omega >= 0)), [1, 2, 3, 4, 5])
        two = w + [Window(datetime(2025, 1, 1, 7), datetime(2025, 1, 1, 9), 5)]
        omega, E_ = window_index(two, g, "overlap")
        self.assertEqual(list(E_), [15.0])  # overlapping windows merge


class TestPorts(unittest.TestCase):
    def test_fc_uc_bus_and_operating_cost(self):
        res = run(grid_load_hub())
        self.assertAlmostEqual(res.economics()["OPEX"], 365 * 24 * 5 * 0.2, places=4)

    def test_mixed_sampling_amounts(self):
        # grid at 2 h, bus at 1 h, load at 0.5 h with a varying profile
        load = np.arange(24.0)
        grid = Standard("GRID", "source", {"y1": FC(E, c_opr=0.1)}, t=2)
        load_ = Standard("LOAD", "sink", {"u1": UC(E, hourly(load))}, t=0.5)
        spill = Standard("SPILL", "sink", {"u1": FC(E)}, t=1)
        bus = Standard("BUS", "bus", {"u1": Other("GRID.y1"), "y1": Other("LOAD.u1"),
                                      "y2": Other("SPILL.u1")}, t=1)  # fmt: skip
        res = run(Hub("h", [grid, load_, spill, bus]))
        a = res.amounts(annual=False)["amount"]
        # eq. (other) preserves the amounts in both directions ...
        self.assertAlmostEqual(a["BUS.u1"], a["GRID.y1"], places=6)  # 2 h → 1 h, hold
        self.assertAlmostEqual(a["BUS.y1"], load.sum(), places=6)  # 0.5 h → 1 h, mean
        # ... and the 2 h grid covers the larger hour of each pair, the rest spills
        self.assertAlmostEqual(a["GRID.y1"], 2 * load[1::2].sum(), places=6)

    def test_bus_eq_vs_ineq(self):
        # a free must-take supply above the load: infeasible with eq, fine with ineq
        def hub(balance):
            src = Standard("SRC", "source", {"y1": UC(E, hourly(10.0))}, t=1)
            load = Standard("LOAD", "sink", {"u1": UC(E, hourly(4.0))}, t=1)
            other = Standard("O", "sink", {"u1": FC(E, c_opr=1.0)}, t=1)
            bus = Standard("BUS", "bus", {"u1": Other("SRC.y1"), "y1": Other("LOAD.u1"),
                                          "y2": Other("O.u1")}, balance=balance, t=1)  # fmt: skip
            return Hub("h", [src, load, other, bus])
        built = build(hub("ineq"), Horizon(START, 1))
        solve(built, solver="highs", progress=False)
        self.assertAlmostEqual(HubResult(built).economics()["OPEX"], 0.0, places=6)
        built = build(hub("eq"), Horizon(START, 1))
        solve(built, solver="highs", progress=False)  # O absorbs the surplus at a cost
        self.assertAlmostEqual(HubResult(built).economics()["OPEX"], 365 * 24 * 6.0, places=3)

    def test_defined_by_amount(self):
        windows = [Window(datetime(2025, 1, 1, 8), datetime(2025, 1, 1, 12), 20.0)]
        price = hourly([0.3] * 10 + [0.1] + [0.3] * 13)  # cheapest at 10:00
        grid = Standard("GRID", "source", {"y1": FC(E, c_opr=price)}, t=1)
        ev = Standard("EV", "sink", {"u1": Amount(E, windows, p_max=15.0)}, t=1)
        spill = Standard("SPILL", "sink", {"u1": FC(E)}, t=1)
        bus = Standard("BUS", "bus", {"u1": Other("GRID.y1"), "y1": Other("EV.u1"),
                                      "y2": Other("SPILL.u1")}, t=1)  # fmt: skip
        res = run(Hub("h", [grid, ev, spill, bus]))
        u = res.port("EV.u1").values
        self.assertAlmostEqual(u.sum(), 20.0, places=6)
        self.assertAlmostEqual(u[10], 15.0, places=6)  # P_max at the cheapest hour
        self.assertAlmostEqual(u[:8].sum() + u[12:].sum(), 0.0, places=6)  # E(ω) = 0 gaps

    def test_negative_lincomb_stays_nonnegative(self):
        # water bus pattern: u1 = y1 - u2 must not become negative, otherwise the
        # paid grid supply WGS would turn into a revenue and the LP be unbounded
        def hub(free):
            src = Standard("FREE", "source", {"y1": free}, t=1)
            need = Standard("NEED", "sink", {"u1": UC("water", hourly(5.0))}, t=1)
            bus = Standard("WB", "bus", {"u2": Other("FREE.y1"), "y1": Other("NEED.u1"),
                                         "u1": LinComb("water", {"y1": 1, "u2": -1})}, t=1)  # fmt: skip
            grid = Standard("WGS", "source", {"y1": Other("WB.u1", c_opr=1.0)}, t=1)
            return Hub("h", [src, need, bus, grid])
        res = run(hub(FC("water", p_max=8.0)))
        self.assertEqual(str(res.built.model.status), "ok")
        self.assertAlmostEqual(res.economics()["OPEX"], 0.0, places=6)
        with self.assertRaisesRegex(ValueError, "negative"):  # 5 - 8 as a parameter
            build(hub(UC("water", hourly(8.0))), Horizon(START, 1))


class TestStatesSizesCosts(unittest.TestCase):
    def test_storage_shifts_energy_and_repeats(self):
        price = hourly([0.1] * 12 + [0.5] * 12)
        grid = Standard("GRID", "source", {"y1": FC(E, c_opr=price)}, t=1)
        load = Standard("LOAD", "sink", {"u1": UC(E, hourly(10.0))}, t=1)
        stor = Standard(
            "ST", "storage", {"u1": FC(E), "y1": FC(E)},
            states={"x1": State({"u1": 1.0, "y1": 1.0}, gamma=0.2)},
            size=Size.max_of_state("x1"), costs=Costs(c_inv=1.0),
        )  # fmt: skip
        stc = Custom("STC", [stor], ports={"u1": "ST.u1", "y1": "ST.y1"}, t=1)
        bus = Standard("BUS", "bus", {"u1": Other("GRID.y1"), "u2": Other("STC.y1"),
                                      "y1": Other("LOAD.u1"), "y2": Other("STC.u1")}, t=1)  # fmt: skip
        res = run(Hub("h", [grid, load, stc, bus]))
        x = res.built.states["ST.x1"].solution.values
        x0 = float(res.built.model.variables["ST_x1_0"].solution)
        self.assertAlmostEqual(x0, x[-1], places=6)  # repeatability
        S = res.sizes().loc["ST", "S_total"]
        self.assertAlmostEqual(S, 120.0 / 0.8, places=4)  # 12 h x 10 kW above γ S
        self.assertGreaterEqual(x.min(), 0.2 * S - 1e-6)
        # the storage is attributed to the custom component as well
        self.assertAlmostEqual(res.costs().loc["STC", "C_inv"], S, places=4)

    def test_profile_size_bounded_by_s_max(self):
        pv = Standard("PV", "source", {"y1": Profile(E, hourly(np.r_[np.zeros(6), np.ones(12),
                                                                    np.zeros(6)]))},
                      size=Size.by_profile(s_exist=1.0, s_max=3.0), costs=Costs(c_inv=0.01), t=1)  # fmt: skip
        sale = Standard("SALE", "sink", {"u1": FC(E, c_rev=0.3)}, t=1)
        x = Standard("X", "sink", {"u1": FC(E)}, t=1)
        bus = Standard("BUS", "bus", {"u1": Other("PV.y1"), "y1": Other("SALE.u1"),
                                      "y2": Other("X.u1")}, t=1)  # fmt: skip
        res = run(Hub("h", [pv, sale, x, bus]))
        self.assertAlmostEqual(res.sizes().loc["PV", "S_total"], 3.0, places=6)
        self.assertAlmostEqual(res.sizes().loc["PV", "S_new"], 2.0, places=6)

    def test_degradation_prices(self):
        k = Costs(c_rpl=100, n_lt=8, degradation="cal")
        self.assertEqual(k.n_rpl(20), 2)  # ceil(20 / 8) - 1
        self.assertAlmostEqual(k.c_deg_cal(20), 10.0)
        self.assertEqual(Costs(c_rpl=1, n_lt=25, degradation="cal").n_rpl(20), 0)
        self.assertAlmostEqual(annuity(0.03, 20), 14.877474860455516)

    def test_peak_scaling_per_billing_period(self):
        # one week: one billing period touched, λ_ω = 12
        hub = grid_load_hub()
        hub["GRID"].ports["y1"].peak = PeakBilling(2.0)
        res = run(hub, days=7)
        self.assertAlmostEqual(res.costs().loc["GRID", "C_peak"], 12 * 2.0 * 5.0, places=4)


class TestValidation(unittest.TestCase):
    def test_type_rules(self):
        with self.assertRaises(ValueError):
            Standard("S", "source", {"u1": FC(E)})
        with self.assertRaises(ValueError):
            Standard("S", "storage", {"u1": FC(E), "y1": FC(E)})
        with self.assertRaises(ValueError):
            Standard("B", "bus", {"u1": FC(E), "y1": FC(E)})

    def test_cycle(self):
        a = Standard("A", "process", {"u1": Other("B.y1"), "y1": LinComb(E, {"u1": 1})}, t=1)
        b = Standard("B", "process", {"u1": Other("A.y1"), "y1": LinComb(E, {"u1": 1})}, t=1)
        with self.assertRaisesRegex(ValueError, "circular"):
            build(Hub("h", [a, b]), Horizon(START, 1))

    def test_non_integer_sampling_ratio(self):
        with self.assertRaisesRegex(ValueError, "integer multiples"):
            build(grid_load_hub(t_grid=2, t_bus=3, t_load=3), Horizon(START, 1))

    def test_bus_vectors(self):
        h2 = Standard("H", "source", {"y1": FC("hydrogen")}, t=1)
        bad = Standard("BUS2", "bus", {"u1": Other("H.y1"), "y1": FC(E), "y2": FC(E)}, t=1)
        with self.assertRaisesRegex(ValueError, "one vector"):
            build(Hub("h", [h2, bad]), Horizon(START, 1))


if __name__ == "__main__":
    unittest.main()
