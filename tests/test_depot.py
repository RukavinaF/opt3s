"""Tests of the case study definitions that need no solver.

Run from the repository root with ``python -m unittest discover tests`` (or pytest).
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depot import inputs, scenarios  # noqa: E402
from depot.hub import build_hub  # noqa: E402
from depot.params import PARAMS, get_params  # noqa: E402
from depot import components as cmp  # noqa: E402
from depot.runner import formulate  # noqa: E402
from depot.studies import r2, r3, r4  # noqa: E402

START = datetime(2025, 1, 1)


class TestTariff(unittest.TestCase):
    def setUp(self):
        self.quarters = PARAMS["SUPP"]["tariff"]
        self.price = inputs.tariff_prices(START, 365, self.quarters, 7, 21)

    def test_hours_and_quarters(self):
        p = self.price
        self.assertEqual(len(p), 8760)
        self.assertGreater(p.min(), 0.0)  # no negative prices to exploit
        for q, month in enumerate((1, 4, 7, 10)):
            high, low = self.quarters[q]
            day = f"2025-{month:02d}-01"
            self.assertEqual(p[f"{day} 06:00"], low)  # low tariff until 7:00
            self.assertEqual(p[f"{day} 07:00"], high)
            self.assertEqual(p[f"{day} 20:00"], high)
            self.assertEqual(p[f"{day} 21:00"], low)  # and from 21:00
        self.assertEqual(p["2025-03-31 12:00"], self.quarters[0][0])  # last day of a quarter
        self.assertEqual(p["2025-12-31 12:00"], self.quarters[3][0])
        # 14 of 24 hours are high tariff in every quarter
        days = pd.Series([90, 91, 92, 92])
        expected = sum(d * (14 * h + 10 * l) for d, (h, l) in zip(days, self.quarters)) / 8760
        self.assertAlmostEqual(p.mean(), expected, places=12)

    def test_supply_prices_of_the_hub(self):
        sc = scenarios.R3b_HEP()[0]
        hub = build_hub(sc.replace(end_date=datetime(2025, 1, 7)))
        buy, sell = hub["ESRC"].ports["y1"].c_opr, hub["ESNK"].ports["u1"].c_rev
        self.assertTrue((buy == self.price[: len(buy)]).all())
        self.assertTrue(((sell - 0.9 * buy).abs() < 1e-15).all())
        # the default stays the day-ahead price, which does go negative
        da = build_hub(scenarios.r2(1, 1, 4)).__getitem__("ESRC").ports["y1"].c_opr
        self.assertLess(da.min(), 0.0)

    def test_unknown_price_source(self):
        with self.assertRaisesRegex(ValueError, "SUPP.price"):
            build_hub(scenarios.r2(1, 1, 4).replace(params={"SUPP.price": "flat"}))
        self.assertEqual(get_params()["SUPP"]["price"], "day-ahead")


class TestR3(unittest.TestCase):
    def test_brothers_differ_in_one_balance(self):
        ps = r3.pairs()
        self.assertEqual([p.part for p in ps], ["a"] * 4 + ["da", "hep"])
        for p in ps:
            a, b = p.eq.to_dict(), p.ineq.to_dict()
            differing = {k for k in a if a[k] != b[k]} - {"name", "notes"}
            if p.bus == "HTS":
                self.assertLessEqual(differing, {"hts_balance", "solver_options"})
                self.assertEqual((p.eq.hts_balance, p.ineq.hts_balance), ("eq", "ineq"))
                # only the time limit of the 0.25 h scenario may differ
                oa, ob = dict(p.eq.solver_options), dict(p.ineq.solver_options)
                oa.pop("time_limit", None), ob.pop("time_limit", None)
                self.assertEqual(oa, ob)
            else:
                self.assertEqual(differing, {"egc_equality"})
                self.assertEqual((p.eq.egc_equality, p.ineq.egc_equality), (True, False))

    def test_stored_brothers_are_the_r2_scenarios(self):
        by_name = {sc.name: sc for sc in r2.scenarios()}
        for p in r3.pairs():
            if p.stored is None:
                self.assertEqual(p.eq.params, {"SUPP.price": "tariff"})
                continue
            stored = p.eq if p.stored == "eq" else p.ineq
            self.assertEqual(stored.fingerprint, by_name[stored.name].fingerprint)
            self.assertEqual([sc.name for sc in p.new], [(p.ineq if p.stored == "eq" else p.eq).name])


class TestR4(unittest.TestCase):
    def test_brothers_differ_in_the_grid_exchange(self):
        gs = r4.groups()
        self.assertEqual([g.times for g in gs], ["1h-1h-4h", "0.5h-1h-4h", "0.25h-1h-4h"])
        by_name = {sc.name: sc for sc in r2.scenarios()}
        for g in gs:
            self.assertEqual(g.split.fingerprint, by_name[g.split.name].fingerprint)
            a = g.split.to_dict()
            for variant in ("net", "free"):
                b = g.of(variant).to_dict()
                differing = {k for k in a if a[k] != b[k]} - {"name", "notes"}
                self.assertLessEqual(differing, {"grid_exchange", "solver_options"})
                self.assertEqual(g.of(variant).grid_exchange, variant)
                oa, ob = dict(g.split.solver_options), dict(g.of(variant).solver_options)
                oa.pop("time_limit", None), ob.pop("time_limit", None)
                self.assertEqual(oa, ob)  # only the time limit of the 0.25 h scenario may differ

    def test_default_keeps_the_fingerprint(self):
        sc = {s.name: s for s in r2.scenarios()}["R2-1h-1h-4h"]
        self.assertEqual(sc.fingerprint, "fb9c776d81")  # as stored before the field existed
        self.assertNotEqual(sc.fingerprint, sc.replace(grid_exchange="net").fingerprint)

    def test_purchase_is_never_cheaper_than_sale(self):
        ctx = cmp.Context(START, 365)
        for params in ({}, {"SUPP.price": "tariff"}):
            P = get_params(params)
            buy, sell = cmp.supply_prices(P, ctx)
            fee = cmp.import_fee(P, ctx).reindex(buy.index, method="ffill")
            margin = (buy + fee) - (sell - P["EGC"]["fee_export"])
            self.assertGreater(margin.min(), 0.012)  # at least both grid fees [EUR/kWh]

    def test_hub_and_lp_of_the_signed_formulations(self):
        base = scenarios.r2(1, 1, 4).replace(end_date=datetime(2025, 1, 2))
        n = 48  # instants of the grid connection in two days
        size = {}
        for variant in r4.VARIANTS:
            sc = base.replace(grid_exchange=variant)
            hub = build_hub(sc)
            names = [c.name for c in hub.components]
            self.assertEqual("SUPP" in names, variant == "split")
            self.assertEqual("u1" in hub["EGC"].ports, variant == "split")
            m = formulate(sc).model
            size[variant] = (m.nvars, m.ncons)
        self.assertEqual(size["net"], (size["split"][0] - n, size["split"][1] + n))
        self.assertEqual(size["free"], (size["split"][0], size["split"][1] + 2 * n))
        with self.assertRaisesRegex(ValueError, "egc_equality"):
            build_hub(base.replace(grid_exchange="net", egc_equality=False))
        with self.assertRaisesRegex(ValueError, "grid_exchange"):
            base.replace(grid_exchange="single")


if __name__ == "__main__":
    unittest.main()
