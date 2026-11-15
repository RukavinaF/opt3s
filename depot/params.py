"""Every parameter of the fleet depot EH, i.e. Supplementary Tables S1–S4.

Values are grouped per component abbreviation.  Prices are per unit of size
(``c_inv`` [€/unit], ``c_mnt`` [€/(unit a)], ``c_rpl`` [€/unit]); ``n_lt`` is the
lifetime [a]; ``s_exist`` and ``s_max`` are the existing size and the upper
bound of the total size (``None``: unbounded).  Derived quantities (conversion
coefficients α, degradation prices) are computed in :mod:`depot.components`
from the physical values given here, so that nothing is entered twice.

Scenario overrides use dotted keys, e.g. ``{"HSLS.c_rev": 8.0}``, see
:func:`get_params`.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

H2_HHV = 39.41  # kWh/kg, higher heating value of hydrogen

PARAMS: dict[str, dict[str, Any]] = {
    # --- electricity supply ----------------------------------------------------
    # price "day-ahead": day-ahead price ± 8 % of its absolute value;
    # price "tariff": two-tariff purchase price, sale price a share of it.
    # The tariff is that of HEP Elektra for 2025, guaranteed supply, category
    # "poduzetništvo", medium voltage, model "Bijeli", after the first two months:
    # (high, low) [€/kWh] for each calendar quarter; the high tariff applies in
    # the hours of the high grid fee (EGC high_start to high_end).
    "SUPP": {"margin": 0.08, "price": "day-ahead",
             "tariff": ((0.153100, 0.090000), (0.178600, 0.105100),
                        (0.158800, 0.093400), (0.161300, 0.094900)),
             "tariff_sale": 0.9},
    # --- PV plant: three 1 kWp arrays (east, west, south) and one inverter -----
    "PVA": {"s_max": 1000, "c_inv": 500, "c_mnt": 8, "c_rpl": 500, "n_lt": 30},
    "PVI": {"eta": 0.85, "s_max": 4000, "c_inv": 300, "c_mnt": 3, "c_rpl": 300, "n_lt": 10},
    # --- battery energy storage system -----------------------------------------
    "BESS": {"beta_PC_BATT": 0.5},  # maximum C-rate [1/h]
    "BATT": {"c_inv": 280, "c_mnt": 7, "c_rpl": 210, "n_lt": 20,
             "n_cyc": 6500, "dod": 0.85},
    "PC": {"c_inv": 240, "c_mnt": 6, "c_rpl": 180, "n_lt": 15,
           "eta_chg": 0.93, "eta_dch": 0.93},
    # --- EV chargers ----------------------------------------------------------
    "EVC": {"p_max": 22.0,  # kW per charger
            "stations": ("LV22-13", "LV22-07", "LV22-10", "LV22-02")},
    # --- fixed consumption ----------------------------------------------------
    "FEC": {"annual_energy": 1_000_000},  # kWh
    "FHC": {"annual_energy": 1_000_000},  # kWh_th
    # --- hydrogen chain -------------------------------------------------------
    "ELYS": {"eta_el": 0.76, "eta_ht": 0.8, "water_per_h2": 10.0, "o2_per_h2": 8.0,
             "c_inv": 1860, "c_mnt": 20, "c_rpl": 1300, "n_lt": 16},
    "HCMP1": {"e_per_kg": 1.6, "c_inv": 5000, "c_mnt": 60, "c_rpl": 5000, "n_lt": 25},
    "HTNK": {"c_inv": 500, "c_mnt": 5, "c_rpl": 500, "n_lt": 20},
    "HCMP2": {"e_per_kg": 1.2, "c_inv": 5000, "c_mnt": 60, "c_rpl": 5000, "n_lt": 25},
    "HSLS": {"c_rev": 10.0,
             # (day type, start hour, end hour, maximum amount [kg])
             "windows": (("weekday", 6, 20, 2500), ("saturday", 8, 16, 1000),
                         ("sunday", 8, 14, 500))},
    "GGI": {"s_max": 100, "c_inv": 5000, "c_rev": 5.0},
    "FCS": {"eta_el": 0.5, "eta_ht": 0.85, "water_per_h2": 9.0, "water_recovered": 0.9,
            "c_inv": 1100, "c_mnt": 55, "c_rpl": 800, "n_lt": 10},
    # --- oxygen chain ---------------------------------------------------------
    "OCMP": {"e_per_kg": 0.2, "c_inv": 130, "c_mnt": 1.3, "c_rpl": 130, "n_lt": 25},
    "OTNK": {"c_inv": 70, "c_mnt": 0.7, "c_rpl": 70, "n_lt": 20},
    "OSLS": {"c_rev": 0.3, "windows": (("weekday", 8, 16, 5000),)},
    # --- water ----------------------------------------------------------------
    "DMN": {"water_per_product": 1.25, "e_per_kg": 0.0025,
            "c_inv": 4500, "c_mnt": 450, "c_rpl": 4500, "n_lt": 20},
    "WPMP": {"e_per_kg": 0.000125, "c_inv": 1875, "c_mnt": 18.75, "c_rpl": 1875, "n_lt": 20},
    "WTNK": {"c_inv": 0.2, "c_mnt": 0.0, "c_rpl": 0.2, "n_lt": 20},
    "RAIN": {"s_max": 10_000, "c_inv": 10, "c_mnt": 0.1, "c_rpl": 10, "n_lt": 30},
    "WGS": {"s_exist": 10_000, "s_max": 100_000, "c_inv": 1.3, "c_opr": 0.002},
    # --- heat -----------------------------------------------------------------
    "BLR": {"s_exist": 2000, "s_max": 2000, "c_mnt": 1, "c_rpl": 10, "n_lt": 20,
            "c_opr": 0.05},
    "HTS": {"s_exist": 2000, "c_inv": 10, "c_mnt": 1, "c_rpl": 10, "n_lt": 20},
    # --- electrical grid connection ---------------------------------------------
    "EGC": {"s_exist": 400, "s_max": 100_000, "c_inv": 200,
            "c_peak_import": 4.341, "c_peak_export": 0.142,  # €/kW per month
            "fee_high": 0.023636, "fee_low": 0.011816,  # €/kWh, import
            "high_start": 7, "high_end": 21,  # hours of the high tariff
            "fee_export": 0.00042},  # €/kWh
}  # fmt: skip


def get_params(overrides: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """A copy of :data:`PARAMS` with ``{"COMP.key": value}`` overrides applied."""
    params = copy.deepcopy(PARAMS)
    for dotted, value in (overrides or {}).items():
        comp, _, key = dotted.partition(".")
        if comp not in params or key not in params[comp]:
            raise KeyError(f"unknown parameter {dotted!r}")
        params[comp][key] = value
    return params


def flat(params: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """``{"COMP.key": value}`` view, e.g. for fingerprinting a run's data."""
    return {f"{c}.{k}": v for c, group in params.items() for k, v in group.items()}
