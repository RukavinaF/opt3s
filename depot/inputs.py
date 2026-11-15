"""Input data of the case study, as calendar-indexed series and windows.

Profiles are labelled from the horizon start, as in Supplementary Table S2;
:mod:`opt3s` resamples them to each component's sampling time.  Data files are
located relative to this package, so the working directory does not matter.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from opt3s import Window

DATA = Path(__file__).resolve().parent.parent / "data"

PV_FILES = (  # PVA1 east, PVA2 west, PVA3 south
    "PVGIS_45.801_15.971_SA3_1kWp_crystSi_0_10deg_-90deg_2023_2023.csv",
    "PVGIS_45.801_15.971_SA3_1kWp_crystSi_0_10deg_90deg_2023_2023.csv",
    "PVGIS_45.801_15.971_SA3_1kWp_crystSi_0_10deg_0deg_2023_2023.csv",
)
DAY_TYPES = {"weekday": range(0, 5), "saturday": (5,), "sunday": (6,)}


@lru_cache(maxsize=None)
def _column(filename: str, **kwargs) -> np.ndarray:
    return pd.read_csv(DATA / filename, header=None, **kwargs).iloc[:, 0].to_numpy(float)


def _yearly_series(values: np.ndarray, start: datetime, name: str) -> pd.Series:
    """One year of samples labelled from ``start`` with an even step."""
    step = pd.Timedelta(hours=8760 / len(values))
    index = pd.date_range(start=start, periods=len(values), freq=step)
    return pd.Series(values, index=index, name=name)


def prices(start: datetime, margin: float) -> tuple[pd.Series, pd.Series]:
    """Purchase and sale price: day-ahead price ± ``margin`` of its absolute value."""
    da = _yearly_series(_column("HR_DA_PRICES_2025.csv"), start, "price")
    return (da + margin * da.abs()).rename("c_opr"), (da - margin * da.abs()).rename("c_rev")


def tariff_prices(start: datetime, n_days: int, quarters, high_start: int,
                  high_end: int) -> pd.Series:  # fmt: skip
    """Hourly two-tariff purchase price.

    ``quarters[q] = (high, low)`` are the prices of calendar quarter ``q``; the
    high tariff applies from ``high_start`` to ``high_end`` every day.
    """
    table = np.asarray(quarters, dtype=float)
    if table.shape != (4, 2):
        raise ValueError(f"tariff needs (high, low) for four quarters, got shape {table.shape}")
    index = pd.date_range(start=start, periods=24 * n_days, freq="1h")
    quarter = (index.month - 1) // 3
    high = (index.hour >= high_start) & (index.hour < high_end)
    values = np.where(high, table[quarter, 0], table[quarter, 1])
    return pd.Series(values, index=index, name="c_opr")


def pv_profile(array: int, start: datetime) -> pd.Series:
    """Production of a 1 kWp array, ``array`` = 0 (east), 1 (west), 2 (south)."""
    return _yearly_series(_column(PV_FILES[array]), start, f"PVA{array + 1}")


def rainfall(start: datetime) -> pd.Series:
    """Rainfall in mm/h, i.e. kg/(m² h)."""
    return _yearly_series(_column("rainfall-45.80N15.92E131m-2025.csv"), start, "rain")


def building_profile(filename: str, start: datetime, annual: float) -> pd.Series:
    """DOE reference building profile scaled to ``annual``, weekdays on the calendar.

    The profiles start on a Sunday; whole days are rolled so that the weekdays
    of the profile match those of the horizon.
    """
    values = _column(filename)
    shift = (start.weekday() - 6) % 7
    values = np.roll(values, -24 * shift)
    return _yearly_series(values * annual / values.sum(), start, filename)


def grid_fee(start: datetime, n_days: int, high: float, low: float,
             high_start: int, high_end: int) -> pd.Series:  # fmt: skip
    """Hourly import grid fee: ``high`` from ``high_start`` to ``high_end``, else ``low``."""
    day = [low] * high_start + [high] * (high_end - high_start) + [low] * (24 - high_end)
    index = pd.date_range(start=start, periods=24 * n_days, freq="1h")
    return pd.Series(np.tile(day, n_days), index=index, name="fee")


@lru_cache(maxsize=None)
def _charging_sessions() -> pd.DataFrame:
    return pd.read_csv(
        DATA / "NREL_Charging_Data.csv",
        sep=";",
        parse_dates=["start_date_time", "end_date_time"],
        date_format="%d.%m.%Y %H:%M",
    )


def ev_windows(station: str, start: datetime, n_days: int) -> list[Window]:
    """Charging sessions of one station as windows with their charged energy [kWh].

    The 2020 sessions are moved onto the horizon by their offset from
    1 January 2020; sessions not fully inside the horizon are left out.
    """
    df = _charging_sessions()
    df = df[(df.station == station) & (df.end_date_time < pd.Timestamp("2020-12-31"))]
    df = df.sort_values("start_date_time")
    t0 = pd.Timestamp("2020-01-01")
    begin, end = pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=n_days)
    windows = []
    for s, e, energy in zip(df.start_date_time, df.end_date_time, df.energy_charged):
        ws, we = begin + (s - t0), begin + (e - t0)
        if ws >= begin and we <= end:
            windows.append(Window(ws.to_pydatetime(), we.to_pydatetime(), float(energy)))
    return windows


def daily_windows(spec, start: datetime, n_days: int) -> list[Window]:
    """Daily sales windows ``(day type, start hour, end hour, amount)``."""
    windows = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        for day_type, h0, h1, amount in spec:
            if day.weekday() in DAY_TYPES[day_type]:
                windows.append(
                    Window(day + timedelta(hours=h0), day + timedelta(hours=h1), amount)
                )
    return windows
