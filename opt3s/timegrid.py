"""Time axes of components and the resampling between them.

Every component has its own sampling time ``t`` [h].  A port value ``P(k)`` is
the mean value over the interval ``[k t, (k+1) t)``, which is what makes the two
resampling rules of eq. (other) amount-preserving:

* a finer axis *holds* the value of a coarser one over its sub-intervals;
* a coarser axis takes the *mean* of the finer values inside its interval.

The same two rules are used for input profiles (prices, production and
consumption profiles), which are intensities as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

_EPS = 1e-9


@dataclass(frozen=True)
class Horizon:
    """The optimization horizon: ``n_days`` full days from ``start``."""

    start: datetime
    n_days: int

    @property
    def lamda(self) -> float:
        """Scaling of operating costs and revenues to one year, ``365 / N_days``."""
        return 365 / self.n_days

    def grid(self, t: float, dim: str) -> "TimeGrid":
        n_h = 24 * self.n_days / t
        if not math.isclose(n_h, round(n_h)):
            raise ValueError(
                f"sampling time {t} h does not divide the horizon of {self.n_days} days"
            )
        return TimeGrid(t=float(t), n_h=int(round(n_h)), dim=dim, start=self.start)


@dataclass(frozen=True)
class TimeGrid:
    """Time instants ``k = 0 .. n_h - 1`` of one component."""

    t: float
    n_h: int
    dim: str
    start: datetime

    @property
    def index(self) -> pd.Index:
        return pd.Index(range(self.n_h), name=self.dim)

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Start of every interval ``[k t, (k+1) t)``."""
        return pd.date_range(
            start=self.start, periods=self.n_h, freq=pd.Timedelta(hours=self.t)
        )

    def hours(self, when: datetime | pd.Timestamp) -> float:
        """Hours from the horizon start to ``when``."""
        return (pd.Timestamp(when) - pd.Timestamp(self.start)) / pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- #
# parameters (profiles, prices)
# --------------------------------------------------------------------------- #
def resample_profile(series: pd.Series, grid: TimeGrid) -> xr.DataArray:
    """Put an input profile on ``grid``: mean for coarser, hold for finer steps.

    ``series`` must be indexed by timestamps starting at the horizon start, with
    a constant step.  The result is aligned onto exactly ``grid.n_h`` instants;
    a plain ``resample`` would end one source step short when upsampling.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("profiles must be pandas Series with a DatetimeIndex")
    if series.index[0] != pd.Timestamp(grid.start):
        raise ValueError(
            f"profile starts at {series.index[0]}, the horizon at {grid.start}"
        )
    source_step = series.index[1] - series.index[0]
    step = pd.Timedelta(hours=grid.t)
    target = grid.timestamps
    if step >= source_step:
        series = series.resample(step).mean()
    aligned = series.reindex(series.index.union(target)).ffill().reindex(target)
    if aligned.isna().any():
        raise ValueError(f"profile {series.name!r} does not cover the horizon")
    return xr.DataArray(aligned.to_numpy(dtype=float), coords=[grid.index])


def to_parameter(value: Any, grid: TimeGrid) -> float | xr.DataArray | None:
    """Scalar, time series or array → scalar or DataArray over ``grid``."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Series):
        return resample_profile(value, grid)
    if isinstance(value, xr.DataArray):
        if value.dims != (grid.dim,):
            raise ValueError(f"parameter over {value.dims}, expected ({grid.dim},)")
        return value
    array = np.asarray(value, dtype=float)
    if array.shape != (grid.n_h,):
        raise ValueError(f"parameter of shape {array.shape}, expected ({grid.n_h},)")
    return xr.DataArray(array, coords=[grid.index])


# --------------------------------------------------------------------------- #
# ports between components, eq. (other)
# --------------------------------------------------------------------------- #
def ratio(src: TimeGrid, dst: TimeGrid) -> tuple[int, bool]:
    """``(n, dst_is_finer)`` with ``n`` the integer ratio of the two sampling times."""
    coarse, fine = (src, dst) if src.t >= dst.t else (dst, src)
    n = round(coarse.t / fine.t)
    if n < 1 or not math.isclose(n * fine.t, coarse.t):
        raise ValueError(
            f"sampling times of {src.dim} ({src.t} h) and {dst.dim} ({dst.t} h) "
            "are not integer multiples of one another"
        )
    return n, dst.t < src.t


def resample_port(value: Any, src: TimeGrid, dst: TimeGrid) -> Any:
    """A port of a component on ``src`` expressed on ``dst``, eq. (other).

    ``value`` is a linopy variable or expression, a DataArray parameter, or a
    number.  Holding over finer steps and averaging over coarser ones both
    preserve the amount ``t Σ P(k)`` exchanged between the two components.
    """
    if isinstance(value, (int, float)) or src.dim == dst.dim:
        return value
    n, dst_finer = ratio(src, dst)
    if n == 1:
        return value.rename({src.dim: dst.dim}).assign_coords({dst.dim: dst.index})
    if dst_finer:  # hold
        held = xr.DataArray(np.arange(dst.n_h) // n, coords=[dst.index])
        return value.isel({src.dim: held})
    groups = xr.DataArray(np.arange(src.n_h) // n, coords=[src.index], name=dst.dim)
    if isinstance(value, xr.DataArray):
        mean = value.groupby(groups).mean()
    else:
        mean = (1 * value).groupby(groups).sum() / n
    return mean.assign_coords({dst.dim: dst.index})


# --------------------------------------------------------------------------- #
# timeframes ω: amounts and peak billing periods
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """Timeframe ω with its amount ``E(ω)``, given in calendar time."""

    start: datetime
    end: datetime
    amount: float


def window_index(
    windows: Iterable[Window | Sequence],
    grid: TimeGrid,
    rounding: str = "floor",
) -> tuple[np.ndarray, np.ndarray]:
    """Map windows onto the instants of ``grid``.

    Returns ``omega_of_k`` (window number per instant, ``-1`` outside every
    window) and ``E`` (amount per window).  The set ``K_ω`` is

    * ``rounding="floor"``: ``floor(start / t) <= k < floor(end / t)``, i.e. a
      window that does not fit the sampling time is rounded down, as for the
      daily sales windows;
    * ``rounding="overlap"``: ``floor(start / t) <= k < ceil(end / t)``, i.e.
      every instant the window overlaps, as for charging sessions.

    Windows that end up sharing an instant are merged into one ω with the sum
    of their amounts, so that no amount is lost or delivered twice.  Windows
    that cover no instant are dropped.
    """
    if rounding not in ("floor", "overlap"):
        raise ValueError(f"unknown rounding {rounding!r}")
    spans = []
    for w in windows:
        w = w if isinstance(w, Window) else Window(*w)
        h0, h1 = grid.hours(w.start), grid.hours(w.end)
        k0 = math.floor(h0 / grid.t + _EPS)
        k1 = (
            math.floor(h1 / grid.t + _EPS)
            if rounding == "floor"
            else math.ceil(h1 / grid.t - _EPS)
        )
        if k1 <= k0:
            continue
        if k0 < 0 or k1 > grid.n_h:
            raise ValueError(f"window {w.start} – {w.end} exceeds the horizon")
        spans.append((k0, k1, float(w.amount)))
    spans.sort(key=lambda s: s[0])

    merged: list[list] = []
    for k0, k1, amount in spans:
        if merged and k0 < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], k1)
            merged[-1][2] += amount
        else:
            merged.append([k0, k1, amount])

    omega_of_k = np.full(grid.n_h, -1, dtype=int)
    for omega, (k0, k1, _) in enumerate(merged):
        omega_of_k[k0:k1] = omega
    amounts = np.array([m[2] for m in merged], dtype=float)
    return omega_of_k, amounts


def billing_periods(grid: TimeGrid, periods: str = "month") -> np.ndarray:
    """Billing period number of every instant (calendar months by default)."""
    if periods != "month":
        raise ValueError(f"unknown billing period {periods!r}")
    stamps = grid.timestamps
    key = stamps.year * 12 + stamps.month
    _, omega = np.unique(np.asarray(key), return_inverse=True)
    return omega.astype(int)


PERIODS_PER_YEAR = {"month": 12}
