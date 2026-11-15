"""Port types, subsection "Ports".

A port is an input ``u_i`` or an output ``y_j`` of a component.  Its value
``P(k)`` is non-negative and is controlled in one of the following ways:

=====  ======================================  =====================================
abbr.  class                                   value
=====  ======================================  =====================================
FC     ``FC``                                  decision variable ``P >= 0``
UC     ``UC``                                  parameter (profile)
DLC    ``LinComb(terms)``                      intermediate ``Σ α_i P_i``
BLC    ``LinComb(terms, bounded=True)``        decision ``P <= Σ α_i P_i``
DP     ``Profile(zeta)``                       intermediate ``ζ(k) S_total``
BP     ``Profile(zeta, bounded=True)``         decision ``P <= ζ(k) S_total``
DA     ``Amount(windows)``                     decision, ``t Σ_{K_ω} P = E(ω)``
BA     ``Amount(windows, bounded=True)``       decision, ``t Σ_{K_ω} P <= E(ω)``
DO     ``Other("COMP.port")``                  the other port, resampled (eq. other)
BO     ``Other("COMP.port", bounded=True)``    decision ``P <=`` the other port
=====  ======================================  =====================================

Every port may in addition carry

* ``p_max``: an upper bound ``P(k) <= P_max(k)`` (scalar or profile);
* ``c_opr`` / ``c_rev``: unit operating price / revenue (scalar or profile),
  eqs. (operating_cost) and (revenue_comp);
* ``peak``: a :class:`PeakBilling`, eqs. (peak_power) and (peak_cost).

Port names are ``u1``, ``u2``, ... for inputs and ``y1``, ``y2``, ... for outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .timegrid import Window

PORT_NAME = re.compile(r"^[uy][1-9][0-9]*$")

#: Units of the vectors used in the case study; any other vector name is allowed.
UNITS = {
    "electricity": "kW",
    "heat": "kW_th",
    "hydrogen": "kg/h",
    "oxygen": "kg/h",
    "water": "kg/h",
}


def is_input(port: str) -> bool:
    return port.startswith("u")


@dataclass(frozen=True)
class PeakBilling:
    """Peak of a port billed per period, eqs. (peak_power) and (peak_cost).

    ``c_peak`` is the price per unit of peak, either one value for all periods
    or one value per billing period in the horizon.
    """

    c_peak: float | Sequence[float]
    periods: str = "month"


class Port:
    """Common part of every port type."""

    abbrev: str = "?"

    def __init__(
        self,
        vector: str | None = None,
        *,
        p_max: Any = None,
        c_opr: Any = None,
        c_rev: Any = None,
        peak: PeakBilling | None = None,
    ) -> None:
        self.vector = vector
        self.p_max = p_max
        self.c_opr = c_opr
        self.c_rev = c_rev
        self.peak = peak

    #: local port names (same component) this port is computed from
    def local_deps(self) -> list[str]:
        return []

    #: global references ``COMP.port`` this port is computed from
    def global_deps(self) -> list[str]:
        return []

    @property
    def is_decision(self) -> bool:
        """Whether the port is a decision variable (or else intermediate/parameter)."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.vector!r})"


class FC(Port):
    """Fully controllable: a decision variable bounded only from below by 0."""

    abbrev = "FC"


class UC(Port):
    """Uncontrollable: a parameter, e.g. a fixed demand."""

    abbrev = "UC"

    def __init__(self, vector: str | None, profile: Any, **kwargs: Any) -> None:
        super().__init__(vector, **kwargs)
        self.profile = profile

    @property
    def is_decision(self) -> bool:
        return False


class LinComb(Port):
    """Defined (DLC) or bounded (BLC) by a linear combination, eq. (lin_combo).

    ``terms`` maps other ports of the same component to their coefficients α.
    """

    def __init__(
        self,
        vector: str | None,
        terms: Mapping[str, float],
        bounded: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(vector, **kwargs)
        if not terms:
            raise ValueError("a linear combination needs at least one term")
        self.terms = {str(k): float(v) for k, v in terms.items()}
        self.bounded = bounded

    @property
    def abbrev(self) -> str:  # type: ignore[override]
        return "BLC" if self.bounded else "DLC"

    def local_deps(self) -> list[str]:
        return list(self.terms)

    @property
    def is_decision(self) -> bool:
        return self.bounded


class Profile(Port):
    """Defined (DP) or bounded (BP) by a profile scaled with the size, eq. (profile)."""

    def __init__(
        self, vector: str | None, zeta: Any, bounded: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(vector, **kwargs)
        self.zeta = zeta
        self.bounded = bounded

    @property
    def abbrev(self) -> str:  # type: ignore[override]
        return "BP" if self.bounded else "DP"

    @property
    def is_decision(self) -> bool:
        return self.bounded


class Amount(Port):
    """Defined (DA) or bounded (BA) by amounts over timeframes ω, eq. (amount).

    ``windows`` are :class:`~opt3s.timegrid.Window` objects or
    ``(start, end, amount)`` tuples in calendar time.  Instants outside every
    window form timeframes with ``E(ω) = 0``, so the port is zero there.
    ``rounding`` decides how a window is mapped onto the sampling time, see
    :func:`~opt3s.timegrid.window_index`.
    """

    def __init__(
        self,
        vector: str | None,
        windows: Iterable[Window | Sequence],
        bounded: bool = False,
        rounding: str = "floor",
        **kwargs: Any,
    ) -> None:
        super().__init__(vector, **kwargs)
        self.windows = list(windows)
        self.bounded = bounded
        self.rounding = rounding

    @property
    def abbrev(self) -> str:  # type: ignore[override]
        return "BA" if self.bounded else "DA"


class Other(Port):
    """Defined (DO) or bounded (BO) by a port of another component, eq. (other).

    ``ref`` is ``"COMP.port"``.  The vector is taken from the referenced port.
    """

    def __init__(self, ref: str, bounded: bool = False, **kwargs: Any) -> None:
        super().__init__(None, **kwargs)
        if ref.count(".") != 1:
            raise ValueError(f"reference {ref!r} is not of the form 'COMP.port'")
        self.ref = ref
        self.bounded = bounded

    @property
    def abbrev(self) -> str:  # type: ignore[override]
        return "BO" if self.bounded else "DO"

    def global_deps(self) -> list[str]:
        return [self.ref]

    @property
    def is_decision(self) -> bool:
        return self.bounded

    def __repr__(self) -> str:
        return f"Other({self.ref!r}, bounded={self.bounded})"
