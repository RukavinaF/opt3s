"""States, sizes, cost terms and size ratios of components.

Subsections "States", "Sizing", "Economic modelling", and eq. (size_ratio1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DEGRADATION = ("none", "cal", "use", "cmb")


@dataclass
class State:
    """A state ``x_s`` of a storage, eq. (state1).

    ``alpha`` maps ports to their coefficients; inputs add to the state and
    outputs subtract from it, so all coefficients are given as positive numbers.
    ``sigma`` is the efficiency of the state (``1 - sigma`` is the self-discharge
    per instant), ``gamma`` the lower limit relative to the size, eq. (dod).
    """

    alpha: Mapping[str, float]
    sigma: Any = 1.0
    gamma: float = 0.0


@dataclass
class Size:
    """How the size ``S_total = S_exist + S_new`` of a component is defined.

    ``rule`` is one of the cases of eq. (sizing1):

    * ``"ports"``: ``S >= P_p(k)`` for every listed port;
    * ``"port_sum"``: ``S >= Σ P_p(k)``;
    * ``"states"``: ``S >= x_s(k)`` for every listed state;
    * ``"state_sum"``: ``S >= Σ x_s(k)``;
    * ``"profile"``: no constraint, the size scales a DP/BP port, eq. (profile).

    ``s_max`` bounds the total size from above, ``S_total <= S_max``.
    """

    rule: str
    items: Sequence[str] = ()
    s_exist: float = 0.0
    s_max: float | None = None

    RULES = ("ports", "port_sum", "states", "state_sum", "profile")

    def __post_init__(self) -> None:
        if self.rule not in self.RULES:
            raise ValueError(f"unknown size rule {self.rule!r}, use one of {self.RULES}")
        if self.rule != "profile" and not self.items:
            raise ValueError(f"size rule {self.rule!r} needs ports or states")
        if self.s_max is not None and self.s_max < self.s_exist:
            raise ValueError(f"S_max = {self.s_max} is below S_exist = {self.s_exist}")
        self.items = tuple(self.items)

    # convenience constructors, named after the cases of eq. (sizing1)
    @classmethod
    def max_of(cls, *ports: str, **kw: Any) -> "Size":
        return cls("ports", ports, **kw)

    @classmethod
    def max_of_sum(cls, *ports: str, **kw: Any) -> "Size":
        return cls("port_sum", ports, **kw)

    @classmethod
    def max_of_state(cls, *states: str, **kw: Any) -> "Size":
        return cls("states", states, **kw)

    @classmethod
    def max_of_state_sum(cls, *states: str, **kw: Any) -> "Size":
        return cls("state_sum", states, **kw)

    @classmethod
    def by_profile(cls, **kw: Any) -> "Size":
        return cls("profile", (), **kw)

    @property
    def s_new_max(self) -> float:
        """Upper bound of the enlargement ``S_new``."""
        return math.inf if self.s_max is None else self.s_max - self.s_exist


@dataclass
class Costs:
    """Unit prices of the size-dependent cost terms of a component.

    ``degradation`` is ``"none"``, ``"cal"`` (calendar, eq. C_deg_cal), ``"use"``
    (with usage, eq. C_deg_use) or ``"cmb"`` (combined, eq. C_deg_total).
    Usage degradation needs ``c_deg_use`` and the degrading ports ``deg_ports``.
    Operating prices, revenues and peak prices belong to ports.
    """

    c_inv: float = 0.0
    c_mnt: float = 0.0
    c_rpl: float = 0.0
    n_lt: float | None = None
    degradation: str = "none"
    c_deg_use: float | None = None
    deg_ports: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.degradation not in DEGRADATION:
            raise ValueError(f"unknown degradation {self.degradation!r}")
        if self.degradation in ("cal", "cmb") and self.n_lt is None:
            raise ValueError("calendar degradation needs the lifetime n_lt")
        if self.degradation in ("use", "cmb") and (
            self.c_deg_use is None or not self.deg_ports
        ):
            raise ValueError("usage degradation needs c_deg_use and deg_ports")
        self.deg_ports = tuple(self.deg_ports)

    def n_rpl(self, n_dl: float) -> int:
        """Number of replacements over the design lifetime, eq. (N_rpl)."""
        if self.n_lt is None:
            return 0
        return max(math.ceil(n_dl / self.n_lt) - 1, 0)

    def c_deg_cal(self, n_dl: float) -> float:
        """Annualized price of calendar degradation per unit of size, eq. (C_deg_cal)."""
        return self.n_rpl(n_dl) / n_dl * self.c_rpl


@dataclass(frozen=True)
class Ratio:
    """Size ratio between two subcomponents, eq. (size_ratio1):
    ``S_total[c1] <relation> beta * S_total[c2]``."""

    c1: str
    relation: str
    beta: float
    c2: str

    def __post_init__(self) -> None:
        if self.relation not in ("<=", "==", ">="):
            raise ValueError(f"unknown relation {self.relation!r}")


def annuity(r: float, n_dl: float) -> float:
    """Annuity factor ``(1 - (1 + r)^-N_DL) / r`` of eq. (cost_function)."""
    return n_dl if r == 0 else (1 - (1 + r) ** (-n_dl)) / r
