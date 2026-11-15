"""From a declared hub to an LP: Algorithms 1 and 2 of subsection Implementation.

Ordering is the only real difficulty, because a port defined or bounded by a
linear combination or by another component references ports that must already
exist.  Algorithms 1 and 2 state this as "such ports last" within a component
and "components without external dependencies first" within a custom
component.  Here the same rule is applied to the dependency graph of all ports
of the hub at once, i.e. every port is defined after the ports it references
(a topological order), which also resolves chains such as
``ELYS.y1 → HCMP1.u1 → HCMP1.y1 → HTNK.u1``.  A cycle is an error.

The build then proceeds as in Algorithm 1 for every standard component (ports,
peaks, states, size, cost terms) and as in Algorithm 2 for every custom
component (external ports, size ratios, summed cost terms), and closes with the
bus balances, ``CAPEX``, ``OPEX``, ``REV`` and the objective, eq. (cost_function).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import xarray as xr

from .backend import Backend, is_linear, lsum
from .components import Component, Custom, Hub, Standard
from .ports import FC, UC, Amount, LinComb, Other, Profile, is_input
from .specs import annuity
from .timegrid import (
    PERIODS_PER_YEAR,
    Horizon,
    TimeGrid,
    billing_periods,
    ratio,
    resample_port,
    to_parameter,
    window_index,
)

COST_TERMS = ("C_inv", "C_mnt", "C_deg", "C_opr", "C_peak", "C_rev")


@dataclass
class BuiltHub:
    """The LP of a hub and everything needed to read its solution."""

    hub: Hub
    horizon: Horizon
    n_dl: float
    r: float
    backend: Backend
    grids: dict[str, TimeGrid]
    ports: dict[str, Any]  # "COMP.port" -> variable / expression / parameter
    vectors: dict[str, str]  # "COMP.port" -> vector
    sizes: dict[str, dict[str, Any]]  # COMP -> S_exist / S_max / S_new / S_total
    states: dict[str, Any]  # "COMP.x1" -> variable over k (x(1) .. x(N_h))
    costs: dict[str, dict[str, Any]]  # COMP -> C_inv ... C_rev, also for customs
    peaks: dict[str, Any]  # "COMP.port" -> peak variable over billing periods
    lamda: float
    annuity: float
    CAPEX: Any = None
    OPEX: Any = None
    REV: Any = None
    NPC: Any = None
    objective_scale: float = 1.0
    order: list[str] = field(default_factory=list)

    @property
    def model(self):
        return self.backend.model

    def grid_of(self, component: str) -> TimeGrid:
        return self.grids[component]


# --------------------------------------------------------------------------- #
# graph of the hub
# --------------------------------------------------------------------------- #
def _sampling_times(hub: Hub) -> dict[str, float]:
    """Sampling time of every component; subcomponents inherit, and must agree."""
    times: dict[str, float] = {}

    def visit(c: Component, inherited: float | None) -> None:
        t = c.t if c.t is not None else inherited
        if inherited is not None and c.t is not None and not math.isclose(c.t, inherited):
            raise ValueError(
                f"{c.name}: sampling time {c.t} h differs from {inherited} h of "
                f"{c.parent.name}; subcomponents of a custom component share it"
            )
        if t is None and isinstance(c, Standard):
            raise ValueError(f"{c.name}: no sampling time given")
        if t is not None:
            times[c.name] = t
        if isinstance(c, Custom):
            for sub in c.components:
                visit(sub, t)
            if t is None:  # a custom component without t takes it from its parts
                ts = {times[s.name] for s in c.components if s.name in times}
                if len(ts) != 1:
                    raise ValueError(f"{c.name}: subcomponents use different sampling times")
                times[c.name] = ts.pop()

    for c in hub.components:
        visit(c, None)
    return times


def _registry(hub: Hub) -> dict[str, tuple[Component, Any]]:
    """Every port of the hub: ``"COMP.port"`` → (component, port spec or alias)."""
    reg: dict[str, tuple[Component, Any]] = {}
    for c in hub.walk():
        for p, spec in c.ports.items():
            reg[f"{c.name}.{p}"] = (c, spec)
    return reg


def _deps(key: str, reg: dict[str, tuple[Component, Any]]) -> list[str]:
    comp, spec = reg[key]
    if isinstance(comp, Custom):  # external port extends a subcomponent port
        return [spec]
    deps = [f"{comp.name}.{d}" for d in spec.local_deps()] + spec.global_deps()
    for d in deps:
        if d not in reg:
            raise ValueError(f"{key} refers to unknown port {d}")
    return deps


def _topological(reg: dict[str, tuple[Component, Any]]) -> list[str]:
    deps = {k: _deps(k, reg) for k in reg}
    users: dict[str, list[str]] = defaultdict(list)
    for k, ds in deps.items():
        for d in ds:
            users[d].append(k)
    missing = {k: len(ds) for k, ds in deps.items()}
    queue = deque(k for k in reg if missing[k] == 0)  # declaration order kept
    order: list[str] = []
    while queue:
        k = queue.popleft()
        order.append(k)
        for u in users[k]:
            missing[u] -= 1
            if missing[u] == 0:
                queue.append(u)
    if len(order) != len(reg):
        stuck = [k for k in reg if missing[k] > 0]
        raise ValueError(f"circular port definitions among: {', '.join(stuck)}")
    return order


def _vectors(reg: dict[str, tuple[Component, Any]], order: list[str]) -> dict[str, str]:
    vec: dict[str, str] = {}
    for key in order:
        comp, spec = reg[key]
        if isinstance(comp, Custom):
            vec[key] = vec[spec]
        elif isinstance(spec, Other):
            vec[key] = vec[spec.ref]
            if spec.vector is not None and spec.vector != vec[key]:
                raise ValueError(f"{key} is {spec.vector}, {spec.ref} is {vec[key]}")
        else:
            if not spec.vector:
                raise ValueError(f"{key}: no vector given")
            vec[key] = spec.vector
    return vec


def _check_vectors(hub: Hub, vec: dict[str, str]) -> None:
    for c in hub.standards():
        if c.kind in ("storage", "bus"):
            used = {vec[f"{c.name}.{p}"] for p in c.ports}
            if len(used) > 1:
                raise ValueError(f"{c.name}: all ports of a {c.kind} use one vector, got {used}")


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #
def build(
    hub: Hub,
    horizon: Horizon,
    n_dl: float = 20,
    r: float = 0.03,
    scale_constraints: bool = True,
    scale_objective: bool = True,
    extend: Callable[[BuiltHub], None] | None = None,
) -> BuiltHub:
    """Formulate the LP of ``hub`` over ``horizon``; no solving happens here.

    ``extend(b)`` is called once all standard components are defined and before
    the cost terms are summed; it may add variables and constraints and change
    ``b.costs``.  It serves formulations outside the declared port types.
    """
    times = _sampling_times(hub)
    grids: dict[str, TimeGrid] = {}
    for top in hub.components:  # one time axis per top-level component
        grid = horizon.grid(times[top.name], f"k_{top.name}")
        grids[top.name] = grid
        if isinstance(top, Custom):
            for sub in top.walk():
                grids[sub.name] = grid

    reg = _registry(hub)
    order = _topological(reg)
    vec = _vectors(reg, order)
    _check_vectors(hub, vec)
    for key in order:  # integer ratio of sampling times of connected ports
        comp, spec = reg[key]
        if isinstance(spec, Other):
            src = reg[spec.ref][0]
            try:
                ratio(grids[src.name], grids[comp.name])
            except ValueError as exc:
                raise ValueError(f"{key} ← {spec.ref}: {exc}") from None

    be = Backend(scale_constraints=scale_constraints)
    b = BuiltHub(
        hub=hub,
        horizon=horizon,
        n_dl=n_dl,
        r=r,
        backend=be,
        grids=grids,
        ports={},
        vectors=vec,
        sizes={},
        states={},
        costs={},
        peaks={},
        lamda=horizon.lamda,
        annuity=annuity(r, n_dl),
        order=order,
    )

    for c in hub.standards():  # sizes first: DP/BP ports scale with them
        _define_size(b, c)
    for key in order:
        _define_port(b, key, *reg[key])
    for c in hub.standards():
        _define_states(b, c)
        _define_size_rule(b, c)
        _define_peaks(b, c)
        _define_bus(b, c)
        b.costs[c.name] = _standard_costs(b, c)
    if extend is not None:
        extend(b)
    for c in reversed(list(hub.walk())):  # children before parents
        if isinstance(c, Custom):
            _define_ratios(b, c)
            b.costs[c.name] = {
                term: lsum(*(b.costs[s.name][term] for s in c.components))
                for term in COST_TERMS
            }

    tops = hub.components
    b.CAPEX = _simplify(lsum(*(b.costs[c.name]["C_inv"] for c in tops)))
    b.OPEX = _simplify(
        lsum(
            *(
                b.costs[c.name][term]
                for c in tops
                for term in ("C_opr", "C_peak", "C_mnt", "C_deg")
            )
        )
    )
    b.REV = _simplify(lsum(*(b.costs[c.name]["C_rev"] for c in tops)))
    b.NPC = lsum(b.CAPEX, b.annuity * lsum(b.OPEX, -1 * b.REV))
    if not is_linear(b.NPC):
        raise ValueError("the objective does not depend on any decision variable")
    b.objective_scale = be.minimize(b.NPC, scale=scale_objective)
    return b


def _simplify(x: Any) -> Any:
    return x.simplify() if is_linear(x) and hasattr(x, "simplify") else x


def _name(key: str) -> str:
    return key.replace(".", "_")


# --- sizes --------------------------------------------------------------------
def _define_size(b: BuiltHub, c: Standard) -> None:
    if c.size is None:
        return
    s = c.size
    upper = s.s_new_max
    if upper == 0:  # nothing to decide, the size is the existing one
        s_new: Any = 0.0
        s_total: Any = float(s.s_exist)
    else:
        s_new = b.backend.decision(f"{c.name}_S_new", lower=0.0, upper=upper)
        s_total = s_new + s.s_exist
    b.sizes[c.name] = {
        "S_exist": float(s.s_exist),
        "S_max": s.s_max,
        "S_new": s_new,
        "S_total": s_total,
    }


def _define_size_rule(b: BuiltHub, c: Standard) -> None:
    if c.size is None or c.size.rule == "profile":
        return
    s_total = b.sizes[c.name]["S_total"]
    rule, items = c.size.rule, c.size.items
    if rule == "ports":
        rows = [(p, b.ports[f"{c.name}.{p}"]) for p in items]
    elif rule == "port_sum":
        rows = [("sum", lsum(*(b.ports[f"{c.name}.{p}"] for p in items)))]
    elif rule == "states":
        rows = [(s, b.states[f"{c.name}.{s}"]) for s in items]
    else:  # state_sum
        rows = [("sum", lsum(*(b.states[f"{c.name}.{s}"] for s in items)))]
    for label, value in rows:
        if not is_linear(value):
            continue  # a parameter cannot exceed the size it is defined from
        con = s_total >= value if is_linear(s_total) else value <= s_total
        b.backend.constrain(con, f"{c.name}_S_total_{label}")


# --- ports --------------------------------------------------------------------
def _define_port(b: BuiltHub, key: str, comp: Component, spec: Any) -> None:
    grid = b.grids[comp.name]
    be = b.backend
    if isinstance(comp, Custom):  # external port = the subcomponent port
        b.ports[key] = b.ports[spec]
        return

    p_max = to_parameter(spec.p_max, grid)
    upper = math.inf if p_max is None else p_max
    name = _name(key)

    if isinstance(spec, FC):
        value: Any = be.decision(name, coords=[grid.index], upper=upper)
    elif isinstance(spec, UC):
        value = to_parameter(spec.profile, grid)
        if not isinstance(value, xr.DataArray):
            value = xr.DataArray(np.full(grid.n_h, value), coords=[grid.index])
    elif isinstance(spec, LinComb):
        combo = lsum(*(a * b.ports[f"{comp.name}.{p}"] for p, a in spec.terms.items()))
        if spec.bounded:
            value = be.decision(name, coords=[grid.index], upper=upper)
            be.constrain(value <= combo, f"{name}_BLC")
        else:
            value = combo
            if any(a < 0 for a in spec.terms.values()):  # ports are non-negative
                if is_linear(value):
                    be.constrain(value >= 0, f"{name}_nonneg")
                elif float(np.min(value)) < 0:
                    raise ValueError(f"{key} is a negative parameter")
    elif isinstance(spec, Profile):
        zeta = to_parameter(spec.zeta, grid)
        s_total = b.sizes[comp.name]["S_total"]
        scaled = s_total * zeta if is_linear(s_total) else zeta * s_total
        if spec.bounded:
            value = be.decision(name, coords=[grid.index], upper=upper)
            be.constrain(value <= scaled, f"{name}_BP")
        else:
            value = scaled
    elif isinstance(spec, Amount):
        value = _define_amount(b, comp, spec, grid, name, p_max)
    elif isinstance(spec, Other):
        src_comp = spec.ref.split(".")[0]
        other = resample_port(b.ports[spec.ref], b.grids[src_comp], grid)
        if spec.bounded:
            value = be.decision(name, coords=[grid.index], upper=upper)
            be.constrain(value <= other, f"{name}_BO")
        else:
            value = other
    else:  # pragma: no cover
        raise TypeError(f"{key}: unknown port type {type(spec).__name__}")

    # decision variables carry P_max as their upper bound; an intermediate
    # variable (DLC, DP, DO) needs it as a row
    if p_max is not None and not spec.is_decision and is_linear(value):
        be.constrain(value <= p_max, f"{name}_max")
    b.ports[key] = value


def _define_amount(
    b: BuiltHub, comp: Standard, spec: Amount, grid: TimeGrid, name: str, p_max: Any
) -> Any:
    omega_of_k, amounts = window_index(spec.windows, grid, spec.rounding)
    inside = omega_of_k >= 0
    # instants outside every window form timeframes with E(ω) = 0, i.e. P = 0
    upper = np.where(inside, np.inf, 0.0)
    if p_max is not None:
        upper = np.minimum(upper, np.broadcast_to(np.asarray(p_max), upper.shape))
    value = b.backend.decision(
        name, coords=[grid.index], upper=xr.DataArray(upper, coords=[grid.index])
    )
    if len(amounts):
        dim = f"omega_{name}"
        k_in = np.flatnonzero(inside)
        groups = xr.DataArray(
            omega_of_k[k_in], coords=[grid.index[k_in]], name=dim
        )
        summed = grid.t * value.isel({grid.dim: k_in}).groupby(groups).sum()
        target = xr.DataArray(amounts, coords=[(dim, np.arange(len(amounts)))])
        con = summed <= target if spec.bounded else summed == target
        b.backend.constrain(con, f"{name}_amount")
    return value


# --- states -------------------------------------------------------------------
def _define_states(b: BuiltHub, c: Standard) -> None:
    grid, be = b.grids[c.name], b.backend
    dim = grid.dim
    for s, state in c.states.items():
        name = f"{c.name}_{s}"
        x0 = be.decision(f"{name}_0")  # x(0)
        x = be.decision(name, coords=[grid.index])  # x(1) .. x(N_h)
        delta = lsum(
            *(
                (grid.t * a if is_input(p) else -grid.t * a) * b.ports[f"{c.name}.{p}"]
                for p, a in state.alpha.items()
            )
        )
        sigma = to_parameter(state.sigma, grid)
        sig0 = sigma if not isinstance(sigma, xr.DataArray) else sigma.isel({dim: 0})
        sig_rest = (
            sigma
            if not isinstance(sigma, xr.DataArray)
            else sigma.isel({dim: slice(1, None)})
        )
        # x(k) relabelled to the instant k+1 it is propagated to, so that the
        # rows align by label as well as by position
        x_prev = x.isel({dim: slice(None, -1)}).assign_coords({dim: grid.index[1:]})
        be.constrain(
            x.isel({dim: 0}) == sig0 * x0 + delta.isel({dim: 0}), f"{name}_first"
        )
        be.constrain(
            x.isel({dim: slice(1, None)})
            == sig_rest * x_prev + delta.isel({dim: slice(1, None)}),
            f"{name}_dynamics",
        )
        be.constrain(x0 == x.isel({dim: -1}), f"{name}_repeatability")
        if state.gamma:
            s_total = b.sizes[c.name]["S_total"]
            be.constrain(x >= state.gamma * s_total, f"{name}_gamma")
        b.states[f"{c.name}.{s}"] = x


# --- peaks --------------------------------------------------------------------
def _define_peaks(b: BuiltHub, c: Standard) -> None:
    grid, be = b.grids[c.name], b.backend
    for p, spec in c.ports.items():
        if spec.peak is None:
            continue
        key, name = f"{c.name}.{p}", f"{c.name}_{p}_peak"
        omega = billing_periods(grid, spec.peak.periods)
        n = int(omega.max()) + 1
        dim = f"omega_{c.name}"
        coords = [(dim, np.arange(n))]
        peak = be.decision(name, coords=coords)
        indexer = xr.DataArray(omega, coords=[grid.index])
        value = b.ports[key]
        if is_linear(value):
            be.constrain(value - peak.isel({dim: indexer}) <= 0, name)
        else:  # a parameter: the peak is at least its maximum in the period
            be.constrain(peak >= value.groupby(xr.DataArray(omega, coords=[grid.index], name=dim)).max(), name)
        b.peaks[key] = peak


def _peak_cost(b: BuiltHub, c: Standard) -> Any:
    terms = []
    for p, spec in c.ports.items():
        if spec.peak is None:
            continue
        peak = b.peaks[f"{c.name}.{p}"]
        dim = peak.dims[0]
        n = peak.sizes[dim]
        lamda_omega = PERIODS_PER_YEAR[spec.peak.periods] / n
        price = spec.peak.c_peak
        if np.ndim(price) == 0:
            weighted = float(price) * peak
        else:
            if len(price) != n:
                raise ValueError(f"{c.name}.{p}: {len(price)} peak prices for {n} periods")
            weighted = xr.DataArray(np.asarray(price, float), coords=[(dim, np.arange(n))]) * peak
        terms.append(lamda_omega * weighted.sum())
    return lsum(*terms)


# --- buses --------------------------------------------------------------------
def _define_bus(b: BuiltHub, c: Standard) -> None:
    if c.kind != "bus" or c.has_defined_lincomb or c.balance == "open":
        return
    ins = lsum(*(b.ports[f"{c.name}.{p}"] for p in c.inputs))
    outs = lsum(*(b.ports[f"{c.name}.{p}"] for p in c.outputs))
    con = ins == outs if c.balance == "eq" else ins >= outs
    b.backend.constrain(con, f"{c.name}_balance")


# --- cost terms -------------------------------------------------------------
def _flow_cost(b: BuiltHub, c: Standard, attr: str) -> Any:
    grid = b.grids[c.name]
    terms = []
    for p, spec in c.ports.items():
        price = to_parameter(getattr(spec, attr), grid)
        if price is None:
            continue
        terms.append((price * b.ports[f"{c.name}.{p}"]).sum())
    if not terms:
        return 0
    return b.lamda * grid.t * lsum(*terms)


def _standard_costs(b: BuiltHub, c: Standard) -> dict[str, Any]:
    costs = c.costs
    size = b.sizes.get(c.name)
    s_new = size["S_new"] if size else 0
    s_total = size["S_total"] if size else 0
    c_deg_cal = costs.c_deg_cal(b.n_dl)

    if costs.degradation == "none":
        c_deg: Any = 0
    elif costs.degradation == "cal":
        c_deg = c_deg_cal * s_total
    else:  # "use" or "cmb": a decision variable, eqs. (C_deg_use), (C_deg_total)
        grid = b.grids[c.name]
        c_deg = b.backend.decision(f"{c.name}_C_deg")
        throughput = lsum(*(b.ports[f"{c.name}.{p}"].sum() for p in costs.deg_ports))
        use = b.lamda * grid.t * costs.c_deg_use * throughput
        b.backend.constrain(
            c_deg >= use - costs.c_rpl / b.n_dl * s_total, f"{c.name}_C_deg_use"
        )
        if costs.degradation == "cmb":
            b.backend.constrain(c_deg >= c_deg_cal * s_total, f"{c.name}_C_deg_cal")

    return {
        "C_inv": costs.c_inv * s_new if costs.c_inv else 0,
        "C_mnt": costs.c_mnt * s_total if costs.c_mnt else 0,
        "C_deg": c_deg,
        "C_opr": _flow_cost(b, c, "c_opr"),
        "C_peak": _peak_cost(b, c),
        "C_rev": _flow_cost(b, c, "c_rev"),
    }


# --- custom components ----------------------------------------------------------
def _define_ratios(b: BuiltHub, c: Custom) -> None:
    for r in c.ratios:
        s1 = b.sizes[r.c1]["S_total"]
        s2 = r.beta * b.sizes[r.c2]["S_total"]
        con = {"<=": s1 <= s2, "==": s1 == s2, ">=": s1 >= s2}[r.relation]
        b.backend.constrain(con, f"{c.name}_ratio_{r.c1}_{r.c2}")
