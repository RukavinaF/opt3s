"""Standard and custom components, subsection "Standard and custom components".

A :class:`Standard` component is declared only by its ports, states, size and
cost terms.  A :class:`Custom` component is a set of subcomponents whose ports
are exposed under its own names.  The :class:`Hub` is the top-level custom
component without external ports.

Nothing here touches the optimization model; :mod:`opt3s.build` turns a hub
into an LP.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

from .ports import PORT_NAME, LinComb, Port, is_input
from .specs import Costs, Ratio, Size, State

KINDS = ("source", "sink", "process", "storage", "bus")
BALANCE = ("eq", "ineq", "open")


class Component:
    """Common part of standard and custom components."""

    def __init__(self, name: str, t: float | None = None, label: str | None = None):
        if not name or "." in name:
            raise ValueError(f"invalid component name {name!r}")
        self.name = name
        self.t = None if t is None else float(t)
        self.label = label or name
        self.parent: Custom | None = None

    @property
    def ports(self) -> Mapping[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class Standard(Component):
    """A standard component: source, sink, process, storage or bus.

    Parameters
    ----------
    kind
        ``"source"`` (outputs only), ``"sink"`` (inputs only), ``"process"``
        (inputs and outputs), ``"storage"`` (states, inputs and outputs of one
        vector) or ``"bus"`` (at least three ports of one vector).
    ports
        Port name (``u1``, ``y1``, ...) → port type, see :mod:`opt3s.ports`.
    states
        State name (``x1``, ...) → :class:`~opt3s.specs.State`.
    size
        :class:`~opt3s.specs.Size`, or ``None`` for a component without size.
    costs
        :class:`~opt3s.specs.Costs` of the size-dependent cost terms.
    balance
        Buses only: ``"eq"`` for eq. (bus_eq), or ``"ineq"`` for
        ``Σ u_i(k) >= Σ y_j(k)``, i.e. a surplus of the inputs is released.
        ``"open"`` adds no balance; it is closed by an ``extend`` function of
        :func:`~opt3s.build.build` (used for comparisons only).
        Applied only if no port of the bus is defined by linear combination,
        since such a port already closes the balance.
    t
        Sampling time [h]; inherited from the enclosing custom component if not
        given.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        ports: Mapping[str, Port],
        states: Mapping[str, State] | None = None,
        size: Size | None = None,
        costs: Costs | None = None,
        balance: str = "eq",
        t: float | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(name, t, label)
        if kind not in KINDS:
            raise ValueError(f"{name}: unknown kind {kind!r}, use one of {KINDS}")
        if balance not in BALANCE:
            raise ValueError(f"{name}: unknown balance {balance!r}")
        self.kind = kind
        self._ports = dict(ports)
        self.states = dict(states or {})
        self.size = size
        self.costs = costs or Costs()
        self.balance = balance
        self._check()

    @property
    def ports(self) -> Mapping[str, Port]:
        return self._ports

    @property
    def inputs(self) -> list[str]:
        return [p for p in self._ports if is_input(p)]

    @property
    def outputs(self) -> list[str]:
        return [p for p in self._ports if not is_input(p)]

    def _check(self) -> None:
        """Rules that can be checked without the rest of the hub."""
        n = self.name
        for p, spec in self._ports.items():
            if not PORT_NAME.match(p):
                raise ValueError(f"{n}: invalid port name {p!r}")
            if not isinstance(spec, Port):
                raise TypeError(f"{n}.{p}: {spec!r} is not a port type")
            for dep in spec.local_deps():
                if dep not in self._ports:
                    raise ValueError(f"{n}.{p} refers to unknown port {dep!r}")
                if dep == p:
                    raise ValueError(f"{n}.{p} is defined by itself")
        ins, outs = self.inputs, self.outputs
        if self.kind == "source" and ins:
            raise ValueError(f"{n}: a source has only outputs")
        if self.kind == "sink" and outs:
            raise ValueError(f"{n}: a sink has only inputs")
        if self.kind in ("process", "storage") and (not ins or not outs):
            raise ValueError(f"{n}: a {self.kind} needs at least one input and output")
        if self.kind == "storage" and not self.states:
            raise ValueError(f"{n}: a storage needs at least one state")
        if self.kind != "storage" and self.states:
            raise ValueError(f"{n}: only storages have states")
        if self.kind == "bus" and len(self._ports) < 3:
            raise ValueError(f"{n}: a bus needs at least three ports")
        for s, state in self.states.items():
            for p in state.alpha:
                if p not in self._ports:
                    raise ValueError(f"{n}.{s} refers to unknown port {p!r}")
        if self.size is not None and self.size.rule in ("ports", "port_sum"):
            for p in self.size.items:
                if p not in self._ports:
                    raise ValueError(f"{n}: size refers to unknown port {p!r}")
        if self.size is not None and self.size.rule in ("states", "state_sum"):
            for s in self.size.items:
                if s not in self.states:
                    raise ValueError(f"{n}: size refers to unknown state {s!r}")
        profile_ports = [p for p, s in self._ports.items() if s.abbrev in ("DP", "BP")]
        if profile_ports and (self.size is None or self.size.rule != "profile"):
            raise ValueError(f"{n}: ports {profile_ports} need Size.by_profile()")
        if self.size is None and (self.costs.c_inv or self.costs.c_mnt):
            raise ValueError(f"{n}: size-dependent costs without a size")
        for p in self.costs.deg_ports:
            if p not in self._ports:
                raise ValueError(f"{n}: degradation refers to unknown port {p!r}")
        if self.costs.degradation != "none" and self.size is None:
            raise ValueError(f"{n}: degradation without a size")

    @property
    def has_defined_lincomb(self) -> bool:
        return any(isinstance(s, LinComb) and not s.bounded for s in self._ports.values())


class Custom(Component):
    """A component composed of other components, Algorithm 2.

    Parameters
    ----------
    components
        Subcomponents (standard or custom).  All of them use the sampling time
        of this component.
    ports
        External port name → ``"SUB.port"`` it extends.
    ratios
        Size ratios between standard subcomponents, eq. (size_ratio1).
    """

    def __init__(
        self,
        name: str,
        components: Sequence[Component],
        ports: Mapping[str, str] | None = None,
        ratios: Sequence[Ratio | tuple] = (),
        t: float | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(name, t, label)
        self.components = list(components)
        self._ports = dict(ports or {})
        self.ratios = [r if isinstance(r, Ratio) else Ratio(*r) for r in ratios]
        names = [c.name for c in self.components]
        duplicates = {x for x in names if names.count(x) > 1}
        if duplicates:
            raise ValueError(f"{name}: duplicate subcomponents {sorted(duplicates)}")
        for c in self.components:
            if c.parent is not None:
                raise ValueError(f"{c.name} is already part of {c.parent.name}")
            c.parent = self
        for p, target in self._ports.items():
            if not PORT_NAME.match(p):
                raise ValueError(f"{name}: invalid port name {p!r}")
            sub, _, sub_port = target.partition(".")
            if sub not in names:
                raise ValueError(f"{name}.{p} extends unknown subcomponent {sub!r}")
            if is_input(p) != is_input(sub_port):
                raise ValueError(f"{name}.{p} and {target} differ in direction")
        for r in self.ratios:
            for c in (r.c1, r.c2):
                if c not in names:
                    raise ValueError(f"{name}: ratio refers to unknown {c!r}")

    @property
    def ports(self) -> Mapping[str, str]:
        return self._ports

    def walk(self) -> Iterator[Component]:
        """All subcomponents, depth first, parents before children."""
        for c in self.components:
            yield c
            if isinstance(c, Custom):
                yield from c.walk()

    def standards(self) -> Iterator[Standard]:
        return (c for c in self.walk() if isinstance(c, Standard))


class Hub(Custom):
    """The complete energy hub: a top-level custom component without external ports."""

    def __init__(self, name: str, components: Sequence[Component], label: str | None = None):
        super().__init__(name, components, ports=None, label=label)
        all_names = [c.name for c in self.walk()]
        duplicates = {x for x in all_names if all_names.count(x) > 1}
        if duplicates:
            raise ValueError(f"component names must be unique, repeated: {sorted(duplicates)}")

    def __getitem__(self, name: str) -> Component:
        for c in self.walk():
            if c.name == name:
                return c
        raise KeyError(name)

    def top_level(self, component: Component) -> Component:
        """The direct child of the hub that contains ``component``."""
        while component.parent is not self:
            if component.parent is None:
                raise ValueError(f"{component.name} is not part of {self.name}")
            component = component.parent
        return component
