"""opt3s: joint sizing and scheduling of energy hubs with heterogeneous sampling times.

Implements the framework of section Methodology: components declared only by
their ports, states, size and cost terms; standard and custom components;
component-specific sampling times; an LP that maximizes the NPV.

Typical use::

    from opt3s import Hub, Standard, FC, Size, Costs, Horizon, build, solve, HubResult

    hub = Hub("example", [...])
    built = build(hub, Horizon(datetime(2025, 1, 1), 365), n_dl=20, r=0.03)
    solve(built, solver="highs")
    result = HubResult(built)
    result.economics()
"""

from .build import COST_TERMS, BuiltHub, build
from .components import Component, Custom, Hub, Standard
from .describe import describe
from .ports import FC, UC, Amount, LinComb, Other, PeakBilling, Port, Profile
from .result import HubResult
from .solve import available_solvers, resolve_solver, solve
from .specs import Costs, Ratio, Size, State, annuity
from .timegrid import Horizon, TimeGrid, Window, resample_port, resample_profile

__version__ = "0.1.0"

__all__ = [
    "Amount", "BuiltHub", "COST_TERMS", "Component", "Costs", "Custom", "FC", "Horizon",
    "Hub", "HubResult", "LinComb", "Other", "PeakBilling", "Port", "Profile", "Ratio",
    "Size", "Standard", "State", "TimeGrid", "UC", "Window", "annuity",
    "available_solvers", "build", "describe", "resample_port", "resample_profile",
    "resolve_solver", "solve",
]  # fmt: skip
