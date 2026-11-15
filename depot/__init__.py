"""The fleet depot energy hub of section Case study, built with :mod:`opt3s`.

Run from the repository root::

    python -m depot list                       # scenario sets and solvers
    python -m depot run R1                     # run a set, each scenario in its own process
    python -m depot run R1-1h --solver highs   # one scenario, another solver
    python -m depot describe                   # Supplementary Tables S5 and S6 from the code
"""

from .hub import COMPONENT_NAMES, build_hub
from .params import PARAMS, get_params

__all__ = ["COMPONENT_NAMES", "PARAMS", "build_hub", "get_params"]
