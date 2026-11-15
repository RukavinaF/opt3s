"""The only module that talks to linopy.

It implements the elementary *define* operation of subsection Implementation:
appending a decision variable to χ, declaring an intermediate variable as a
linear expression, or appending rows to ``A_eq, b_eq`` / ``A_ineq, b_ineq``.
Optionally, every row is normalised to unit coefficient norm (numerical
conditioning only, the feasible set is unchanged).
"""

from __future__ import annotations

import math
from typing import Any

import linopy as lp
import numpy as np
import xarray as xr

LINOPY_TYPES = (lp.Variable, lp.LinearExpression)


def is_linear(value: Any) -> bool:
    return isinstance(value, LINOPY_TYPES)


def lsum(*terms: Any) -> Any:
    """Sum of variables, expressions, DataArrays and numbers, in a safe order.

    ``DataArray + LinearExpression`` lets xarray try first and can fail, so the
    linear terms are summed first and the parameters added to them.
    """
    terms = [t for t in terms if not (isinstance(t, (int, float)) and t == 0)]
    linear = [t for t in terms if is_linear(t)]
    other = [t for t in terms if not is_linear(t)]
    if not linear:
        return sum(other) if other else 0
    total = linear[0] if isinstance(linear[0], lp.LinearExpression) else 1 * linear[0]
    for t in linear[1:] + other:
        total = total + t
    return total


def _norm_rows(coeffs: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(coeffs, nan=0.0)
    nrm = np.linalg.norm(data, axis=-1)
    return np.where(nrm > 0, nrm, 1.0)


class Backend:
    """A linopy model with the define operations used by :mod:`opt3s.build`."""

    def __init__(self, scale_constraints: bool = True) -> None:
        self.model = lp.Model()
        self.scale_constraints = scale_constraints

    # --- χ: decision variables ---------------------------------------------
    def decision(
        self,
        name: str,
        coords: Any = None,
        lower: Any = 0.0,
        upper: Any = math.inf,
    ) -> lp.Variable:
        kwargs = {"lower": lower, "upper": upper, "name": name}
        if coords is not None:
            kwargs["coords"] = coords
        return self.model.add_variables(**kwargs)

    # --- rows of A_eq / A_ineq ----------------------------------------------
    def constrain(self, constraint: lp.constraints.Constraint, name: str) -> None:
        """Add ``constraint`` (built with ``==``, ``<=`` or ``>=``) under ``name``."""
        if self.scale_constraints:
            constraint = self._scaled(constraint)
        self.model.add_constraints(constraint, name=name)

    @staticmethod
    def _scaled(con: Any) -> Any:
        coeffs = con.coeffs.data
        nrm = _norm_rows(coeffs)
        con.coeffs.data = coeffs / nrm[..., np.newaxis]
        con.rhs.data = con.rhs.data / nrm
        return con

    # --- objective ------------------------------------------------------------
    def minimize(self, expr: lp.LinearExpression, scale: bool = True) -> float:
        """Set ``expr`` (minus its constant) as the objective; returns the scale used."""
        expr = 1 * expr
        nrm = 1.0
        if scale:
            nrm = float(np.linalg.norm(np.nan_to_num(expr.coeffs.data, nan=0.0)))
            nrm = nrm if nrm > 0 else 1.0
            expr.coeffs.data = expr.coeffs.data / nrm
        self.model.add_objective(expr.reset_const(), sense="min")
        return nrm


def value_of(x: Any) -> Any:
    """Solved value of a variable, expression, parameter or number."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, lp.Variable):
        try:
            return x.solution
        except KeyError:  # a resampled view of a variable keeps no solution itself
            return (1 * x).solution
    if is_linear(x):
        return x.solution
    if isinstance(x, xr.DataArray):
        return x
    raise TypeError(f"cannot evaluate {type(x).__name__}")


def scalar(x: Any) -> float:
    v = value_of(x)
    if isinstance(v, xr.DataArray):
        if v.size != 1:
            raise ValueError("not a scalar")
        return float(v.values)
    return float(v)
