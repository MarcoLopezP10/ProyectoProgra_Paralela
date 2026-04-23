"""objectives.economic_dispatch

Economic Load Dispatch (ELD/EDL) objective support.

This module adds a constrained dispatch objective on top of the existing PSO
infrastructure without changing the PSO core. The search variables are the
generator power outputs, one dimension per generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class GeneratorUnit:
    """One thermal generator with optional valve-point coefficients."""

    name: str
    p_min: float
    p_max: float
    a: float
    b: float
    c: float
    e: float = 0.0
    f: float = 0.0

    def __post_init__(self) -> None:
        if float(self.p_min) >= float(self.p_max):
            raise ValueError(
                f"Generator {self.name!r} must satisfy p_min < p_max."
            )


@dataclass(frozen=True)
class LossModel:
    """Quadratic transmission-loss model: (P^T B P) / base + B0^T P + B00."""

    B: NDArray[np.float64]
    B0: NDArray[np.float64]
    B00: float = 0.0
    quadratic_base_mva: float = 1.0

    def __post_init__(self) -> None:
        B = np.asarray(self.B, dtype=float)
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise ValueError("Loss matrix B must be a square 2-D array.")

        B0 = np.asarray(self.B0, dtype=float)
        if B0.ndim != 1 or B0.shape[0] != B.shape[0]:
            raise ValueError("Loss vector B0 must have one entry per generator.")

        object.__setattr__(self, "B", B)
        object.__setattr__(self, "B0", B0)
        object.__setattr__(self, "B00", float(self.B00))
        if float(self.quadratic_base_mva) <= 0.0:
            raise ValueError("quadratic_base_mva must be strictly positive.")
        object.__setattr__(self, "quadratic_base_mva", float(self.quadratic_base_mva))


@dataclass(frozen=True)
class EconomicDispatchCase:
    """Full dispatch case definition."""

    case_name: str
    demand: float
    generators: tuple[GeneratorUnit, ...]
    loss_model: LossModel | None = None

    def __post_init__(self) -> None:
        if not self.generators:
            raise ValueError("Economic dispatch case requires at least one generator.")
        if float(self.demand) <= 0.0:
            raise ValueError("Demand must be strictly positive.")
        if self.loss_model is not None and self.loss_model.B.shape[0] != self.dim:
            raise ValueError("Loss model size must match the number of generators.")

    @property
    def dim(self) -> int:
        """Number of decision variables (one per generator)."""
        return len(self.generators)

    @property
    def generator_names(self) -> list[str]:
        """Ordered generator labels."""
        return [generator.name for generator in self.generators]

    @property
    def lower_bounds(self) -> list[float]:
        """Lower dispatch limits."""
        return [float(generator.p_min) for generator in self.generators]

    @property
    def upper_bounds(self) -> list[float]:
        """Upper dispatch limits."""
        return [float(generator.p_max) for generator in self.generators]


@dataclass(frozen=True)
class EconomicDispatchObjective:
    """Callable objective compatible with the project's PSO evaluators."""

    case: EconomicDispatchCase
    use_valve_point: bool = False
    use_losses: bool = False
    penalty_power_balance: float = 1e6
    _pmin: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _a: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _b: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _c: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _e: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _f: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.use_losses and self.case.loss_model is None:
            raise ValueError("use_losses=True requires a case with a loss model.")
        if float(self.penalty_power_balance) <= 0.0:
            raise ValueError("penalty_power_balance must be strictly positive.")

        object.__setattr__(
            self,
            "_pmin",
            np.asarray([generator.p_min for generator in self.case.generators], dtype=float),
        )
        object.__setattr__(
            self,
            "_a",
            np.asarray([generator.a for generator in self.case.generators], dtype=float),
        )
        object.__setattr__(
            self,
            "_b",
            np.asarray([generator.b for generator in self.case.generators], dtype=float),
        )
        object.__setattr__(
            self,
            "_c",
            np.asarray([generator.c for generator in self.case.generators], dtype=float),
        )
        object.__setattr__(
            self,
            "_e",
            np.asarray([generator.e for generator in self.case.generators], dtype=float),
        )
        object.__setattr__(
            self,
            "_f",
            np.asarray([generator.f for generator in self.case.generators], dtype=float),
        )

    def _as_vector(self, powers: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(powers, dtype=float)
        if values.ndim != 1 or values.shape[0] != self.case.dim:
            raise ValueError(
                f"Expected a 1-D power vector of length {self.case.dim}, "
                f"got shape {values.shape}."
            )
        return values

    def base_cost(self, powers: ArrayLike) -> float:
        """Quadratic fuel cost without valve-point ripples."""
        p = self._as_vector(powers)
        return float(np.sum(self._a * p**2 + self._b * p + self._c))

    def valve_point_cost(self, powers: ArrayLike) -> float:
        """Non-smooth valve-point contribution."""
        p = self._as_vector(powers)
        return float(np.sum(np.abs(self._e * np.sin(self._f * (self._pmin - p)))))

    def fuel_cost(self, powers: ArrayLike) -> float:
        """Fuel cost for the active EDL variant."""
        base = self.base_cost(powers)
        if not self.use_valve_point:
            return base
        return base + self.valve_point_cost(powers)

    def transmission_loss(self, powers: ArrayLike) -> float:
        """Transmission losses for the active EDL variant."""
        if not self.use_losses:
            return 0.0
        p = self._as_vector(powers)
        assert self.case.loss_model is not None  # guarded in __post_init__
        quadratic_term = float(p @ self.case.loss_model.B @ p)
        return float(
            quadratic_term / self.case.loss_model.quadratic_base_mva
            + self.case.loss_model.B0 @ p
            + self.case.loss_model.B00
        )

    def power_balance_residual(self, powers: ArrayLike) -> float:
        """Signed balance mismatch: generation - (demand + losses)."""
        p = self._as_vector(powers)
        losses = self.transmission_loss(p)
        required_generation = float(self.case.demand) + losses
        return float(np.sum(p) - required_generation)

    def power_balance_error(self, powers: ArrayLike) -> float:
        """Absolute balance mismatch."""
        return abs(self.power_balance_residual(powers))

    def penalty(self, powers: ArrayLike) -> float:
        """Quadratic penalty for power-balance infeasibility."""
        residual = self.power_balance_residual(powers)
        return float(self.penalty_power_balance * residual * residual)

    def components(self, powers: ArrayLike) -> dict[str, float]:
        """Return the full objective breakdown for one dispatch vector."""
        p = self._as_vector(powers)
        base = self.base_cost(p)
        valve = self.valve_point_cost(p) if self.use_valve_point else 0.0
        fuel = base + valve
        losses = self.transmission_loss(p)
        residual = float(np.sum(p) - (float(self.case.demand) + losses))
        error = abs(residual)
        penalty = float(self.penalty_power_balance * residual * residual)
        return {
            "base_cost": base,
            "valve_point_cost": valve,
            "fuel_cost": fuel,
            "transmission_loss": losses,
            "power_balance_residual": residual,
            "power_balance_error": error,
            "penalty": penalty,
            "fitness": fuel + penalty,
        }

    def __call__(self, powers: ArrayLike) -> float:
        """Return the scalar fitness value consumed by PSO."""
        return float(self.components(powers)["fitness"])


def load_case(path: str | Path) -> EconomicDispatchCase:
    """Load an economic dispatch case from JSON."""
    case_path = Path(path)
    with case_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    generators_data = data.get("generators")
    if not isinstance(generators_data, list) or not generators_data:
        raise ValueError("Case JSON must provide a non-empty 'generators' list.")

    generators = tuple(
        GeneratorUnit(
            name=str(generator["name"]),
            p_min=float(generator["p_min"]),
            p_max=float(generator["p_max"]),
            a=float(generator["a"]),
            b=float(generator["b"]),
            c=float(generator["c"]),
            e=float(generator.get("e", 0.0)),
            f=float(generator.get("f", 0.0)),
        )
        for generator in generators_data
    )

    losses_data = data.get("losses")
    loss_model = None
    if losses_data is not None:
        zeros = [0.0] * len(generators)
        loss_model = LossModel(
            B=np.asarray(losses_data["B"], dtype=float),
            B0=np.asarray(losses_data.get("B0", zeros), dtype=float),
            B00=float(losses_data.get("B00", 0.0)),
            quadratic_base_mva=float(losses_data.get("quadratic_base_mva", 1.0)),
        )

    case_name = str(data.get("case_name", case_path.stem))
    return EconomicDispatchCase(
        case_name=case_name,
        demand=float(data["demand"]),
        generators=generators,
        loss_model=loss_model,
    )


__all__ = [
    "EconomicDispatchCase",
    "EconomicDispatchObjective",
    "GeneratorUnit",
    "LossModel",
    "load_case",
]
