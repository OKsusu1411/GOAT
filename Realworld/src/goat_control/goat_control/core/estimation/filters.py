# goat_control/core/estimation/filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union


Number = Union[int, float]
Vector = List[float]


@dataclass
class FirstOrderLowPassFilter:
    """First-order low-pass filter.

    y[k] = alpha * x[k] + (1 - alpha) * y[k-1]

    alpha:
      - 1.0  -> no filtering (y = x)
      - 0.0  -> output freezes at initial y
    """
    alpha: float
    initial_value: Optional[Union[Number, Sequence[Number]]] = None

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")

        self._initialized = False
        self._y_scalar: float = 0.0
        self._y_vector: Vector = []

        if self.initial_value is not None:
            self.reset(self.initial_value)

    def reset(self, value: Union[Number, Sequence[Number]]) -> None:
        """Reset internal state."""
        if isinstance(value, (int, float)):
            self._y_scalar = float(value)
            self._y_vector = []
        else:
            self._y_vector = [float(v) for v in value]
            self._y_scalar = 0.0
        self._initialized = True

    def apply(self, x: Union[Number, Sequence[Number]]) -> Union[float, Vector]:
        """Filter scalar or vector input."""
        if isinstance(x, (int, float)):
            x_scalar = float(x)
            if not self._initialized:
                self.reset(x_scalar)
                return x_scalar
            self._y_scalar = self.alpha * x_scalar + (1.0 - self.alpha) * self._y_scalar
            return self._y_scalar

        x_vector = [float(v) for v in x]
        if not self._initialized:
            self.reset(x_vector)
            return x_vector

        # Vector filtering
        if not self._y_vector:
            # Safety fallback
            self._y_vector = x_vector.copy()

        if len(x_vector) != len(self._y_vector):
            raise ValueError("Vector length changed. Reset the filter or keep a fixed size.")

        for i in range(len(x_vector)):
            self._y_vector[i] = self.alpha * x_vector[i] + (1.0 - self.alpha) * self._y_vector[i]
        return self._y_vector


