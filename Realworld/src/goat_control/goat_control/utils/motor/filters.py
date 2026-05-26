# goat_control/core/estimation/filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union


Number = Union[int, float]
Vector = List[float]

# ============================================================================
# [DEPRECATED] FirstOrderLowPassFilter — generic per-element LPF. Superseded
# by the vectorized per-joint torque LPF inside MotorManager.torque_lpf(). Not
# imported or instantiated anywhere in the current codebase.
# ============================================================================

# [DEPRECATED] @dataclass
# [DEPRECATED] class FirstOrderLowPassFilter:
# [DEPRECATED]     """First-order low-pass filter.
# [DEPRECATED] 
# [DEPRECATED]     y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
# [DEPRECATED] 
# [DEPRECATED]     alpha:
# [DEPRECATED]       - 1.0  -> no filtering (y = x)
# [DEPRECATED]       - 0.0  -> output freezes at initial y
# [DEPRECATED]     """
# [DEPRECATED]     alpha: float
# [DEPRECATED]     initial_value: Optional[Union[Number, Sequence[Number]]] = None
# [DEPRECATED] 
# [DEPRECATED]     def __post_init__(self) -> None:
# [DEPRECATED]         self.alpha = float(self.alpha)
# [DEPRECATED]         if not (0.0 <= self.alpha <= 1.0):
# [DEPRECATED]             raise ValueError("alpha must be in [0, 1].")
# [DEPRECATED] 
# [DEPRECATED]         self._initialized = False
# [DEPRECATED]         self._y_scalar: float = 0.0
# [DEPRECATED]         self._y_vector: Vector = []
# [DEPRECATED] 
# [DEPRECATED]         if self.initial_value is not None:
# [DEPRECATED]             self.reset(self.initial_value)
# [DEPRECATED] 
# [DEPRECATED]     def reset(self, value: Union[Number, Sequence[Number]]) -> None:
# [DEPRECATED]         """Reset internal state."""
# [DEPRECATED]         if isinstance(value, (int, float)):
# [DEPRECATED]             self._y_scalar = float(value)
# [DEPRECATED]             self._y_vector = []
# [DEPRECATED]         else:
# [DEPRECATED]             self._y_vector = [float(v) for v in value]
# [DEPRECATED]             self._y_scalar = 0.0
# [DEPRECATED]         self._initialized = True
# [DEPRECATED] 
# [DEPRECATED]     def apply(self, x: Union[Number, Sequence[Number]]) -> Union[float, Vector]:
# [DEPRECATED]         """Filter scalar or vector input."""
# [DEPRECATED]         if isinstance(x, (int, float)):
# [DEPRECATED]             x_scalar = float(x)
# [DEPRECATED]             if not self._initialized:
# [DEPRECATED]                 self.reset(x_scalar)
# [DEPRECATED]                 return x_scalar
# [DEPRECATED]             self._y_scalar = self.alpha * x_scalar + (1.0 - self.alpha) * self._y_scalar
# [DEPRECATED]             return self._y_scalar
# [DEPRECATED] 
# [DEPRECATED]         x_vector = [float(v) for v in x]
# [DEPRECATED]         if not self._initialized:
# [DEPRECATED]             self.reset(x_vector)
# [DEPRECATED]             return x_vector
# [DEPRECATED] 
# [DEPRECATED]         # Vector filtering
# [DEPRECATED]         if not self._y_vector:
# [DEPRECATED]             # Safety fallback
# [DEPRECATED]             self._y_vector = x_vector.copy()
# [DEPRECATED] 
# [DEPRECATED]         if len(x_vector) != len(self._y_vector):
# [DEPRECATED]             raise ValueError("Vector length changed. Reset the filter or keep a fixed size.")
# [DEPRECATED] 
# [DEPRECATED]         for i in range(len(x_vector)):
# [DEPRECATED]             self._y_vector[i] = self.alpha * x_vector[i] + (1.0 - self.alpha) * self._y_vector[i]
# [DEPRECATED]         return self._y_vector


