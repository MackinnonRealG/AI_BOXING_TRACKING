"""One-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

An adaptive low-pass filter: smooths hard at low speeds (removes jitter that
would register as phantom velocity) and lightly at high speeds (preserves the
punch peaks we measure). This speed-adaptive behavior is why it is preferred
over a Kalman filter for interactive pose streams.
"""

from __future__ import annotations

import math


class OneEuroFilter:
    """Filters a single scalar signal sampled at irregular intervals."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._prev_value: float | None = None
        self._prev_derivative = 0.0
        self._prev_t: float | None = None

    def filter(self, value: float, t: float) -> float:
        """Return the filtered value for a new sample at time ``t`` (seconds)."""
        if self._prev_value is None or self._prev_t is None or t <= self._prev_t:
            self._prev_value, self._prev_t = value, t
            return value

        dt = t - self._prev_t
        derivative = (value - self._prev_value) / dt
        d_alpha = _alpha(self._d_cutoff, dt)
        smoothed_derivative = _lerp(self._prev_derivative, derivative, d_alpha)

        cutoff = self._min_cutoff + self._beta * abs(smoothed_derivative)
        alpha = _alpha(cutoff, dt)
        smoothed = _lerp(self._prev_value, value, alpha)

        self._prev_value = smoothed
        self._prev_derivative = smoothed_derivative
        self._prev_t = t
        return smoothed


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low-pass at ``cutoff`` Hz."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _lerp(a: float, b: float, alpha: float) -> float:
    """Linear interpolation from ``a`` to ``b``."""
    return a + alpha * (b - a)
