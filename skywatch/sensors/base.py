"""Sensor abstraction layer.

A *sensor source* is anything that produces batches of normalised
``Observation`` objects (one batch = one scan / message burst of one
sensor). The pipeline consumes sources through a single async interface, so
simulated sensors, file replays, network feeds and real hardware drivers are
interchangeable.
"""
from __future__ import annotations

import math
from typing import AsyncIterator

import numpy as np

from ..core.types import Observation


class SensorSource:
    """Async producer of observation batches."""

    name: str = "source"

    async def stream(self) -> AsyncIterator[list[Observation]]:  # pragma: no cover - interface
        raise NotImplementedError
        yield []


def polar_to_enu(origin: np.ndarray, rng: float, az: float, el: float) -> np.ndarray:
    """Range / azimuth (from north, clockwise) / elevation -> ENU position."""
    ce = math.cos(el)
    return origin + np.array([rng * ce * math.sin(az), rng * ce * math.cos(az), rng * math.sin(el)])


def enu_to_polar(origin: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    d = p - origin
    r = float(np.linalg.norm(d))
    az = math.atan2(d[0], d[1])
    el = math.asin(np.clip(d[2] / max(r, 1e-9), -1, 1))
    return r, az, el


def polar_covariance(rng: float, az: float, el: float, s_r: float, s_az: float, s_el: float) -> np.ndarray:
    """First-order propagation of polar measurement noise into ENU covariance."""
    ce, se, ca, sa = math.cos(el), math.sin(el), math.cos(az), math.sin(az)
    J = np.array(
        [
            [ce * sa, rng * ce * ca, -rng * se * sa],
            [ce * ca, -rng * ce * sa, -rng * se * ca],
            [se, 0.0, rng * ce],
        ]
    )
    S = np.diag([s_r**2, s_az**2, s_el**2])
    return J @ S @ J.T
