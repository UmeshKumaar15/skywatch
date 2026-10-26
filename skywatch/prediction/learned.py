"""Data-driven motion model.

Inputs are expressed in a *body frame* (origin at the current position, x
axis along the current horizontal velocity) and normalised by speed, so a
single network generalises across slow birds and fast airliners. The network
predicts the residual between the true future path and a constant-velocity
extrapolation, i.e. it learns *how motion deviates from straight-line
physics* (turns, decelerations, route following, orbits...). A per-horizon
uncertainty model, fitted on held-out data, turns the output into a Gaussian
hypothesis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..classification.nn import MLP
from ..core.types import OBJECT_CLASSES
from ..tracking.history import resample

HIST_SECONDS = 12
DEFAULT_TIMES = np.unique(np.r_[np.arange(1.0, 5.0), np.arange(2.0, 60.0 + 1e-9, 2.0)])


def horizon_scale(times: np.ndarray) -> np.ndarray:
    """Per-horizon target normalisation so short and long horizons weigh comparably in the loss."""
    return np.repeat(1.0 + times / 10.0, 3)


@dataclass
class BodyFrame:
    origin: np.ndarray  # world position at reference time
    heading: float  # rad, ENU (atan2(vy, vx))
    scale: float  # m/s normalisation
    velocity: np.ndarray  # world velocity used for the CV baseline

    def rot(self) -> np.ndarray:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # world -> body


def build_input(history, t_ref: float, pos: np.ndarray, vel: np.ndarray, class_probs: np.ndarray,
                mode_probs: np.ndarray) -> Optional[tuple[np.ndarray, BodyFrame]]:
    s = resample(history, t_ref, HIST_SECONDS, dt=1.0)
    if len(s) < 4:
        return None
    hs = float(np.linalg.norm(vel[:2]))
    heading = math.atan2(vel[1], vel[0]) if hs > 0.5 else 0.0
    frame = BodyFrame(pos.copy(), heading, max(hs, 2.0), vel.copy())
    R = frame.rot()
    rel = (s.pos - pos) @ R.T / frame.scale  # (n, 3) body-frame, seconds-equivalent
    hist = np.zeros((HIST_SECONDS + 1, 3))
    hist[-len(rel):] = rel[-(HIST_SECONDS + 1):]
    if len(rel) < HIST_SECONDS + 1:  # pad with the oldest available point
        hist[: HIST_SECONDS + 1 - len(rel)] = rel[0]
    vb = R @ vel / frame.scale
    x = np.r_[hist.ravel(), vb, math.log1p(hs), math.log1p(max(pos[2], 0.0)), vel[2] / 10.0, class_probs, mode_probs]
    return x, frame


def targets_from_truth(future: np.ndarray, frame: BodyFrame, times: np.ndarray) -> np.ndarray:
    """future: (H, 3) world positions at ``times`` -> normalised body-frame residuals (H*3,)."""
    cv = frame.origin + frame.velocity * times[:, None]
    res = (future - cv) @ frame.rot().T / frame.scale
    return res.ravel()


class LearnedMotionModel:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.net: Optional[MLP] = None
        self.times = DEFAULT_TIMES
        self.sigma_norm: Optional[np.ndarray] = None  # (H, 3) residual std in normalised units
        self.sigma_floor: Optional[np.ndarray] = None  # (H,) absolute floor in metres
        self.out_scale: Optional[np.ndarray] = None
        if path and Path(path).exists():
            d = np.load(path, allow_pickle=False)
            self.net = MLP.from_state(d, "mm_")
            self.times = d["times"]
            self.sigma_norm = d["sigma_norm"]
            self.sigma_floor = d["sigma_floor"]
            self.out_scale = d["out_scale"] if "out_scale" in d else np.ones(len(self.times) * 3)

    @property
    def ready(self) -> bool:
        return self.net is not None

    def predict(self, x: np.ndarray, frame: BodyFrame, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame means (len(times), 3) and covariances (len(times), 3, 3)."""
        out = (self.net.predict(x[None, :])[0] * self.out_scale).reshape(len(self.times), 3)
        Rt = frame.rot().T  # body -> world
        cv = frame.origin + frame.velocity * self.times[:, None]
        means_m = cv + (out * frame.scale) @ Rt.T
        sig = self.sigma_norm * frame.scale + self.sigma_floor[:, None]
        mt = np.r_[0.0, self.times]
        means_m = np.vstack([frame.origin, means_m])
        sig = np.vstack([np.zeros(3), sig])
        means = np.stack([np.interp(times, mt, means_m[:, k]) for k in range(3)], axis=1)
        # Extrapolate beyond the trained horizon linearly in time.
        if times[-1] > self.times[-1]:
            late = times > self.times[-1]
            means[late] = means_m[-1] + frame.velocity * (times[late, None] - self.times[-1])
        sig_t = np.stack([np.interp(times, mt, sig[:, k]) for k in range(3)], axis=1)
        covs = np.einsum("ab,hb,cb->hac", Rt, sig_t**2, Rt)
        return means, covs


CLASS_DIM = len(OBJECT_CLASSES)
