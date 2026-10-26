"""Kinematic + signature feature extraction for object classification."""
from __future__ import annotations

import math

import numpy as np

from ..tracking.history import resample, turn_rates

FEATURE_NAMES = [
    "log_speed_mean", "speed_std", "log_speed_max", "log_alt_mean", "alt_std", "vz_abs_mean", "vz_mean",
    "turn_rate_abs", "hover_frac", "straightness", "accel_mean",
    "rcs_mean", "rcs_std", "rotor_mean", "flap_mean", "has_radar",
    "log_size", "has_size", "has_adsb", "has_rid", "rotorcraft_cat", "priority_status", "log_age",
]


def _mean(vals, default):
    return float(np.mean(vals)) if len(vals) else default


def extract(track, now: float, window: float = 60.0) -> np.ndarray:
    s = resample(track.history, now, window, dt=1.0)
    if len(s) >= 2:
        sp = np.linalg.norm(s.vel, axis=1)
        alt = s.pos[:, 2]
        vz = s.vel[:, 2]
        tr = turn_rates(s)
        path = float(np.sum(np.linalg.norm(np.diff(s.pos[:, :2], axis=0), axis=1)))
        net = float(np.linalg.norm(s.pos[-1, :2] - s.pos[0, :2]))
        straight = net / path if path > 5 else 0.0
        acc = np.linalg.norm(np.diff(s.vel, axis=0), axis=1) / np.maximum(np.diff(s.t), 1e-3)
    else:
        v = track.velocity
        sp = np.array([np.linalg.norm(v)])
        alt = np.array([track.position[2]])
        vz = np.array([v[2]])
        tr = np.zeros(0)
        straight = 0.0
        acc = np.zeros(1)
    rcs = track.feature_values("rcs_dbsm")
    rotor = track.feature_values("rotor_score")
    flap = track.feature_values("flap_score")
    size = track.feature_values("size_m")
    cat = track.feature_values("emitter_category")
    prio = track.feature_values("priority_status")
    has_adsb = 1.0 if track.sources.get("adsb", 0) > 0 else 0.0
    has_rid = 1.0 if track.sources.get("remote_id", 0) > 0 else 0.0
    return np.array([
        math.log1p(float(np.mean(sp))),
        float(np.std(sp)),
        math.log1p(float(np.max(sp))),
        math.log1p(max(float(np.mean(alt)), 0.0)),
        float(np.std(alt)),
        float(np.mean(np.abs(vz))),
        float(np.mean(vz)),
        float(np.mean(np.abs(tr))) if len(tr) else 0.0,
        float(np.mean(sp < 1.5)),
        straight,
        float(np.mean(acc)),
        _mean(rcs, -15.0),
        float(np.std(rcs)) if len(rcs) > 1 else 0.0,
        _mean(rotor, 0.5),
        _mean(flap, 0.5),
        1.0 if rcs else 0.0,
        math.log(max(_mean(size, 1.0), 0.05)),
        1.0 if size else 0.0,
        has_adsb,
        has_rid,
        1.0 if cat and np.mean(cat) > 5 else 0.0,
        _mean(prio, 0.0),
        math.log1p(max(now - track.created, 0.0)),
    ])
