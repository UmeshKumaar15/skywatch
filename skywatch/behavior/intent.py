"""Operational intent inference.

Behaviour features are extracted from the recent track history (path
straightness, orbit consistency, survey-leg reversals, hover fraction,
corridor adherence...). Each intent has an interpretable log-likelihood
built from soft (sigmoid / Gaussian) evidence terms, coupled with the class
posterior. A sticky hidden-Markov forward filter accumulates the evidence
over time so intents change only when behaviour persistently changes.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..core.types import INTENTS, Intent
from ..tracking.history import distance_to_polylines, fit_circle, polyline_distances, resample, turn_rates, wrap_angle

N_INT = len(INTENTS)
IDX = {it: i for i, it in enumerate(INTENTS)}


def _sig(x: float, center: float, width: float) -> float:
    z = (x - center) / max(width, 1e-9)
    z = max(min(z, 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-z))


def behaviour_features(track, now: float, window: float, geo: dict) -> Optional[dict[str, float]]:
    s = resample(track.history, now, window, dt=1.0)
    if len(s) < 6:
        return None
    sp = np.linalg.norm(s.vel[:, :2], axis=1)
    xy = s.pos[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(seg.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    dh = turn_rates(s, min_speed=1.5)
    tot_turn = float(np.sum(np.abs(dh)))  # rad (1 s sampling)
    consistency = abs(float(np.sum(dh))) / tot_turn if tot_turn > 1e-3 else 0.0
    # Survey reversals: heading changes of > 150 deg completed within 3-19 s (vectorised).
    h = s.heading
    moving = sp > 2.0
    n = len(h)
    rev = np.zeros(n, dtype=bool)
    for k in range(3, 20):
        if k >= n:
            break
        big = (np.abs(wrap_angle(h[k:] - h[:-k])) > math.radians(150)) & moving[k:] & moving[:-k]
        rev[:-k] |= big
    # Count separate reversal events (rising edges, at least 8 s apart).
    reversals = 0
    last = -100
    for i in np.nonzero(rev & ~np.r_[False, rev[:-1]])[0]:
        if i - last >= 8:
            reversals += 1
            last = i
    c, r, resid = fit_circle(xy) if len(xy) >= 10 else (xy.mean(axis=0), math.inf, math.inf)
    circ_quality = 0.0
    if math.isfinite(r) and r > 5:
        ang = np.unwrap(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
        sweep = abs(ang[-1] - ang[0])
        circ_quality = float(np.clip(1 - resid / (0.25 * r), 0, 1)) * float(np.clip(sweep / math.pi, 0, 1))
    corridors = geo.get("corridors") or []
    d_corr = distance_to_polylines(s.pos[-1], corridors)[0] if corridors else math.inf
    on_corr = 0.0
    if corridors:
        on_corr = float(np.mean(polyline_distances(s.pos[:: max(1, len(s) // 8)], corridors) < 120))
    parks = geo.get("parks") or []
    in_park = 0.0
    if parks:
        p = s.pos[-1]
        in_park = float(max(1.0 - _sig(math.hypot(p[0] - pk[0], p[1] - pk[1]), pk[2] * 1.2, 40.0) for pk in parks))
    return {
        "in_park": in_park,
        "speed": float(np.mean(sp)),
        "speed_std": float(np.std(sp)),
        "hover_frac": float(np.mean(sp < 1.5)),
        "hover_recent": float(np.mean(sp[-8:] < 1.5)),
        "straightness": net / path if path > 10 else 0.0,
        "turn_total": tot_turn,
        "turn_consistency": consistency,
        "reversals": float(reversals),
        "circle_radius": float(r) if math.isfinite(r) else 1e6,
        "circle_quality": circ_quality,
        "altitude": float(np.mean(s.pos[:, 2])),
        "vz": float(np.mean(s.vel[-10:, 2])),
        "corridor_distance": float(d_corr),
        "on_corridor": on_corr,
        "duration": float(s.t[-1] - s.t[0]),
    }


class IntentInference:
    def __init__(self, cfg: dict[str, Any], geography: Optional[dict] = None) -> None:
        self.window = float(cfg.get("window", 90))
        stay = float(cfg.get("stickiness", 0.97))
        self.T = np.full((N_INT, N_INT), (1 - stay) / (N_INT - 1)) + np.eye(N_INT) * (stay - (1 - stay) / (N_INT - 1))
        self.geo = geography or {}

    def loglik(self, f: dict[str, float], cp: dict[str, float], anomaly: float, priority: float) -> np.ndarray:
        drone = cp.get("delivery_drone", 0) + cp.get("recreational_drone", 0) + cp.get("industrial_drone", 0)
        manned = cp.get("helicopter", 0) + cp.get("commercial_aircraft", 0) + cp.get("emergency_vehicle", 0)
        orbit_small = f["circle_quality"] * _sig(f["turn_total"], 3.0, 0.8) * (1 - _sig(f["circle_radius"], 200, 40))
        orbit_large = f["circle_quality"] * _sig(f["turn_total"], 2.5, 0.8) * _sig(f["circle_radius"], 200, 40)
        racetrack = _sig(f["turn_total"], 4.0, 1.0) * f["turn_consistency"] * (1 - f["straightness"])
        moving = _sig(f["speed"], 2.5, 0.8)
        L = np.zeros(N_INT)
        eps = 1e-3

        def put(intent: Intent, p: float) -> None:
            L[IDX[intent]] = math.log(max(p, eps))

        put(Intent.DELIVERY_ROUTE, (0.15 + 0.85 * cp.get("delivery_drone", 0)) * (0.2 + 0.8 * f["on_corridor"]) * (0.3 + 0.7 * max(moving, f["hover_recent"])))
        put(Intent.TRANSIT, moving * _sig(f["straightness"], 0.8, 0.07) * (1 - 0.7 * f["on_corridor"] * cp.get("delivery_drone", 0))
            * (1 - 0.6 * cp.get("emergency_vehicle", 0)) * (1 - 0.6 * cp.get("recreational_drone", 0) * (0.4 + 0.6 * f["in_park"])))
        put(Intent.INSPECTION, (0.2 + 0.8 * drone) * max(orbit_small, 0.02) * moving)
        put(Intent.MAPPING, (0.2 + 0.8 * drone) * _sig(f["reversals"], 1.5, 0.4) * (1 - f["turn_consistency"] * 0.7) * moving)
        put(Intent.RECREATIONAL, (0.1 + 0.9 * cp.get("recreational_drone", 0)) * (0.25 + 0.75 * f["in_park"])
            * (1 - 0.6 * _sig(f["straightness"], 0.75, 0.1)) * (0.5 + 0.5 * _sig(f["speed_std"], 1.5, 0.7)) * (1 - f["circle_quality"] * 0.6))
        put(Intent.HOLDING_PATTERN, (0.2 + 0.8 * manned) * (1 - 0.7 * cp.get("emergency_vehicle", 0)) * max(orbit_large, racetrack) * _sig(f["speed"], 15, 4))
        put(Intent.EMERGENCY_RESPONSE, max(cp.get("emergency_vehicle", 0), priority) * (0.4 + 0.6 * _sig(f["speed"], 40, 8)) + 0.002)
        put(Intent.HOVERING, _sig(f["hover_recent"], 0.6, 0.1) * (0.3 + 0.7 * (drone + cp.get("helicopter", 0))))
        put(Intent.DRIFTING, (0.05 + 0.95 * cp.get("weather_balloon", 0)) * _sig(f["vz"], 2.0, 0.6) * (1 - _sig(f["speed"], 25, 5)))
        put(Intent.ANOMALOUS, 0.02 + 0.98 * _sig(anomaly, 0.6, 0.08))
        return L

    def update(self, track, now: float, class_probs: dict[str, float], anomaly: float) -> Optional[dict]:
        # Faster objects fly larger patterns: scale the behaviour window with speed so a
        # several-minute airliner holding circuit fits (90 s for drones, up to 300 s).
        speed = float(np.linalg.norm(track.velocity[:2]))
        window = float(np.clip(60.0 + 3.0 * speed, self.window, 300.0))
        f = behaviour_features(track, now, window, self.geo)
        if f is None:
            return None
        prio = float(np.mean(track.feature_values("priority_status"))) if track.feature_values("priority_status") else 0.0
        L = self.loglik(f, class_probs, anomaly, prio)
        st = track.analytics.get("_intent")
        prior = np.full(N_INT, 1.0 / N_INT) if st is None else self.T.T @ st
        # Tempered likelihood: features from overlapping windows are correlated.
        post = prior * np.exp(0.6 * (L - L.max()))
        post /= post.sum()
        track.analytics["_intent"] = post
        best = int(np.argmax(post))
        return {"label": INTENTS[best].value, "probs": {it.value: float(p) for it, p in zip(INTENTS, post)},
                "confidence": float(post[best]), "features": {k: round(v, 3) for k, v in f.items()}}
