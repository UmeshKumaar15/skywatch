"""Behavioural anomaly detection.

Combines
  * class-conditional statistical baselines (online Welford mean/variance
    with informative priors) -> robust z-scores of kinematic features,
  * a learned spatial flow map of normal traffic,
  * contextual checks (hover duration, cooperative-signal silence, corridor
    adherence, restricted areas).

Each detector yields a graded score in [0, 1]. Scores above a threshold are
surfaced as *insights* with human-readable evidence, rate-limited per track
and kind, instead of hard alarms.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional

import numpy as np

from ..core.types import Insight
from ..tracking.history import distance_to_polylines, resample, wrap_angle
from .flowmap import FlowMap

# Prior (mean, std) per class for: |vz| (m/s), |a| (m/s2), max turn rate (deg/s), mean NIS
PRIORS = {
    "delivery_drone": [(1.2, 1.4), (1.0, 1.0), (12, 12), (3, 3)],
    "recreational_drone": [(1.2, 1.5), (1.8, 1.6), (25, 20), (3, 3)],
    "industrial_drone": [(1.0, 1.2), (1.0, 1.0), (14, 12), (3, 3)],
    "bird": [(0.9, 0.9), (1.5, 1.2), (25, 20), (3, 3)],
    "helicopter": [(1.5, 2.0), (0.8, 0.8), (5, 5), (3, 3)],
    "commercial_aircraft": [(4.0, 4.0), (0.8, 0.8), (2, 2), (3, 3)],
    "emergency_vehicle": [(2.0, 2.5), (1.0, 1.0), (8, 8), (3, 3)],
    "weather_balloon": [(5.0, 1.5), (0.8, 0.8), (10, 12), (3, 3)],
    "unknown": [(1.5, 2.0), (1.8, 1.6), (25, 20), (3, 3)],
}
FEATS = ("vz", "acc", "turn", "nis")


class Baselines:
    """Online class-conditional Gaussian statistics with prior pseudo-counts.

    Priors come from ``models/baselines.json`` (calibrated on simulated
    normal traffic by ``skywatch train``) when available, else from the
    hand-set defaults above.
    """

    def __init__(self, n0: float = 200.0, priors: Optional[dict] = None) -> None:
        self.stats: dict[str, np.ndarray] = {}
        # Heavy-tail floor for the scale: kinematic features are far from Gaussian
        # (e.g. a delivery drone flies straight most of the time but legitimately
        # turns sharply at corridor waypoints), so the z-score scale is never
        # allowed below (p99.5 - mean) / 2.58 measured on normal traffic.
        self.tail: dict[str, np.ndarray] = {}
        for cls, pri in (priors or PRIORS).items():
            arr = np.zeros((len(FEATS), 3))
            tail = np.zeros(len(FEATS))
            for i, entry in enumerate(pri):
                m, s = entry[0], entry[1]
                arr[i] = (n0, m, n0 * s * s)  # count, mean, M2
                tail[i] = entry[2] if len(entry) > 2 else s
            self.stats[cls] = arr
            self.tail[cls] = tail

    def z(self, cls: str, i: int, x: float) -> float:
        key = cls if cls in self.stats else "unknown"
        n, m, M2 = self.stats[key][i]
        sd = max(math.sqrt(M2 / max(n - 1, 1)), self.tail[key][i])
        return (x - m) / max(sd, 1e-3)

    def update(self, cls: str, values: list[float]) -> None:
        arr = self.stats.get(cls)
        if arr is None:
            return
        for i, x in enumerate(values):
            n, m, M2 = arr[i]
            n += 1
            d = x - m
            m += d / n
            M2 += d * (x - m)
            arr[i] = (min(n, 20000), m, M2 * (min(n, 20000) / n))

    def describe(self) -> dict[str, dict[str, list[float]]]:
        out = {}
        for cls, arr in self.stats.items():
            out[cls] = {f: [round(float(arr[i, 1]), 2), round(max(math.sqrt(arr[i, 2] / max(arr[i, 0] - 1, 1)), float(self.tail[cls][i])), 2)] for i, f in enumerate(FEATS)}
        return out


def kinematic_features(track, now: float, s=None) -> Optional[tuple[float, float, float, float, float]]:
    """(|vz| mean 5 s, altitude change 15 s, |accel| mean 3 s, max turn rate deg/s over 2 s spans, mean NIS)."""
    if s is None:
        s = resample(track.history, now, 30.0, dt=1.0)
    if len(s) < 6:
        return None
    vz = float(np.mean(np.abs(s.vel[-5:, 2])))
    dz = float(s.pos[-1, 2] - s.pos[max(0, len(s) - 15), 2])
    dv = np.diff(s.vel, axis=0)
    acc = float(np.mean(np.linalg.norm(dv[-3:], axis=1)))
    h = s.heading[-9:]
    spd = s.speed[-9:]
    ok = (spd[2:] > 3.0) & (spd[:-2] > 3.0)
    dh = np.abs(wrap_angle(h[2:] - h[:-2])) / 2.0
    turn = float(np.max(dh[ok])) * 180 / math.pi if ok.any() else 0.0
    nis = float(np.mean(list(track.nis)[-6:])) if track.nis else 3.0
    return vz, dz, acc, turn, min(nis, 60.0)


def _sig(x: float, c: float, w: float) -> float:
    z = max(min((x - c) / w, 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-z))


KIND_TEXT = {
    "altitude_change": "Unexpected altitude change",
    "excessive_acceleration": "Excessive acceleration",
    "abrupt_maneuver": "Abrupt manoeuvring",
    "prolonged_hover": "Prolonged hovering",
    "comms_loss": "Cooperative signal lost",
    "signal_loss": "Sensor contact lost",
    "route_deviation": "Deviation from operational corridor",
    "unusual_path": "Unusual flight path",
    "restricted_area": "Restricted area entry",
}


class AnomalyDetector:
    def __init__(self, cfg: dict[str, Any], geography: Optional[dict] = None, radius: float = 6000.0) -> None:
        self.cfg = cfg
        self.threshold = float(cfg.get("emit_threshold", 0.6))
        self.cooldown = float(cfg.get("cooldown", 45))
        self.hover_limits = dict(cfg.get("hover_limits", {}))
        self.comms_after = float(cfg.get("comms_loss_after", 8))
        self.geo = geography or {}
        self.min_age = float(cfg.get("min_track_age", 15.0))
        self.radius = radius
        self.kind_cooldown = {"restricted_area": 600.0, "comms_loss": 300.0, "prolonged_hover": 300.0, "signal_loss": 120.0,
                              **cfg.get("kind_cooldown", {})}
        priors = None
        bp = cfg.get("baselines_path")
        if bp:
            from ..config import resolve_path
            import json
            path = resolve_path(bp)
            if path.exists():
                priors = {k: [tuple(x) for x in v] for k, v in json.loads(path.read_text()).items()}
                for k, v in PRIORS.items():
                    priors.setdefault(k, v)
        self.baselines = Baselines(priors=priors)
        self.flowmap = FlowMap(radius, float(cfg.get("flowmap_cell", 400)))

    def update(self, track, now: float, cls_label: str, cls_probs: dict[str, float]) -> tuple[dict, list[Insight]]:
        s = resample(track.history, now, 30.0, dt=1.0)
        scores: dict[str, float] = {}
        ev: dict[str, dict[str, float]] = {}
        state = track.state
        pos, vel = state[0:3], state[3:6]
        speed = float(np.linalg.norm(vel[:2]))
        cls = cls_label if cls_label in PRIORS else "unknown"
        feat_vals = None

        settled = now - track.created >= self.min_age and track.n_updates >= 8 and len(s) >= 6
        kin = kinematic_features(track, now, s) if settled else None
        if kin is not None:
            vz, dz, acc, turn, nis = kin
            feat_vals = [vz, acc, turn, nis]
            z = [self.baselines.z(cls, i, v) for i, v in enumerate(feat_vals)]
            # Kinematic anomalies concern *operated* aircraft: erratic flight is natural for
            # birds and passive drift for balloons, so their scores are discounted.
            operated = max(0.0, 1.0 - cls_probs.get("bird", 0.0) - cls_probs.get("weather_balloon", 0.0))
            scores["altitude_change"] = operated * _sig(z[0], 3.5, 0.7) * _sig(abs(dz), 40, 8)
            ev["altitude_change"] = {"dz_15s": dz, "vz": vz, "z": z[0]}
            scores["excessive_acceleration"] = operated * _sig(z[1], 4.0, 0.8)
            ev["excessive_acceleration"] = {"accel": acc, "z": z[1]}
            scores["abrupt_maneuver"] = operated * max(_sig(z[2], 3.5, 0.8) * _sig(speed, 4, 1), _sig(z[3], 6.0, 1.5) * 0.9)
            ev["abrupt_maneuver"] = {"turn_rate_deg_s": turn, "nis": nis, "z_turn": z[2], "z_nis": z[3]}

            # Prolonged hover: how long the object has stayed within 25 m of its current
            # position (robust to gust-induced velocity noise).
            long = resample(track.history, now, 400.0, dt=1.0)
            hov = 0.0
            if len(long) > 1:
                d = np.linalg.norm(long.pos[:, :2] - long.pos[-1, :2], axis=1)[::-1]
                far = np.nonzero(d > 25.0)[0]
                hov = float(far[0] if len(far) else len(d)) * 1.0
            limit = self.hover_limits.get(cls)
            if limit:
                scores["prolonged_hover"] = _sig(hov, limit * 1.15, limit * 0.12)
                ev["prolonged_hover"] = {"hover_s": hov, "typical_max_s": limit}

        # Cooperative signal silence while still observed by other sensors.
        coop_n = track.sources.get("adsb", 0) + track.sources.get("remote_id", 0)
        if track.last_coop_time is not None and coop_n >= 5:
            silence = now - track.last_coop_time
            still_seen = now - track.last_update < 4.0
            scores["comms_loss"] = _sig(silence, self.comms_after * 1.5, 2.0) * (1.0 if still_seen else 0.5)
            ev["comms_loss"] = {"silence_s": silence}
        # Unexpected loss of all sensor contact (not near the ground, not at the zone edge,
        # not an intermittently-detected bird, and only for tracks with a steady history).
        expected = pos[2] < 30 or float(np.linalg.norm(pos[:2])) > self.radius * 0.9 or cls == "bird"
        if not expected and track.n_updates > 30:
            scores["signal_loss"] = _sig(now - track.last_update, 7.0, 1.0) * 0.8
            ev["signal_loss"] = {"since_update_s": now - track.last_update}

        corridors = self.geo.get("corridors") or []
        p_del = cls_probs.get("delivery_drone", 0.0)
        if corridors and cls == "delivery_drone" and p_del > 0.6 and track.n_updates > 20:
            d = distance_to_polylines(pos, corridors)[0]
            scores["route_deviation"] = p_del * _sig(d, 320, 60)
            ev["route_deviation"] = {"corridor_distance_m": d}

        fm, fev = self.flowmap.surprise(cls, pos, vel) if cls != "unknown" else (0.0, {})
        if fev and "learning" not in fev:
            # Require persistence: a single unusual sample is not an unusual path.
            hist = track.analytics.setdefault("_fm_hist", deque(maxlen=10))
            hist.append(fm)
            scores["unusual_path"] = 0.85 * float(np.mean(hist)) * min(1.0, len(hist) / 6.0)
            ev["unusual_path"] = fev

        drone_p = cls_probs.get("delivery_drone", 0) + cls_probs.get("recreational_drone", 0) + cls_probs.get("industrial_drone", 0)
        for z_ in self.geo.get("nofly") or []:
            if math.hypot(pos[0] - z_[0], pos[1] - z_[1]) < z_[2] and pos[2] < 400:
                scores["restricted_area"] = 0.95 * drone_p
                ev["restricted_area"] = {"zone_radius_m": z_[2]}
                ev["restricted_area_name"] = z_[3] if len(z_) > 3 else "zone"

        active = {k: v for k, v in scores.items() if v > 0.3}
        overall = 1.0 - float(np.prod([1 - v for v in active.values()])) if active else 0.0
        # Learn normal behaviour only from tracks that currently look normal.
        if overall < 0.3 and track.n_updates > 10:
            if feat_vals is not None and now - track.created > 20:
                self.baselines.update(cls, feat_vals)
            self.flowmap.update(cls, pos, vel)

        insights = []
        st = track.analytics.setdefault("_anom_emitted", {})
        for kind, sc in scores.items():
            if sc >= self.threshold and now - st.get(kind, -1e9) >= self.kind_cooldown.get(kind, self.cooldown):
                st[kind] = now
                insights.append(Insight(track.id, kind, sc, now, self._message(kind, ev.get(kind, {}), cls, ev), ev.get(kind, {})))
        result = {
            "score": overall,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "top": max(scores, key=scores.get) if scores else None,
        }
        return result, insights

    def _message(self, kind: str, e: dict, cls: str, ev_all: dict) -> str:
        nice = cls.replace("_", " ")
        if kind == "altitude_change":
            return f"Unexpected altitude change of {e.get('dz_15s', 0):+.0f} m in 15 s ({e.get('vz', 0):.1f} m/s, {e.get('z', 0):.1f} sd above {nice} norm)"
        if kind == "excessive_acceleration":
            return f"Acceleration {e.get('accel', 0):.1f} m/s2 is {e.get('z', 0):.1f} sd above typical {nice} behaviour"
        if kind == "abrupt_maneuver":
            return f"Abrupt manoeuvre: turn rate {e.get('turn_rate_deg_s', 0):.0f} deg/s, motion-model surprise (NIS) {e.get('nis', 0):.1f}"
        if kind == "prolonged_hover":
            return f"Hovering for {e.get('hover_s', 0):.0f} s (typical maximum for {nice}: {e.get('typical_max_s', 0):.0f} s)"
        if kind == "comms_loss":
            return f"No cooperative (ADS-B / Remote ID) message for {e.get('silence_s', 0):.0f} s while still tracked by sensors"
        if kind == "signal_loss":
            return f"No sensor contact for {e.get('since_update_s', 0):.0f} s - position is extrapolated"
        if kind == "route_deviation":
            return f"Likely delivery drone {e.get('corridor_distance_m', 0):.0f} m away from the nearest operational corridor"
        if kind == "unusual_path":
            return f"Motion rarely observed for {nice} traffic in this area (surprise {e.get('direction_surprise', 0):.1f} nats)"
        if kind == "restricted_area":
            return f"Drone-like object inside restricted area '{ev_all.get('restricted_area_name', 'zone')}'"
        return KIND_TEXT.get(kind, kind)
