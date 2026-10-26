"""Simulated sensor models with realistic noise, detection and latency behaviour.

Each model turns world ground truth into ``Observation`` objects exactly as a
real driver would after normalisation; nothing downstream can tell the two
apart (apart from the evaluation-only ``truth_id``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.types import Observation, ObjectClass, SensorType
from .base import enu_to_polar, polar_covariance, polar_to_enu


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class SimSensor:
    sensor_type: SensorType

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.id = cfg["id"]
        self.position = np.asarray(cfg.get("position", [0.0, 0.0, 0.0]), dtype=float)
        self.rate = float(cfg.get("rate_hz", 1.0))
        self.range = float(cfg.get("range", 5000.0))
        self.latency = cfg.get("latency", [0.05, 0.02])  # mean, jitter (s)
        self.enabled = bool(cfg.get("enabled", True))
        self.next_scan = float(cfg.get("phase", 0.0))
        self.degraded = 1.0  # noise multiplier (weather / interference)

    def due(self, t: float) -> bool:
        return self.enabled and t + 1e-9 >= self.next_scan

    def scan(self, world, t: float, rng: np.random.Generator) -> list[tuple[float, Observation]]:
        self.next_scan += 1.0 / self.rate
        out = []
        for agent in world.agents.values():
            obs = self.observe(agent, t, rng)
            if obs is not None:
                out.append(obs)
        out.extend(self.clutter(world, t, rng))
        delivered = []
        for o in out:
            lat = max(0.0, rng.normal(self.latency[0], self.latency[1]))
            delivered.append((t + lat, o))
        return delivered

    def observe(self, agent, t, rng):  # pragma: no cover - interface
        raise NotImplementedError

    def clutter(self, world, t, rng) -> list[Observation]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.sensor_type.value, "position": self.position.tolist(), "range": self.range,
                "rate_hz": self.rate, "fov": self.cfg.get("fov_deg"), "heading": self.cfg.get("heading_deg")}


class RadarSensor(SimSensor):
    """Rotating surveillance radar with RCS-dependent detection, clutter and micro-Doppler cues."""

    sensor_type = SensorType.RADAR

    def snr_db(self, rcs: float, r: float) -> float:
        r0 = float(self.cfg.get("reference_range", 20000.0))
        return rcs + 40 * math.log10(r0 / max(r, 1.0))

    def observe(self, agent, t, rng):
        r, az, el = enu_to_polar(self.position, agent.pos)
        if r > self.range or r < 30:
            return None
        rcs = agent.profile["rcs"] + agent.rcs_offset + rng.normal(0, 2.5)
        snr = self.snr_db(rcs, r)
        # Low-altitude targets at long range suffer from ground clutter / multipath.
        if agent.pos[2] < 15 and r > 1500:
            snr -= 8
        pd = _logistic((snr - 11.0) / 1.6)
        if rng.random() > pd:
            return None
        s_r = self.cfg.get("sigma_range", 8.0) * self.degraded
        s_az = math.radians(self.cfg.get("sigma_az_deg", 0.25)) * self.degraded
        s_el = math.radians(self.cfg.get("sigma_el_deg", 0.6)) * self.degraded
        mr, maz, mel = r + rng.normal(0, s_r), az + rng.normal(0, s_az), el + rng.normal(0, s_el)
        pos = polar_to_enu(self.position, mr, maz, mel)
        cov = polar_covariance(mr, maz, mel, s_r, s_az, s_el)
        # Micro-Doppler cue quality degrades with SNR.
        q = float(np.clip((snr - 8) / 20, 0.05, 1.0))
        noise = 0.35 * (1.2 - q)
        feats = {
            "rcs_dbsm": rcs,
            "snr_db": snr,
            "rotor_score": float(np.clip(agent.profile["rotor"] + rng.normal(0, noise), 0, 1)),
            "flap_score": float(np.clip(agent.profile["flap"] + rng.normal(0, noise), 0, 1)),
            "radial_velocity": float(agent.vel @ ((agent.pos - self.position) / max(r, 1e-6)) + rng.normal(0, 0.5)),
        }
        return Observation(self.id, self.sensor_type, t, pos, cov, features=feats, quality=pd, truth_id=agent.id)

    def clutter(self, world, t, rng):
        lam = float(self.cfg.get("false_alarms_per_scan", 1.0))
        out = []
        for _ in range(rng.poisson(lam)):
            r = self.range * math.sqrt(rng.random())
            az = rng.uniform(-math.pi, math.pi)
            el = math.atan2(rng.uniform(5, 400), max(r, 1.0))
            s_r, s_az, s_el = self.cfg.get("sigma_range", 8.0), math.radians(self.cfg.get("sigma_az_deg", 0.25)), math.radians(self.cfg.get("sigma_el_deg", 0.6))
            pos = polar_to_enu(self.position, r, az, el)
            feats = {"rcs_dbsm": rng.uniform(-30, -5), "snr_db": rng.uniform(10, 14), "rotor_score": rng.uniform(0, 1),
                     "flap_score": rng.uniform(0, 1), "radial_velocity": rng.normal(0, 3)}
            out.append(Observation(self.id, self.sensor_type, t, pos, polar_covariance(r, az, el, s_r, s_az, s_el), features=feats, quality=0.5))
        return out


class LidarSensor(SimSensor):
    """Short-range, high-accuracy 3D LiDAR."""

    sensor_type = SensorType.LIDAR

    def observe(self, agent, t, rng):
        d = agent.pos - self.position
        r = float(np.linalg.norm(d))
        if r > self.range:
            return None
        size = agent.profile["size"]
        pd = float(np.clip(0.98 - (r / self.range) ** 2 * (0.6 / max(size, 0.3)), 0.2, 0.98))
        if rng.random() > pd:
            return None
        s = self.cfg.get("sigma", 0.4) * (1 + r / self.range) * self.degraded
        pos = agent.pos + rng.normal(0, s, 3)
        feats = {"size_m": float(max(0.05, size * (1 + rng.normal(0, 0.15))))}
        return Observation(self.id, self.sensor_type, t, pos, np.eye(3) * s**2, features=feats, quality=pd, truth_id=agent.id)


class CameraSensor(SimSensor):
    """EO/IR camera: precise bearing, poor depth (monocular range from apparent size)."""

    sensor_type = SensorType.CAMERA

    def observe(self, agent, t, rng):
        r, az, el = enu_to_polar(self.position, agent.pos)
        if r > self.range or r < 5:
            return None
        hd = math.radians(self.cfg.get("heading_deg", 0.0))
        fov = math.radians(self.cfg.get("fov_deg", 90.0))
        if abs((az - hd + math.pi) % (2 * math.pi) - math.pi) > fov / 2:
            return None
        pd = float(np.clip(0.95 - 0.6 * r / self.range, 0.2, 0.95))
        if rng.random() > pd:
            return None
        s_ang = math.radians(self.cfg.get("sigma_deg", 0.05)) * self.degraded
        s_r = r * self.cfg.get("range_rel_sigma", 0.12)
        mr, maz, mel = r * (1 + rng.normal(0, self.cfg.get("range_rel_sigma", 0.12))), az + rng.normal(0, s_ang), el + rng.normal(0, s_ang)
        pos = polar_to_enu(self.position, mr, maz, mel)
        cov = polar_covariance(mr, maz, mel, s_r, s_ang, s_ang)
        feats = {"size_m": float(max(0.05, agent.profile["size"] * (1 + rng.normal(0, 0.25))))}
        return Observation(self.id, self.sensor_type, t, pos, cov, features=feats, quality=pd, truth_id=agent.id)


class AdsbReceiver(SimSensor):
    """Cooperative surveillance: transponder broadcasts position, velocity and identity."""

    sensor_type = SensorType.ADSB

    def observe(self, agent, t, rng):
        if not agent.adsb_id or not agent.transponder_on:
            return None
        if np.linalg.norm(agent.pos[:2] - self.position[:2]) > self.range:
            return None
        if rng.random() < self.cfg.get("message_loss", 0.05):
            return None
        sh, sv = self.cfg.get("sigma_h", 8.0), self.cfg.get("sigma_v", 7.6)
        pos = agent.pos + rng.normal(0, 1, 3) * np.array([sh, sh, sv])
        pos[2] = round(pos[2] / 7.62) * 7.62  # 25 ft altitude quantisation
        vel = agent.vel + rng.normal(0, 0.8, 3)
        cat = 7.0 if agent.cls in (ObjectClass.HELICOPTER, ObjectClass.EMERGENCY_VEHICLE) else 3.0
        feats = {"emitter_category": cat, "priority_status": float(agent.priority_status)}
        return Observation(self.id, self.sensor_type, t, pos, np.diag([sh**2, sh**2, sv**2]), velocity=vel,
                           velocity_covariance=np.eye(3) * 0.8**2, identity=f"ICAO:{agent.adsb_id}", features=feats,
                           quality=0.97, truth_id=agent.id)


class RemoteIdReceiver(SimSensor):
    """Drone Remote-ID / GPS telemetry broadcast receiver."""

    sensor_type = SensorType.REMOTE_ID

    def observe(self, agent, t, rng):
        if not agent.rid_id or not agent.transponder_on:
            return None
        if np.linalg.norm(agent.pos - self.position) > self.range:
            return None
        if rng.random() < self.cfg.get("message_loss", 0.1):
            return None
        sh, sv = self.cfg.get("sigma_h", 4.0), self.cfg.get("sigma_v", 3.0)
        pos = agent.pos + rng.normal(0, 1, 3) * np.array([sh, sh, sv])
        vel = agent.vel + rng.normal(0, 0.4, 3)
        return Observation(self.id, self.sensor_type, t, pos, np.diag([sh**2, sh**2, sv**2]), velocity=vel,
                           velocity_covariance=np.eye(3) * 0.4**2, identity=agent.rid_id, features={"remote_id": 1.0},
                           quality=0.95, truth_id=agent.id)


SENSOR_MODELS = {
    "radar": RadarSensor,
    "lidar": LidarSensor,
    "camera": CameraSensor,
    "adsb": AdsbReceiver,
    "remote_id": RemoteIdReceiver,
}


def build_sensors(cfgs: list[dict[str, Any]]) -> list[SimSensor]:
    return [SENSOR_MODELS[c["type"]](c) for c in cfgs]
