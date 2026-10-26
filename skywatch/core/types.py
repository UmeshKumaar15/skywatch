"""Common data representations shared by every subsystem.

All positions are expressed in a local East-North-Up (ENU) Cartesian frame in
metres, centred on the observation-zone origin defined in the configuration.
Timestamps are seconds (float) on a single monotonic clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class SensorType(str, Enum):
    RADAR = "radar"
    LIDAR = "lidar"
    CAMERA = "camera"
    ADSB = "adsb"
    REMOTE_ID = "remote_id"  # GPS telemetry broadcast by cooperative drones
    EXTERNAL = "external"


class ObjectClass(str, Enum):
    DELIVERY_DRONE = "delivery_drone"
    RECREATIONAL_DRONE = "recreational_drone"
    INDUSTRIAL_DRONE = "industrial_drone"  # inspection / mapping / survey drones
    BIRD = "bird"
    HELICOPTER = "helicopter"
    COMMERCIAL_AIRCRAFT = "commercial_aircraft"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    WEATHER_BALLOON = "weather_balloon"
    UNKNOWN = "unknown"


OBJECT_CLASSES: list[ObjectClass] = list(ObjectClass)


class Intent(str, Enum):
    DELIVERY_ROUTE = "delivery_route"
    TRANSIT = "transit"
    INSPECTION = "inspection"
    MAPPING = "mapping"
    RECREATIONAL = "recreational"
    HOLDING_PATTERN = "holding_pattern"
    EMERGENCY_RESPONSE = "emergency_response"
    HOVERING = "hovering"
    DRIFTING = "drifting"
    ANOMALOUS = "anomalous"


INTENTS: list[Intent] = list(Intent)


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class Observation:
    """A single normalised detection from any sensor modality.

    ``position``/``covariance`` are in ENU metres. Optional fields are filled
    only by modalities that provide them (e.g. ADS-B provides velocity and a
    transponder identity; radar provides an RCS and micro-Doppler cues).
    """

    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    position: np.ndarray
    covariance: np.ndarray
    velocity: Optional[np.ndarray] = None
    velocity_covariance: Optional[np.ndarray] = None
    identity: Optional[str] = None
    features: dict[str, float] = field(default_factory=dict)
    quality: float = 1.0
    # Ground-truth object id. Only populated by the simulator and used solely
    # for evaluation / training -- the tracker never reads it.
    truth_id: Optional[str] = None
    received_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "t": round(self.timestamp, 3),
            "pos": [round(float(v), 2) for v in self.position],
            "sigma": [round(float(np.sqrt(max(self.covariance[i, i], 0))), 2) for i in range(3)],
            "quality": round(self.quality, 3),
        }
        if self.identity:
            d["identity"] = self.identity
        if self.truth_id:
            d["truth_id"] = self.truth_id
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Observation":
        """Build an observation from the external JSON schema (see README)."""
        pos = np.asarray(d["position"], dtype=float)
        if "covariance" in d:
            cov = np.asarray(d["covariance"], dtype=float).reshape(3, 3)
        else:
            s = np.asarray(d.get("sigma", [10.0, 10.0, 10.0]), dtype=float)
            cov = np.diag(s**2)
        vel = d.get("velocity")
        vcov = d.get("velocity_covariance")
        if vel is not None and vcov is None:
            vs = np.asarray(d.get("velocity_sigma", [2.0, 2.0, 2.0]), dtype=float)
            vcov = np.diag(vs**2)
        return Observation(
            sensor_id=str(d.get("sensor_id", "external")),
            sensor_type=SensorType(d.get("sensor_type", "external")),
            timestamp=float(d["timestamp"]),
            position=pos,
            covariance=cov,
            velocity=None if vel is None else np.asarray(vel, dtype=float),
            velocity_covariance=None if vcov is None else np.asarray(vcov, dtype=float).reshape(3, 3),
            identity=d.get("identity"),
            features={k: float(v) for k, v in d.get("features", {}).items()},
            quality=float(d.get("quality", 1.0)),
        )


@dataclass
class Insight:
    """An analytical observation surfaced to operators (not a hard alert)."""

    track_id: str
    kind: str
    score: float  # 0..1 anomaly strength
    timestamp: float
    message: str
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        if self.score >= 0.85:
            return "significant"
        if self.score >= 0.65:
            return "notable"
        return "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "kind": self.kind,
            "score": round(self.score, 3),
            "severity": self.severity,
            "t": round(self.timestamp, 2),
            "message": self.message,
            "evidence": {k: round(float(v), 3) for k, v in self.evidence.items()},
        }
