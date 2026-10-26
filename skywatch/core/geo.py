"""Geodetic <-> local ENU conversions (WGS-84) used by real sensor adapters."""
from __future__ import annotations

import math

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_m) * math.cos(lat) * math.cos(lon),
            (n + alt_m) * math.cos(lat) * math.sin(lon),
            (n * (1 - _E2) + alt_m) * math.sin(lat),
        ]
    )


class LocalFrame:
    """East-North-Up tangent plane anchored at a geodetic origin."""

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0) -> None:
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        self._ecef0 = geodetic_to_ecef(lat0, lon0, alt0)
        lat, lon = math.radians(lat0), math.radians(lon0)
        self._R = np.array(
            [
                [-math.sin(lon), math.cos(lon), 0.0],
                [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
                [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
            ]
        )

    def to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        return self._R @ (geodetic_to_ecef(lat, lon, alt) - self._ecef0)

    def to_geodetic(self, enu: np.ndarray) -> tuple[float, float, float]:
        ecef = self._R.T @ np.asarray(enu, dtype=float) + self._ecef0
        x, y, z = ecef
        lon = math.atan2(y, x)
        p = math.hypot(x, y)
        lat = math.atan2(z, p * (1 - _E2))
        alt = 0.0
        for _ in range(6):
            n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
            alt = p / math.cos(lat) - n
            lat = math.atan2(z, p * (1 - _E2 * n / (n + alt)))
        return math.degrees(lat), math.degrees(lon), alt

    def velocity_from_ground_track(self, ground_speed: float, track_deg: float, vrate: float) -> np.ndarray:
        """ADS-B style speed/track/vertical rate -> ENU velocity (m/s)."""
        tr = math.radians(track_deg)
        return np.array([ground_speed * math.sin(tr), ground_speed * math.cos(tr), vrate])
