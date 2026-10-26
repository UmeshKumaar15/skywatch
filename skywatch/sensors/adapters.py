"""Adapters for real (non-simulated) observation sources.

All adapters emit the same normalised ``Observation`` batches as the
simulator, so the perception pipeline is unchanged when moving to hardware.

* ``SbsAdsbAdapter``  - ADS-B via the SBS-1 / BaseStation text protocol
                        (dump1090 / readsb ``--net`` port 30003).
* ``JsonlReplayAdapter`` - replays recorded observation JSONL files
                        (``python -m skywatch simulate --jsonl ...``).
* HTTP ingest           - ``POST /api/ingest`` (see server/app.py) accepts
                        observations in the external JSON schema from any
                        radar / LiDAR / camera gateway process.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

import numpy as np

from ..core.geo import LocalFrame
from ..core.types import Observation, SensorType
from .base import SensorSource

FT = 0.3048
KT = 0.514444


class SbsAdsbAdapter(SensorSource):
    """Streams ADS-B positions from a BaseStation (SBS-1) TCP feed."""

    def __init__(self, host: str, port: int, frame: LocalFrame, sensor_id: str = "adsb-sbs",
                 sigma_h: float = 15.0, sigma_v: float = 8.0, max_range: float = 60000.0) -> None:
        self.name = sensor_id
        self.host, self.port = host, port
        self.frame = frame
        self.sigma_h, self.sigma_v = sigma_h, sigma_v
        self.max_range = max_range
        self._vel: dict[str, tuple[float, np.ndarray]] = {}
        self._call: dict[str, str] = {}

    def parse_line(self, line: str, now: Optional[float] = None) -> Optional[Observation]:
        f = line.strip().split(",")
        if len(f) < 22 or f[0] != "MSG":
            return None
        now = time.time() if now is None else now
        icao = f[4].strip().upper()
        mtype = f[1].strip()
        try:
            if f[10].strip():
                self._call[icao] = f[10].strip()
            if mtype == "4" and f[12] and f[13]:
                gs, trk = float(f[12]) * KT, float(f[13])
                vr = float(f[16]) * FT / 60.0 if f[16] else 0.0
                self._vel[icao] = (now, self.frame.velocity_from_ground_track(gs, trk, vr))
                return None
            if mtype != "3" or not (f[11] and f[14] and f[15]):
                return None
            alt = float(f[11]) * FT
            lat, lon = float(f[14]), float(f[15])
        except ValueError:
            return None
        pos = self.frame.to_enu(lat, lon, alt)
        if np.linalg.norm(pos[:2]) > self.max_range:
            return None
        vel = vcov = None
        if icao in self._vel and now - self._vel[icao][0] < 5.0:
            vel = self._vel[icao][1]
            vcov = np.eye(3) * 2.0**2
        squawk = f[17].strip() if len(f) > 17 else ""
        emergency = 1.0 if (len(f) > 19 and f[19].strip() in ("1", "-1")) or squawk in ("7700",) else 0.0
        feats = {"priority_status": emergency}
        return Observation(self.name, SensorType.ADSB, now, pos, np.diag([self.sigma_h**2, self.sigma_h**2, self.sigma_v**2]),
                           velocity=vel, velocity_covariance=vcov, identity=f"ICAO:{icao}", features=feats, quality=0.95)

    async def stream(self) -> AsyncIterator[list[Observation]]:
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.host, self.port)
            except OSError:
                await asyncio.sleep(3.0)
                continue
            try:
                while True:
                    raw = await reader.readline()
                    if not raw:
                        break
                    obs = self.parse_line(raw.decode(errors="ignore"))
                    if obs is not None:
                        yield [obs]
            finally:
                writer.close()
            await asyncio.sleep(1.0)


class JsonlReplayAdapter(SensorSource):
    """Replays a JSONL observation log, re-timed onto the wall clock."""

    def __init__(self, path: str, speed: float = 1.0, loop: bool = False) -> None:
        self.name = f"replay:{path}"
        self.path, self.speed, self.loop = path, speed, loop

    async def stream(self) -> AsyncIterator[list[Observation]]:
        while True:
            with open(self.path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            if not records:
                return
            records.sort(key=lambda r: r["timestamp"])
            t_first = records[0]["timestamp"]
            wall0 = time.time()
            batch: list[Observation] = []
            cur_key = None
            for r in records:
                due = wall0 + (r["timestamp"] - t_first) / self.speed
                delay = due - time.time()
                key = (r.get("sensor_id"), round(r["timestamp"], 3))
                if batch and key != cur_key:
                    yield batch
                    batch = []
                if delay > 0:
                    await asyncio.sleep(delay)
                r = dict(r)
                r["timestamp"] = due
                if r.get("velocity") is None:
                    r.pop("velocity", None)
                batch.append(Observation.from_dict(r))
                cur_key = key
            if batch:
                yield batch
            if not self.loop:
                return
