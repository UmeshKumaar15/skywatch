"""Simulation engine: advances the world and runs simulated sensors.

Produces observation batches in *delivery* order (sensor time + comms
latency), so the perception pipeline sees the same timing artefacts it would
see from real, networked sensors.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Any

import numpy as np

from ..core.types import Observation
from ..sensors.simulated import SimSensor, build_sensors
from .world import World


class SimulationEngine:
    def __init__(self, scenario: dict[str, Any], sensor_cfgs: list[dict[str, Any]], dt: float = 0.1) -> None:
        self.scenario = scenario
        self.world = World(scenario)
        self.dt = float(scenario.get("dt", dt))
        self.sensors: list[SimSensor] = build_sensors([self._resolve_anchor(c) for c in scenario.get("sensors", sensor_cfgs)])
        # Separate RNG stream for sensor noise so changing sensors does not
        # change the trajectories of the agents.
        self.sensor_rng = np.random.default_rng(self.world.seed + 7919)
        self._pending: list = []
        self._seq = itertools.count()
        self.duration = float(scenario.get("duration", 1800))
        self._apply_events_cfg()

    def _resolve_anchor(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Allow sensors to be placed relative to generated geography, e.g. ``anchor: depot0``."""
        cfg = dict(cfg)
        anchor = cfg.get("anchor")
        if anchor:
            geo = self.world.geo
            name = anchor.rstrip("0123456789")
            idx = int(anchor[len(name):] or 0)
            table = {"depot": geo.depots, "park": geo.parks, "inspection": geo.inspection_sites, "destination": geo.destinations}
            if name in table:
                xy = table[name][idx % len(table[name])][:2]
            else:
                xy = getattr(geo, name)[:2]
            h = cfg.get("position", [0, 0, 10])[2] if cfg.get("position") else cfg.get("height", 10.0)
            cfg["position"] = [float(xy[0]), float(xy[1]), float(h)]
        return cfg

    def _apply_events_cfg(self) -> None:
        """Scripted anomalies: [{t: 120, kind: erratic, target: delivery_drone}]."""
        self._scripted = sorted(self.scenario.get("events", []), key=lambda e: e["t"])

    @property
    def t(self) -> float:
        return self.world.t

    def _run_scripted(self) -> None:
        while self._scripted and self._scripted[0]["t"] <= self.world.t:
            ev = self._scripted.pop(0)
            kind = ev["kind"]
            if kind == "sensor_outage":
                for s in self.sensors:
                    if s.id == ev.get("sensor"):
                        s.enabled = False
                        self._scripted.append({"t": ev["t"] + ev.get("duration", 30), "kind": "sensor_restore", "sensor": s.id})
                        self._scripted.sort(key=lambda e: e["t"])
            elif kind == "sensor_restore":
                for s in self.sensors:
                    if s.id == ev.get("sensor"):
                        s.enabled = True
                        s.next_scan = max(s.next_scan, self.world.t)
            elif kind == "sensor_degrade":
                for s in self.sensors:
                    if s.id == ev.get("sensor"):
                        s.degraded = float(ev.get("factor", 2.0))
            else:
                cands = [a for a in self.world.agents.values() if a.cls.value == ev.get("target", a.cls.value) and not a.anomalies]
                if cands:
                    self.world.inject_anomaly(cands[0], kind, self.world.t)

    def step(self) -> list[list[Observation]]:
        """Advance one tick; return observation batches delivered during it."""
        self.world.step(self.dt)
        self._run_scripted()
        t = self.world.t
        for s in self.sensors:
            while s.due(t):
                scan_t = s.next_scan
                for deliver_t, obs in s.scan(self.world, scan_t, self.sensor_rng):
                    heapq.heappush(self._pending, (deliver_t, next(self._seq), s.id, obs))
        # Group deliveries by sensor for this tick
        batches: dict[tuple, list[Observation]] = {}
        while self._pending and self._pending[0][0] <= t:
            deliver_t, _, sid, obs = heapq.heappop(self._pending)
            obs.received_at = deliver_t
            batches.setdefault((sid, round(obs.timestamp, 3)), []).append(obs)
        return [batches[k] for k in sorted(batches, key=lambda k: k[1])]

    @property
    def finished(self) -> bool:
        return self.world.t >= self.duration

    def describe(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.get("name", "custom"),
            "seed": self.world.seed,
            "radius": self.world.radius,
            "geography": self.world.geo.to_dict(),
            "sensors": [s.describe() for s in self.sensors],
            "wind": self.world.wind.tolist(),
        }
