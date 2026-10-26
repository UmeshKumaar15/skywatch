"""Glue between observation sources and the pipeline (headless or paced in real time)."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional

from .core.bus import TOPIC_OBSERVATIONS, InMemoryBus
from .pipeline import PerceptionPipeline
from .sim.engine import SimulationEngine
from .storage.store import Store


class SimRunner:
    """Runs a simulated scenario through the full perception pipeline."""

    def __init__(self, cfg: dict[str, Any], scenario: dict[str, Any], store: Optional[Store] = None,
                 analytics: bool = True, bus: Optional[InMemoryBus] = None) -> None:
        self.cfg = cfg
        self.scenario = scenario
        self.engine = SimulationEngine(scenario, cfg["sensors"])
        self.bus = bus or InMemoryBus()
        self.description = self.engine.describe()
        self.store = store
        self.run_id = None
        if store is not None and cfg.get("storage", {}).get("record", True):
            self.run_id = store.start_run("simulation", scenario.get("name"), self.engine.world.seed, self.description)
        self.pipeline = PerceptionPipeline(cfg, self.bus, self.description["geography"], self.engine.world.radius,
                                           store=store, run_id=self.run_id,
                                           truth_provider=self.engine.world.truth_snapshot, analytics=analytics)
        self.paused = False
        self.time_scale = float(cfg.get("server", {}).get("time_scale", 1.0))
        # The learned flow map of normal traffic persists across sessions per (scenario, seed).
        fm_dir = cfg.get("anomaly", {}).get("flowmap_dir")
        self.flowmap_path = None
        if fm_dir and store is not None:
            from .config import resolve_path

            self.flowmap_path = resolve_path(fm_dir) / f"flowmap_{scenario.get('name', 'custom')}_{self.engine.world.seed}.npz"
            self.pipeline.anomaly.flowmap.load(self.flowmap_path)

    async def step(self) -> Optional[dict]:
        for batch in self.engine.step():
            await self.bus.publish(TOPIC_OBSERVATIONS, batch)
        return await self.pipeline.tick(self.engine.t)

    async def run_headless(self, duration: Optional[float] = None,
                           on_step: Optional[Callable[["SimRunner", Optional[dict]], Awaitable[None] | None]] = None) -> None:
        end = self.engine.t + (duration if duration is not None else self.engine.duration)
        while self.engine.t < end - 1e-9:
            snap = await self.step()
            if on_step is not None:
                r = on_step(self, snap)
                if asyncio.iscoroutine(r):
                    await r
        self.close()

    async def run_realtime(self, stop: asyncio.Event) -> None:
        """Pace simulated time against the wall clock (``time_scale`` x real time)."""
        wall0, sim0 = time.monotonic(), self.engine.t
        while not stop.is_set() and not self.engine.finished:
            if self.paused:
                await asyncio.sleep(0.05)
                wall0, sim0 = time.monotonic(), self.engine.t
                continue
            target = sim0 + (time.monotonic() - wall0) * self.time_scale
            # Work in short slices so the event loop (API, websockets) stays responsive.
            slice_start = time.monotonic()
            while self.engine.t < target and time.monotonic() - slice_start < 0.05:
                await self.step()
            if target - self.engine.t > 2.0:  # cannot keep up: re-anchor instead of spiralling
                wall0, sim0 = time.monotonic(), self.engine.t
            await asyncio.sleep(0.005)
        self.close()

    def set_speed(self, scale: float) -> None:
        self.time_scale = max(0.1, min(float(scale), 50.0))

    def close(self) -> None:
        if self.flowmap_path is not None:
            self.flowmap_path.parent.mkdir(parents=True, exist_ok=True)
            self.pipeline.anomaly.flowmap.save(self.flowmap_path)
        if self.store is not None and self.run_id is not None:
            self.store.end_run(self.run_id)
