"""FastAPI service: live websocket stream, control API, playback API, ingest API."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import list_scenarios, load_scenario, resolve_path
from ..core.bus import TOPIC_ENRICHED, TOPIC_OBSERVATIONS, InMemoryBus, QueuedSubscriber
from ..core.geo import LocalFrame
from ..core.types import Observation
from ..pipeline import PerceptionPipeline
from ..runner import SimRunner
from ..storage.store import Store

STATIC = Path(__file__).parent / "static"


class ExternalRunner:
    """Pipeline driven by the wall clock, fed by real sensor adapters / HTTP ingest."""

    def __init__(self, cfg: dict[str, Any], store: Optional[Store], source: str) -> None:
        self.cfg = cfg
        self.bus = InMemoryBus()
        zone = cfg.get("zone", {})
        geo = zone.get("geography") or {"radius": zone.get("radius", 6000), "corridors": [], "nofly": []}
        self.description = {"scenario": source, "seed": None, "radius": zone.get("radius", 6000), "geography": geo,
                            "sensors": [], "wind": [0, 0, 0]}
        self.store = store
        self.run_id = store.start_run("external", source, None, self.description) if store is not None else None
        self.pipeline = PerceptionPipeline(cfg, self.bus, geo, zone.get("radius", 6000), store=store, run_id=self.run_id)
        self.source = source
        self.paused = False
        self.time_scale = 1.0
        o = zone.get("origin", {})
        self.frame = LocalFrame(o.get("lat", 0.0), o.get("lon", 0.0), o.get("alt", 0.0))

    async def _consume(self, adapter) -> None:
        async for batch in adapter.stream():
            await self.bus.publish(TOPIC_OBSERVATIONS, batch)

    async def run_realtime(self, stop: asyncio.Event) -> None:
        from ..sensors.adapters import JsonlReplayAdapter, SbsAdsbAdapter

        tasks = []
        if self.source.startswith("sbs://"):
            host, port = self.source[6:].split(":")
            tasks.append(asyncio.create_task(self._consume(SbsAdsbAdapter(host, int(port), self.frame))))
        elif self.source.startswith("replay:"):
            tasks.append(asyncio.create_task(self._consume(JsonlReplayAdapter(self.source[7:], loop=True))))
        try:
            while not stop.is_set():
                if not self.paused:
                    await self.pipeline.tick(time.time())
                await asyncio.sleep(0.05)
        finally:
            for t in tasks:
                t.cancel()
            if self.store is not None and self.run_id is not None:
                self.store.end_run(self.run_id)

    def set_speed(self, scale: float) -> None:
        pass


class AppState:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.store = Store(resolve_path(cfg["storage"]["path"])) if cfg.get("storage", {}).get("record", True) else None
        self.runner: Any = None
        self.task: Optional[asyncio.Task] = None
        self.stop: Optional[asyncio.Event] = None
        self.latest: Optional[dict] = None
        self.clients: set[QueuedSubscriber] = set()
        self.started_at = time.time()

    def _on_snapshot(self, snap: dict) -> None:
        self.latest = snap
        for c in list(self.clients):
            c(snap)

    async def start(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> None:
        await self.shutdown()
        srv = self.cfg.get("server", {})
        source = srv.get("source", "sim")
        if source == "sim":
            name = scenario or srv.get("scenario", "urban_mixed")
            sc = load_scenario(name, seed=seed if seed is not None else srv.get("seed"))
            sc["duration"] = max(float(sc.get("duration", 3600)), 24 * 3600)  # run "forever" in live mode
            self.runner = SimRunner(self.cfg, sc, store=self.store)
            self.runner.set_speed(float(srv.get("time_scale", 1.0)))
        else:
            self.runner = ExternalRunner(self.cfg, self.store, source)
        self.runner.bus.subscribe(TOPIC_ENRICHED, self._on_snapshot)
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(self.runner.run_realtime(self.stop))
        meta = self.meta()
        for c in list(self.clients):
            c({"type": "meta", **meta})

    async def shutdown(self) -> None:
        if self.task is not None and self.stop is not None:
            self.stop.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.task, 5.0)
        self.task = None

    def meta(self) -> dict[str, Any]:
        r = self.runner
        return {
            "mode": "simulation" if isinstance(r, SimRunner) else "external",
            "description": r.description if r else None,
            "run_id": getattr(r, "run_id", None),
            "paused": getattr(r, "paused", False),
            "time_scale": getattr(r, "time_scale", 1.0),
            "scenarios": list_scenarios(),
        }


def create_app(cfg: dict[str, Any]) -> FastAPI:
    state = AppState(cfg)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.start()
        yield
        await state.shutdown()
        if state.store is not None:
            state.store.close()

    app = FastAPI(title="SkyWatch", version="0.1.0", lifespan=lifespan)
    app.state.sky = state
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    async def meta() -> dict:
        return state.meta()

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "uptime_s": round(time.time() - state.started_at, 1), "t": state.latest["t"] if state.latest else None}

    @app.get("/api/scenarios")
    async def scenarios() -> list:
        return list_scenarios()

    @app.get("/api/snapshot")
    async def snapshot() -> JSONResponse:
        return JSONResponse(state.latest or {})

    @app.post("/api/control")
    async def control(cmd: dict = Body(...)) -> dict:
        action = cmd.get("action")
        r = state.runner
        if action == "pause":
            r.paused = True
        elif action == "resume":
            r.paused = False
        elif action == "speed":
            r.set_speed(float(cmd.get("speed", 1.0)))
        elif action == "restart":
            await state.start(cmd.get("scenario"), cmd.get("seed"))
        else:
            raise HTTPException(400, f"unknown action {action!r}")
        return state.meta()

    @app.post("/api/ingest")
    async def ingest(payload: Any = Body(...)) -> dict:
        """Accept observations from external sensor gateways (external JSON schema)."""
        items = payload if isinstance(payload, list) else [payload]
        try:
            obs = [Observation.from_dict(d) for d in items]
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(422, f"invalid observation: {e}")
        groups: dict[tuple, list[Observation]] = {}
        for o in obs:
            groups.setdefault((o.sensor_id, round(o.timestamp, 2)), []).append(o)
        for batch in groups.values():
            await state.runner.bus.publish(TOPIC_OBSERVATIONS, batch)
        return {"accepted": len(obs)}

    @app.get("/api/models")
    async def models() -> dict:
        p = resolve_path("models/training_report.json")
        rep = json.loads(p.read_text()) if p.exists() else None
        base = state.runner.pipeline.anomaly.baselines.describe() if state.runner else {}
        return {"training_report": rep, "anomaly_baselines": base}

    # ------------------------------------------------------------- playback
    def _store() -> Store:
        if state.store is None:
            raise HTTPException(404, "recording disabled")
        state.store.flush()
        return state.store

    @app.get("/api/runs")
    async def runs() -> list:
        return [r for r in _store().list_runs() if r["frames"]]

    @app.get("/api/runs/{run_id}")
    async def run_info(run_id: int) -> dict:
        st = _store()
        return {"meta": st.run_meta(run_id), "frame_times": st.frame_times(run_id)}

    @app.get("/api/runs/{run_id}/frame")
    async def run_frame(run_id: int, t: float) -> JSONResponse:
        fr = _store().get_frame(run_id, t)
        if fr is None:
            raise HTTPException(404, "no frame")
        return JSONResponse(fr)

    @app.get("/api/runs/{run_id}/track/{track_id}")
    async def run_track(run_id: int, track_id: str, t0: float, t1: float) -> list:
        return _store().track_path(run_id, track_id, t0, t1)

    @app.get("/api/runs/{run_id}/insights")
    async def run_insights(run_id: int) -> list:
        return _store().insights(run_id)

    # ------------------------------------------------------------ websocket
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        sub = QueuedSubscriber(maxsize=2)
        state.clients.add(sub)
        try:
            await ws.send_text(json.dumps({"type": "meta", **state.meta()}))
            if state.latest:
                await ws.send_text(json.dumps({"type": "snapshot", **state.latest}))
            while True:
                msg = await sub.queue.get()
                if msg.get("type") == "meta":
                    await ws.send_text(json.dumps(msg))
                else:
                    await ws.send_text(json.dumps({"type": "snapshot", **msg}, separators=(",", ":")))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state.clients.discard(sub)

    return app
