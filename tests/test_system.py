import asyncio

import numpy as np
from fastapi.testclient import TestClient

from skywatch.config import load_config, load_scenario
from skywatch.core.geo import LocalFrame
from skywatch.core.types import Observation
from skywatch.runner import SimRunner
from skywatch.sensors.adapters import SbsAdsbAdapter
from skywatch.server.app import create_app
from skywatch.storage.store import Store


def test_pipeline_end_to_end_with_recording(tmp_path):
    cfg = load_config()
    cfg["anomaly"]["flowmap_dir"] = str(tmp_path)
    store = Store(tmp_path / "t.db")
    sc = load_scenario("urban_mixed")
    runner = SimRunner(cfg, sc, store=store)
    snaps = []
    asyncio.run(runner.run_headless(60.0, lambda r, s: snaps.append(s) if s else None))
    last = snaps[-1]
    assert last["stats"]["tracks"] >= 10
    t = max(last["tracks"], key=lambda x: x["age"])
    assert t["class"]["probs"] and t["prediction"]["hypotheses"]
    assert abs(sum(t["class"]["probs"].values()) - 1) < 0.05 or len(t["class"]["probs"]) == 4
    runs = store.list_runs()
    assert runs and runs[0]["frames"] > 50
    fr = store.get_frame(runs[0]["id"], 30.0)
    assert fr and abs(fr["t"] - 30.0) < 1.0
    files = store.export_csv(runs[0]["id"], tmp_path / "exp")
    assert any("track_states" in f.name for f in files)
    assert list(tmp_path.glob("flowmap_urban_mixed_*.npz"))  # learned normal-traffic model persisted


def test_sbs_parser_and_external_schema():
    fr = LocalFrame(52.37, 4.90)
    ad = SbsAdsbAdapter("localhost", 30003, fr)
    ad.parse_line("MSG,4,1,1,4840D6,1,2026/10/26,10:00:00.000,2026/10/26,10:00:00.000,,,250,90,,,0,,,,,0", now=1.0)
    o = ad.parse_line("MSG,3,1,1,4840D6,1,2026/10/26,10:00:00.000,2026/10/26,10:00:00.000,,3000,,,52.38,4.91,,,,,,0", now=1.2)
    assert o is not None and o.identity == "ICAO:4840D6"
    assert abs(o.position[2] - 914.4) < 1.0 and o.velocity is not None and o.velocity[0] > 120
    ob = Observation.from_dict({"timestamp": 3.0, "sensor_type": "lidar", "position": [1, 2, 3], "sigma": [1, 1, 1]})
    assert ob.covariance[0, 0] == 1 and ob.sensor_type.value == "lidar"


def test_api(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["storage"]["path"] = str(tmp_path / "api.db")
    cfg["anomaly"]["flowmap_dir"] = str(tmp_path)
    with TestClient(create_app(cfg)) as client:
        meta = client.get("/api/meta").json()
        assert meta["mode"] == "simulation" and meta["scenarios"]
        r = client.post("/api/ingest", json=[{"timestamp": 1.0, "sensor_type": "radar", "position": [0, 0, 100]}])
        assert r.json() == {"accepted": 1}
        assert client.post("/api/control", json={"action": "pause"}).json()["paused"] is True
        assert client.post("/api/control", json={"action": "bogus"}).status_code == 400
        assert client.get("/").status_code == 200
