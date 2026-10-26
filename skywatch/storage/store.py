"""Persistent storage of observations, track states, predictions and insights.

SQLite keeps deployment trivial; the schema is deliberately plain so it
ports to PostgreSQL/TimescaleDB unchanged. Dashboard snapshots are stored as
compressed JSON "frames" which power frame-by-frame playback.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL, ended_at REAL, source TEXT, scenario TEXT, seed INTEGER,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS frames (
    run_id INTEGER, t REAL, data BLOB, PRIMARY KEY (run_id, t)
);
CREATE TABLE IF NOT EXISTS track_states (
    run_id INTEGER, t REAL, track_id TEXT, status TEXT,
    x REAL, y REAL, z REAL, vx REAL, vy REAL, vz REAL, pos_sigma REAL, confidence REAL,
    class_label TEXT, class_probs TEXT, intent TEXT, intent_probs TEXT,
    anomaly_score REAL, identity TEXT, truth_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts_run_track ON track_states(run_id, track_id, t);
CREATE TABLE IF NOT EXISTS predictions (
    run_id INTEGER, t REAL, track_id TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_run_track ON predictions(run_id, track_id, t);
CREATE TABLE IF NOT EXISTS observations (
    run_id INTEGER, t REAL, sensor_id TEXT, sensor_type TEXT, x REAL, y REAL, z REAL,
    sx REAL, sy REAL, sz REAL, identity TEXT, truth_id TEXT
);
CREATE TABLE IF NOT EXISTS insights (
    run_id INTEGER, t REAL, track_id TEXT, kind TEXT, score REAL, severity TEXT, message TEXT, evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_ins_run ON insights(run_id, t);
"""


class Store:
    def __init__(self, path: str | Path, record_observations: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.record_observations = record_observations
        self._buf: dict[str, list] = {"frames": [], "track_states": [], "predictions": [], "observations": [], "insights": []}
        self._last_flush = time.time()

    # ------------------------------------------------------------------ runs
    def start_run(self, source: str, scenario: Optional[str], seed: Optional[int], meta: dict[str, Any]) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO runs(started_at, source, scenario, seed, meta) VALUES (?,?,?,?,?)",
                (time.time(), source, scenario, seed, json.dumps(meta)),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def end_run(self, run_id: int) -> None:
        self.flush()
        with self.lock:
            self.conn.execute("UPDATE runs SET ended_at=? WHERE id=?", (time.time(), run_id))
            self.conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT r.id, r.started_at, r.ended_at, r.source, r.scenario, r.seed, "
                "(SELECT COUNT(*) FROM frames f WHERE f.run_id=r.id), (SELECT MIN(t) FROM frames f WHERE f.run_id=r.id), "
                "(SELECT MAX(t) FROM frames f WHERE f.run_id=r.id) FROM runs r ORDER BY r.id DESC"
            ).fetchall()
        return [dict(zip(("id", "started_at", "ended_at", "source", "scenario", "seed", "frames", "t_min", "t_max"), r)) for r in rows]

    def run_meta(self, run_id: int) -> dict[str, Any]:
        with self.lock:
            r = self.conn.execute("SELECT meta FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(r[0]) if r and r[0] else {}

    # --------------------------------------------------------------- writing
    def add_frame(self, run_id: int, t: float, snapshot: dict[str, Any]) -> None:
        blob = zlib.compress(json.dumps(snapshot, separators=(",", ":")).encode(), 6)
        self._buf["frames"].append((run_id, round(t, 3), blob))

    def add_track_states(self, run_id: int, rows: Iterable[tuple]) -> None:
        self._buf["track_states"].extend((run_id, *r) for r in rows)

    def add_prediction(self, run_id: int, t: float, track_id: str, data: dict) -> None:
        self._buf["predictions"].append((run_id, t, track_id, json.dumps(data, separators=(",", ":"))))

    def add_observations(self, run_id: int, obs: Iterable) -> None:
        if not self.record_observations:
            return
        for o in obs:
            sd = [float(o.covariance[i, i]) ** 0.5 for i in range(3)]
            self._buf["observations"].append((run_id, o.timestamp, o.sensor_id, o.sensor_type.value, *map(float, o.position), *sd, o.identity, o.truth_id))

    def add_insights(self, run_id: int, insights: Iterable) -> None:
        for i in insights:
            self._buf["insights"].append((run_id, i.timestamp, i.track_id, i.kind, i.score, i.severity, i.message, json.dumps(i.to_dict()["evidence"])))

    def maybe_flush(self, every: float = 2.0) -> None:
        if time.time() - self._last_flush >= every:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            b = self._buf
            if b["frames"]:
                self.conn.executemany("INSERT OR REPLACE INTO frames VALUES (?,?,?)", b["frames"])
            if b["track_states"]:
                self.conn.executemany("INSERT INTO track_states VALUES (" + ",".join("?" * 19) + ")", b["track_states"])
            if b["predictions"]:
                self.conn.executemany("INSERT INTO predictions VALUES (?,?,?,?)", b["predictions"])
            if b["observations"]:
                self.conn.executemany("INSERT INTO observations VALUES (" + ",".join("?" * 12) + ")", b["observations"])
            if b["insights"]:
                self.conn.executemany("INSERT INTO insights VALUES (?,?,?,?,?,?,?,?)", b["insights"])
            self.conn.commit()
            for k in b:
                b[k] = []
        self._last_flush = time.time()

    # --------------------------------------------------------------- reading
    def frame_times(self, run_id: int) -> list[float]:
        with self.lock:
            return [r[0] for r in self.conn.execute("SELECT t FROM frames WHERE run_id=? ORDER BY t", (run_id,))]

    def get_frame(self, run_id: int, t: float) -> Optional[dict[str, Any]]:
        """Frame at or just before ``t``."""
        with self.lock:
            r = self.conn.execute("SELECT data FROM frames WHERE run_id=? AND t<=? ORDER BY t DESC LIMIT 1", (run_id, t + 1e-6)).fetchone()
            if r is None:
                r = self.conn.execute("SELECT data FROM frames WHERE run_id=? ORDER BY t LIMIT 1", (run_id,)).fetchone()
        return json.loads(zlib.decompress(r[0])) if r else None

    def track_path(self, run_id: int, track_id: str, t0: float, t1: float) -> list[list[float]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT t, x, y, z FROM track_states WHERE run_id=? AND track_id=? AND t BETWEEN ? AND ? ORDER BY t",
                (run_id, track_id, t0, t1)).fetchall()
        return [[round(a, 2), round(b, 1), round(c, 1), round(d, 1)] for a, b, c, d in rows]

    def insights(self, run_id: int, limit: int = 500) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT t, track_id, kind, score, severity, message, evidence FROM insights WHERE run_id=? ORDER BY t DESC LIMIT ?",
                                     (run_id, limit)).fetchall()
        return [{"t": r[0], "track_id": r[1], "kind": r[2], "score": r[3], "severity": r[4], "message": r[5], "evidence": json.loads(r[6])} for r in rows]

    def export_csv(self, run_id: int, out_dir: str | Path) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.flush()
        files = []
        for table in ("track_states", "predictions", "insights", "observations"):
            with self.lock:
                cur = self.conn.execute(f"SELECT * FROM {table} WHERE run_id=?", (run_id,))
                cols = [c[0] for c in cur.description]
                rows = cur.fetchall()
            if not rows:
                continue
            p = out / f"run{run_id}_{table}.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
            files.append(p)
        return files

    def close(self) -> None:
        self.flush()
        self.conn.close()
