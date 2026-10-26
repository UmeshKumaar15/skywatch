"""The perception & prediction pipeline.

    sensors --(sensor.observations)--> [reorder buffer] -> tracker
        -> per updated track: classification -> anomaly -> intent -> prediction
        --(analytics.enriched / analytics.insights)--> dashboard, storage

Every track that receives a new observation is re-analysed (rate-limited per
track), so classification, intent and the multi-hypothesis prediction all
evolve continuously with incoming data. Coasting tracks are re-analysed on a
slower cadence so their predictions keep reflecting growing uncertainty.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import Counter, deque
from typing import Any, Callable, Optional

import numpy as np

from .anomaly.detector import AnomalyDetector
from .behavior.intent import IntentInference
from .classification.classifier import ObjectClassifier
from .config import resolve_path
from .core.bus import TOPIC_ENRICHED, TOPIC_INSIGHTS, TOPIC_OBSERVATIONS, TOPIC_TRACKS, EventBus
from .core.types import Insight, Observation, TrackStatus
from .prediction.learned import LearnedMotionModel
from .prediction.predictor import TrajectoryPredictor, serialize
from .storage.store import Store
from .tracking.tracker import MultiObjectTracker, Track


def _r(v: float, n: int = 1) -> float:
    return round(float(v), n)


class PerceptionPipeline:
    def __init__(self, cfg: dict[str, Any], bus: EventBus, geography: Optional[dict] = None, radius: float = 6000.0,
                 store: Optional[Store] = None, run_id: Optional[int] = None,
                 truth_provider: Optional[Callable[[], list]] = None, analytics: bool = True) -> None:
        self.cfg = cfg
        self.bus = bus
        pc = cfg.get("pipeline", {})
        self.reorder_window = float(pc.get("reorder_window", 1.2))
        self.max_age = float(pc.get("max_observation_age", 6.0))
        self.snapshot_period = 1.0 / float(pc.get("snapshot_rate_hz", 4))
        self.history_seconds = float(pc.get("history_seconds", 120))
        self.analytics_interval = float(pc.get("analytics_interval", 1.0))
        self.geo = geography or {}
        self.tracker = MultiObjectTracker({**cfg.get("tracking", {}), "zone_radius": radius})
        self.classifier = ObjectClassifier(cfg.get("classification", {}), resolve_path(cfg.get("classification", {}).get("model_path", "models/classifier.npz")))
        self.learned = LearnedMotionModel(resolve_path(cfg.get("prediction", {}).get("learned_model_path", "models/motion_model.npz")))
        self.predictor = TrajectoryPredictor(cfg.get("prediction", {}), self.learned, self.geo)
        self.intent = IntentInference(cfg.get("intent", {}), self.geo)
        self.anomaly = AnomalyDetector(cfg.get("anomaly", {}), self.geo, radius)
        self.analytics_enabled = analytics
        self.store = store
        self.run_id = run_id
        self.frame_period = 1.0 / float(cfg.get("storage", {}).get("frame_rate_hz", 2))
        self.truth_provider = truth_provider
        self._heap: list = []
        self._seq = itertools.count()
        self.now = -math.inf
        self._dirty: set[str] = set()
        self._last_snapshot = -math.inf
        self._last_frame = -math.inf
        self._last_state_rec: dict[str, float] = {}
        self.insights: deque[Insight] = deque(maxlen=300)
        self._new_insights: list[Insight] = []
        self.recent_obs: deque = deque(maxlen=600)
        self.stats = {"obs_total": 0, "stale_dropped": 0, "batches": 0, "track_ms": 0.0, "analytics_ms": 0.0,
                      "analytics_runs": 0, "latency_s": 0.0}
        self._rate = deque(maxlen=200)  # (time, n_obs)
        self.insight_counts: Counter = Counter()
        bus.subscribe(TOPIC_OBSERVATIONS, self.on_observations)

    # -------------------------------------------------------------- ingest
    def on_observations(self, batch: list[Observation]) -> None:
        if not batch:
            return
        t = round(max(o.timestamp for o in batch), 3)
        heapq.heappush(self._heap, (t, next(self._seq), batch))

    async def tick(self, now: float) -> Optional[dict]:
        """Advance the pipeline clock to ``now``; returns a snapshot when one is due."""
        self.now = now
        horizon = now - self.reorder_window
        t0 = time.perf_counter()
        processed_any = False
        while self._heap and self._heap[0][0] <= horizon:
            t, _, batch = heapq.heappop(self._heap)
            # Re-assemble scans that arrived in fragments (per-message network jitter).
            sid = batch[0].sensor_id
            while self._heap and self._heap[0][0] == t and self._heap[0][2][0].sensor_id == sid:
                batch = batch + heapq.heappop(self._heap)[2]
            if t < now - self.max_age or t < self.tracker.t - self.max_age:
                self.stats["stale_dropped"] += len(batch)
                continue
            self.tracker.process(batch)
            processed_any = True
            self.stats["obs_total"] += len(batch)
            self.stats["batches"] += 1
            self._rate.append((now, len(batch)))
            rec = batch[0].received_at
            if rec is not None:
                self.stats["latency_s"] = 0.95 * self.stats["latency_s"] + 0.05 * (now - batch[0].timestamp)
            for o in batch:
                self.recent_obs.append((o.timestamp, o))
            for tr in self.tracker.tracks.values():
                if tr.last_batch == self.tracker._batch_seq:
                    self._dirty.add(tr.id)
            if self.store is not None and self.run_id is not None:
                self.store.add_observations(self.run_id, batch)
        if not processed_any:
            self.tracker.advance(horizon)
        self.stats["track_ms"] = 0.9 * self.stats["track_ms"] + 0.1 * (time.perf_counter() - t0) * 1000
        if processed_any:
            await self.bus.publish(TOPIC_TRACKS, self.tracker)

        if self.analytics_enabled:
            self._run_analytics(now)
        if self._new_insights:
            ins, self._new_insights = self._new_insights, []
            await self.bus.publish(TOPIC_INSIGHTS, ins)
            if self.store is not None and self.run_id is not None:
                self.store.add_insights(self.run_id, ins)

        snap = None
        if now - self._last_snapshot >= self.snapshot_period - 1e-9:
            self._last_snapshot = now
            snap = self.snapshot(now)
            await self.bus.publish(TOPIC_ENRICHED, snap)
            if self.store is not None and self.run_id is not None and now - self._last_frame >= self.frame_period - 1e-9:
                self._last_frame = now
                self.store.add_frame(self.run_id, now, snap)
                self.store.maybe_flush()
        return snap

    # ----------------------------------------------------------- analytics
    def _run_analytics(self, now: float) -> None:
        t0 = time.perf_counter()
        t_eval = self.tracker.t
        for tr in list(self.tracker.tracks.values()):
            if tr.status == TrackStatus.TENTATIVE:
                continue
            last = tr.analytics.get("_t", -math.inf)
            due = (tr.id in self._dirty and t_eval - last >= self.analytics_interval) or t_eval - last >= 2.0 * self.analytics_interval
            if due:
                self.analyze(tr, t_eval)
        self._dirty.clear()
        dt = (time.perf_counter() - t0) * 1000
        self.stats["analytics_ms"] = 0.9 * self.stats["analytics_ms"] + 0.1 * dt

    def analyze(self, tr: Track, now: float) -> None:
        if tr.t < now - 0.05:
            tr.imm.predict(now - tr.t)
            tr.t = now
        a = tr.analytics
        a["_t"] = now
        self.stats["analytics_runs"] += 1
        # Class and intent posteriors evolve slowly once established: refresh them on
        # a slower cadence than kinematics-driven anomaly scores and predictions.
        slow_due = now - a.get("_t_slow", -math.inf) >= (2.0 if now - tr.created > 30 else 0.0)
        if slow_due or "class" not in a:
            cls = self.classifier.update(tr, now)
            a["class"] = {k: v for k, v in cls.items() if k != "features"}
        cls = a["class"]
        an, insights = self.anomaly.update(tr, now, cls["label"], cls["probs"])
        a["anomaly"] = an
        for i in insights:
            self.insights.append(i)
            self._new_insights.append(i)
            self.insight_counts[i.kind] += 1
        intent = self.intent.update(tr, now, cls["probs"], an["score"]) if slow_due or "intent" not in a else None
        if slow_due:
            a["_t_slow"] = now
        if intent is not None:
            a["intent"] = {k: v for k, v in intent.items() if k != "features"}
            a["intent_features"] = intent["features"]
        ip = a.get("intent", {}).get("probs", {})
        hyps = self.predictor.predict(tr, now, cls["probs"], ip)
        a["_hyps"] = hyps
        a["prediction"] = serialize(hyps, now, self.predictor.prune, self.predictor.max_h)
        trail = a.setdefault("_trail", deque(maxlen=int(self.history_seconds / 2)))
        if not trail or now - trail[-1][0] >= 2.0:
            p = tr.position
            trail.append((round(now, 1), _r(p[0]), _r(p[1]), _r(p[2])))
        if self.store is not None and self.run_id is not None:
            if now - self._last_state_rec.get(tr.id, -1e9) >= 1.0:
                self._last_state_rec[tr.id] = now
                self.store.add_track_states(self.run_id, [self._state_row(tr, now)])
                if int(now) % 5 == 0:
                    self.store.add_prediction(self.run_id, now, tr.id, a["prediction"])

    def _state_row(self, tr: Track, now: float) -> tuple:
        import json

        s = tr.state
        a = tr.analytics
        c = a.get("class", {})
        it = a.get("intent", {})
        sig = math.sqrt(max(np.trace(tr.covariance[0:3, 0:3]) / 3, 0))
        return (now, tr.id, tr.status.value, *[float(v) for v in s[0:6]], sig, tr.confidence(now), c.get("label"),
                json.dumps({k: round(v, 4) for k, v in c.get("probs", {}).items()}), it.get("label"),
                json.dumps({k: round(v, 4) for k, v in it.get("probs", {}).items()}),
                float(a.get("anomaly", {}).get("score", 0.0)), tr.identity, tr.truth_id)

    # ------------------------------------------------------------ snapshot
    def track_view(self, tr: Track, now: float) -> dict[str, Any]:
        a = tr.analytics
        P = tr.covariance
        s = tr.state
        disp_t = now
        pos = tr.position_at(disp_t)
        vel = s[3:6]
        c = a.get("class", {})
        it = a.get("intent", {})
        an = a.get("anomaly", {})
        probs = c.get("probs", {})
        top_cls = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
        iprobs = it.get("probs", {})
        top_int = sorted(iprobs.items(), key=lambda kv: -kv[1])[:4]
        view = {
            "id": tr.id,
            "status": tr.status.value,
            "t": _r(tr.t, 2),
            "pos": [_r(pos[0]), _r(pos[1]), _r(pos[2])],
            "vel": [_r(vel[0], 2), _r(vel[1], 2), _r(vel[2], 2)],
            "speed": _r(np.linalg.norm(vel[:2]), 2),
            "heading": _r((math.degrees(math.atan2(vel[0], vel[1])) + 360) % 360),
            "cov": [_r(P[0, 0]), _r(P[0, 1]), _r(P[1, 1])],
            "sigma_z": _r(math.sqrt(max(P[2, 2], 0))),
            "confidence": _r(tr.confidence(now), 3),
            "identity": tr.identity,
            "sources": dict(tr.sources),
            "age": _r(now - tr.created),
            "since_update": _r(now - tr.last_update, 2),
            "modes": {k: _r(v, 3) for k, v in tr.imm.mode_probabilities().items()},
            "class": {"label": c.get("label", "unknown"), "confidence": _r(c.get("confidence", 0), 3),
                      "entropy": _r(c.get("entropy", 1), 3), "probs": {k: _r(v, 3) for k, v in top_cls}},
            "intent": {"label": it.get("label"), "confidence": _r(it.get("confidence", 0), 3),
                       "probs": {k: _r(v, 3) for k, v in top_int}},
            "anomaly": {"score": _r(an.get("score", 0), 3), "top": an.get("top"),
                        "scores": {k: v for k, v in an.get("scores", {}).items() if v >= 0.15}},
            "trail": list(a.get("_trail", ())),
            "prediction": a.get("prediction"),
        }
        if tr.truth_id:
            view["truth_id"] = tr.truth_id
        return view

    def snapshot(self, now: float) -> dict[str, Any]:
        tracks = [self.track_view(tr, now) for tr in self.tracker.tracks.values()
                  if tr.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]
        tentative = [[tr.id, _r(tr.position[0]), _r(tr.position[1]), _r(tr.position[2])] for tr in self.tracker.tracks.values()
                     if tr.status == TrackStatus.TENTATIVE]
        while self.recent_obs and self.recent_obs[0][0] < now - self.reorder_window - 1.5:
            self.recent_obs.popleft()
        obs = [[o.sensor_type.value, _r(o.position[0]), _r(o.position[1]), _r(o.position[2])] for _, o in self.recent_obs]
        rate_win = [n for (t, n) in self._rate if t >= now - 5.0]
        cls_counts = Counter(t["class"]["label"] for t in tracks)
        intent_counts = Counter(t["intent"]["label"] for t in tracks if t["intent"]["label"])
        snap = {
            "t": _r(now, 2),
            "run_id": self.run_id,
            "tracks": tracks,
            "tentative": tentative,
            "observations": obs,
            "insights": [i.to_dict() for i in list(self.insights)[-25:]][::-1],
            "stats": {
                "tracks": len(tracks),
                "tentative": len(tentative),
                "obs_per_s": _r(sum(rate_win) / 5.0),
                "obs_total": self.stats["obs_total"],
                "stale_dropped": self.stats["stale_dropped"] + self.tracker.stats.stale_dropped,
                "track_ms": _r(self.stats["track_ms"], 2),
                "analytics_ms": _r(self.stats["analytics_ms"], 2),
                "latency_s": _r(self.stats["latency_s"], 2),
                "tracker": vars(self.tracker.stats).copy(),
                "classes": dict(cls_counts),
                "intents": dict(intent_counts),
                "insight_counts": dict(self.insight_counts),
                "prediction_accuracy": {k: {kk: _r(vv, 3) for kk, vv in v.items()} for k, v in self.predictor.accuracy_summary().items()},
                "models": {"classifier": self.classifier.ready, "learned_motion": self.learned.ready,
                           "flowmap_samples": int(self.anomaly.flowmap.total_updates)},
            },
        }
        if self.truth_provider is not None:
            snap["truth"] = self.truth_provider()
        return snap
