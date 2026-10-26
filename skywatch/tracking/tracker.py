"""Multi-object, multi-sensor tracker.

Per sensor scan (batch):
  1. predict every live track to the scan time (IMM),
  2. gate with the chi-square test on the Mahalanobis distance,
  3. solve a global-nearest-neighbour assignment (Hungarian algorithm) whose
     cost is the negative log-likelihood, with identity (ADS-B / Remote ID)
     constraints that forbid identity switches,
  4. update assigned tracks, initiate tentative tracks from the leftovers,
  5. manage life cycle (tentative -> confirmed -> coasting -> deleted) and
     merge duplicate tracks that never received distinct detections from the
     same scan.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ..core.types import Observation, SensorType, TrackStatus
from .history import TrackHistory
from .imm import NX, IMMFilter, measurement_matrices

COOPERATIVE = (SensorType.ADSB, SensorType.REMOTE_ID)


@dataclass
class Track:
    id: str
    imm: IMMFilter
    t: float
    created: float
    status: TrackStatus = TrackStatus.TENTATIVE
    last_update: float = 0.0
    n_updates: int = 0
    identity: Optional[str] = None
    identity_votes: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    last_by_type: dict = field(default_factory=dict)
    last_coop_time: Optional[float] = None
    history: TrackHistory = field(default_factory=TrackHistory)
    features: dict = field(default_factory=dict)  # name -> deque of recent values
    nis: deque = field(default_factory=lambda: deque(maxlen=30))
    truth_votes: Counter = field(default_factory=Counter)  # evaluation only
    last_batch: int = -1
    update_batches: deque = field(default_factory=lambda: deque(maxlen=8))
    merge_streak: dict = field(default_factory=dict)
    # Filled by the analytics stage
    analytics: dict = field(default_factory=dict)
    _pg: Optional[tuple] = None

    # ------------------------------------------------------------ accessors
    @property
    def state(self) -> np.ndarray:
        return self.imm.state

    @property
    def covariance(self) -> np.ndarray:
        return self.imm.covariance

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def truth_id(self) -> Optional[str]:
        return self.truth_votes.most_common(1)[0][0] if self.truth_votes else None

    def pregate(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Position, velocity and their 1-sigma radii (cached per filter state)."""
        combined = self.imm._combined()
        if self._pg is None or self._pg[0] is not combined:
            P = combined[1]
            self._pg = (combined, combined[0][0:3].copy(), combined[0][3:6].copy(),
                        math.sqrt(max(np.trace(P[0:3, 0:3]), 0.0)), math.sqrt(max(np.trace(P[3:6, 3:6]), 0.0)))
        return self._pg[1], self._pg[2], self._pg[3], self._pg[4]

    def position_at(self, t: float) -> np.ndarray:
        """Cheap constant-velocity extrapolation of the current estimate to time ``t``."""
        return self.position + self.velocity * (t - self.t)

    def feature_values(self, name: str) -> list[float]:
        return list(self.features.get(name, ()))

    def confidence(self, now: float) -> float:
        """Track existence/quality confidence in [0, 1].

        Combines evidence count, staleness and positional precision.
        """
        existence = 1.0 - math.exp(-self.n_updates / 3.0)
        stale = max(0.0, now - self.last_update - 1.5)
        staleness = math.exp(-stale / 6.0)
        sig = math.sqrt(max(np.trace(self.covariance[0:3, 0:3]), 0.0) / 3)
        precision = 1.0 / (1.0 + sig / 40.0)
        return float(existence * staleness * (0.55 + 0.45 * precision))


@dataclass
class TrackerStats:
    batches: int = 0
    observations: int = 0
    associated: int = 0
    initiated: int = 0
    confirmed: int = 0
    deleted: int = 0
    merged: int = 0
    reacquired: int = 0
    stale_dropped: int = 0


class MultiObjectTracker:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.q = dict(cfg.get("process_noise", {"cv": 0.6, "ca": 3.0, "ct": 1.0, "turn_rate": 0.04}))
        self.gate3 = chi2.ppf(cfg.get("gate_probability", 0.9995), 3)
        self.tracks: dict[str, Track] = {}
        self.t = -math.inf
        self._next = 1
        self.stats = TrackerStats()
        self._batch_seq = 0
        self.deleted_recent: deque = deque(maxlen=200)
        self._last_merge = -math.inf
        self._last_manage = -math.inf
        self.zone_radius: Optional[float] = cfg.get("zone_radius")

    # ----------------------------------------------------------- utilities
    def _new_track(self, obs: Observation) -> Track:
        x0 = np.zeros(NX)
        x0[0:3] = obs.position
        P0 = np.zeros((NX, NX))
        P0[0:3, 0:3] = obs.covariance
        vs = self.cfg.get("init_velocity_sigma", 60.0)
        if obs.velocity is not None:
            x0[3:6] = obs.velocity
            P0[3:6, 3:6] = obs.velocity_covariance if obs.velocity_covariance is not None else np.eye(3) * 4
        else:
            P0[3:6, 3:6] = np.diag([vs**2, vs**2, (vs * 0.25) ** 2])
        P0[6:9, 6:9] = np.eye(3) * 4.0
        P0[9, 9] = 0.05**2
        imm = IMMFilter(x0, P0, self.q, self.cfg.get("imm_transition_stay", 0.95))
        tid = f"T{self._next:04d}"
        self._next += 1
        tr = Track(id=tid, imm=imm, t=obs.timestamp, created=obs.timestamp)
        self.stats.initiated += 1
        self._record_update(tr, obs, nis=0.0)
        return tr

    @staticmethod
    def _obs_model(obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if obs.velocity is not None:
            z = np.r_[obs.position, obs.velocity]
            R = np.zeros((6, 6))
            R[0:3, 0:3] = obs.covariance
            R[3:6, 3:6] = obs.velocity_covariance if obs.velocity_covariance is not None else np.eye(3) * 4
            return z, R, measurement_matrices(True)
        return obs.position, obs.covariance, measurement_matrices(False)

    def _record_update(self, tr: Track, obs: Observation, nis: float) -> None:
        tr.last_update = obs.timestamp
        tr.n_updates += 1
        tr.sources[obs.sensor_type.value] += 1
        tr.last_by_type[obs.sensor_type.value] = obs.timestamp
        if obs.sensor_type in COOPERATIVE:
            tr.last_coop_time = obs.timestamp
        if obs.identity:
            tr.identity_votes[obs.identity] += 1
            tr.identity = tr.identity_votes.most_common(1)[0][0]
        for k, v in obs.features.items():
            dq = tr.features.get(k)
            if dq is None:
                dq = tr.features[k] = deque(maxlen=60)
            dq.append(float(v))
        tr.nis.append(nis)
        if obs.truth_id:
            tr.truth_votes[obs.truth_id] += 1
        tr.last_batch = self._batch_seq
        tr.update_batches.append(self._batch_seq)
        s = tr.state
        P = tr.covariance
        tr.history.append(obs.timestamp, s[0:3], s[3:6], float(math.sqrt(max(np.trace(P[0:3, 0:3]) / 3, 0))), obs.sensor_type.value)

    # ------------------------------------------------------------- process
    def process(self, batch: list[Observation]) -> None:
        if not batch:
            return
        t = max(o.timestamp for o in batch)
        if t < self.t - 1e-3:
            self.stats.stale_dropped += len(batch)
            return
        self.t = max(self.t, t)
        self._batch_seq += 1
        self.stats.batches += 1
        self.stats.observations += len(batch)

        live_all = [tr for tr in self.tracks.values() if tr.status != TrackStatus.DELETED]
        opos = np.array([o.position for o in batch])
        orad = np.array([math.sqrt(max(np.trace(o.covariance), 0.0)) for o in batch])
        # Cheap pre-gate: constant-velocity extrapolation with a generous
        # uncertainty bound. Only candidate tracks get a full IMM prediction;
        # the others are predicted lazily when a covering sensor reports.
        live: list[Track] = []
        if live_all:
            tpos = np.empty((len(live_all), 3))
            trad = np.empty(len(live_all))
            for k, tr in enumerate(live_all):
                dt = t - tr.t
                pos, vel, prad, vrad = tr.pregate()
                tpos[k] = pos + vel * dt
                trad[k] = prad + (vrad + 5.0) * dt + 20.0
            dist = np.linalg.norm(tpos[:, None, :] - opos[None, :, :], axis=2)
            near = (dist < 4.5 * (trad[:, None] + orad[None, :])).any(axis=1)
            ids = {o.identity for o in batch if o.identity}
            for k, tr in enumerate(live_all):
                if near[k] or (tr.identity and tr.identity in ids):
                    live.append(tr)
        for tr in live:
            tr.imm.predict(t - tr.t)
            tr.t = t

        n_t, n_o = len(live), len(batch)
        assigned_obs: set[int] = set()
        if n_t and n_o:
            BIG = 1e6
            cost = np.full((n_t, n_o), BIG)
            ocov = np.array([o.covariance for o in batch])
            oid = [o.identity for o in batch]
            for i, tr in enumerate(live):
                Pp = tr.covariance[0:3, 0:3]
                xp = tr.position
                prad = math.sqrt(max(np.trace(Pp), 0.0))
                if tr.identity:
                    cand = np.arange(n_o)
                else:
                    cand = np.nonzero(np.linalg.norm(opos - xp, axis=1) < 5.0 * (prad + orad) + 1.0)[0]
                if tr.identity:
                    cand = np.array([j for j in cand if not (oid[j] and oid[j] != tr.identity)], dtype=int)
                if len(cand) == 0:
                    continue
                S = Pp[None] + ocov[cand]
                d = opos[cand] - xp
                try:
                    sol = np.linalg.solve(S, d[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    continue
                d2 = np.einsum("ka,ka->k", d, sol)
                same = np.array([bool(tr.identity) and oid[j] == tr.identity for j in cand])
                ok = d2 <= self.gate3 * np.where(same, 25.0, 1.0)
                if not ok.any():
                    continue
                _, logdet = np.linalg.slogdet(S[ok])
                c = 0.5 * d2[ok] + 0.5 * logdet - 50.0 * same[ok] + (2.0 if tr.status == TrackStatus.TENTATIVE else 0.0)
                cost[i, cand[ok]] = c
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e5:
                    continue
                tr, obs = live[i], batch[j]
                z, R, H = self._obs_model(obs)
                y, S = tr.imm.innovation(z[:3], R[:3, :3], H[:3])
                nis = float(y @ np.linalg.solve(S, y))
                tr.imm.update(z, R, H)
                self._record_update(tr, obs, nis)
                assigned_obs.add(j)
                self.stats.associated += 1

        # Second pass: a confirmed track that missed its gate (typically during a
        # manoeuvre) may absorb a nearby leftover detection, with the measurement
        # covariance inflated to the observed discrepancy. Without this, the
        # detection would seed a duplicate track of the same object.
        updated = {tr.id for tr in live if tr.last_batch == self._batch_seq}
        free = [tr for tr in live if tr.id not in updated and tr.status != TrackStatus.TENTATIVE]
        left = [j for j in range(n_o) if j not in assigned_obs and (batch[j].quality >= 0.3 or batch[j].sensor_type in COOPERATIVE)]
        if free and left:
            ext = self.gate3 * self.cfg.get("extended_gate_factor", 6.0)
            C2 = np.full((len(free), len(left)), 1e6)
            lpos = opos[left]
            lcov = np.array([batch[j].covariance for j in left])
            for i, tr in enumerate(free):
                S = tr.covariance[0:3, 0:3][None] + lcov
                d = lpos - tr.position
                try:
                    d2 = np.einsum("ka,ka->k", d, np.linalg.solve(S, d[..., None])[..., 0])
                except np.linalg.LinAlgError:
                    continue
                for k, j in enumerate(left):
                    if tr.identity and batch[j].identity and tr.identity != batch[j].identity:
                        continue
                    if d2[k] < ext:
                        C2[i, k] = d2[k]
            for i, k in zip(*linear_sum_assignment(C2)):
                if C2[i, k] >= 1e6:
                    continue
                tr, obs = free[i], batch[left[k]]
                z, R, H = self._obs_model(obs)
                infl = max(1.0, C2[i, k] / self.gate3)
                R = R.copy()
                R[0:3, 0:3] *= infl
                tr.imm.update(z, R, H)
                self._record_update(tr, obs, float(C2[i, k]))
                assigned_obs.add(left[k])
                self.stats.associated += 1
                self.stats.reacquired += 1

        for j, obs in enumerate(batch):
            if j in assigned_obs:
                continue
            # Low-quality detections (likely clutter) do not seed tracks unless cooperative.
            if obs.quality < 0.3 and obs.sensor_type not in COOPERATIVE:
                continue
            tr = self._new_track(obs)
            self.tracks[tr.id] = tr

        if t - self._last_manage >= 0.2:
            self._last_manage = t
            self._manage(t)
        if t - self._last_merge >= 1.0:
            self._last_merge = t
            self._merge_duplicates()

    def _manage(self, t: float) -> None:
        c = self.cfg
        for tr in list(self.tracks.values()):
            age_since = t - tr.last_update
            if tr.status == TrackStatus.TENTATIVE:
                coop = any(k in tr.sources for k in ("adsb", "remote_id"))
                if tr.n_updates >= c.get("confirm_hits", 3) or (coop and tr.n_updates >= 2):
                    tr.status = TrackStatus.CONFIRMED
                    self.stats.confirmed += 1
                elif age_since > c.get("tentative_max_age", 4.0) or (t - tr.created > c.get("confirm_window", 5) * 2 and tr.n_updates < 2):
                    self._delete(tr)
                continue
            # Contact loss is expected near the ground (landing, below radar cover) and
            # beyond the zone edge; such tracks are retired sooner.
            pos = tr.position_at(t)
            expected_loss = pos[2] < c.get("low_altitude_m", 15.0) or (
                self.zone_radius is not None and float(np.linalg.norm(pos[:2])) > self.zone_radius * 1.15)
            if age_since > c.get("max_coast_time", 12.0) or (expected_loss and age_since > c.get("expected_loss_coast_time", 4.0)):
                self._delete(tr)
            elif age_since > 2.5:
                tr.status = TrackStatus.COASTING
            else:
                tr.status = TrackStatus.CONFIRMED

    def _delete(self, tr: Track) -> None:
        tr.status = TrackStatus.DELETED
        self.tracks.pop(tr.id, None)
        self.deleted_recent.append(tr)
        self.stats.deleted += 1

    def _merge_duplicates(self) -> None:
        """Merge track pairs that are statistically the same object.

        A pair is merged only after several consecutive agreeing scans *and*
        if the two were never updated by distinct detections of the same
        scan (which would prove they are two objects).
        """
        trs = [t for t in self.tracks.values() if t.status != TrackStatus.TENTATIVE]
        thr = chi2.ppf(0.99, 6) * self.cfg.get("merge_distance_sigma", 2.0) / 2.0
        for a_i in range(len(trs)):
            a = trs[a_i]
            for b in trs[a_i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                if a.identity and b.identity and a.identity != b.identity:
                    continue
                if np.linalg.norm(a.position - b.position) > 500:
                    a.merge_streak.pop(b.id, None)
                    continue
                if set(a.update_batches) & set(b.update_batches):
                    a.merge_streak.pop(b.id, None)
                    continue
                if a.t != b.t:
                    a.imm.predict(self.t - a.t)
                    b.imm.predict(self.t - b.t)
                    a.t = b.t = self.t
                d = a.state[0:6] - b.state[0:6]
                S = a.covariance[0:6, 0:6] + b.covariance[0:6, 0:6]
                d2 = float(d @ np.linalg.solve(S, d))
                same_identity = a.identity is not None and a.identity == b.identity
                # Statistical agreement, or physical proximity: duplicates of one object often
                # carry overconfident, slightly biased covariances, so also accept a small
                # absolute separation with matching velocity (close formations are protected
                # by the same-scan test above).
                close = float(np.linalg.norm(d[0:3])) < self.cfg.get("merge_distance_m", 30.0) and float(np.linalg.norm(d[3:6])) < 4.0
                if d2 < thr or same_identity or close:
                    a.merge_streak[b.id] = a.merge_streak.get(b.id, 0) + 1
                    if a.merge_streak[b.id] >= 3 or same_identity:
                        keep, drop = (a, b) if (a.created, a.n_updates) <= (b.created, b.n_updates) else (b, a)
                        keep.n_updates += drop.n_updates
                        keep.sources.update(drop.sources)
                        keep.identity_votes.update(drop.identity_votes)
                        keep.truth_votes.update(drop.truth_votes)
                        if keep.identity is None and drop.identity:
                            keep.identity = drop.identity
                        if drop.last_update > keep.last_update:
                            keep.imm = drop.imm
                            keep.last_update = drop.last_update
                        self._delete(drop)
                        self.stats.merged += 1
                else:
                    a.merge_streak.pop(b.id, None)

    def advance(self, t: float) -> None:
        """Advance management clock without observations (e.g. total sensor outage)."""
        if t > self.t:
            self.t = t
            self._manage(t)

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]
