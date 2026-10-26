"""Multi-hypothesis trajectory prediction.

For every track we generate several *motion hypotheses*, each a sequence of
Gaussian position estimates over the prediction horizon:

  physics   cv       constant velocity (IMM CV state)
            turn     continue the current coordinated turn (IMM CT state)
            accel    continue current acceleration briefly (IMM CA state)
            stop     decelerate to a hover (rotorcraft / drones)
  context   route    follow the nearest known operational corridor
            orbit    continue a fitted circular orbit / holding pattern
  learned   learned  data-driven MLP residual model

Hypothesis weights are maintained with a Bayesian filter: the prediction
each hypothesis made a few seconds ago is scored against where the object
actually is now, the likelihoods multiply into the weights, and a fraction
of the context-dependent prior is mixed back in so discarded hypotheses can
be revived when behaviour changes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.types import OBJECT_CLASSES, ObjectClass
from ..tracking.history import distance_to_polylines, fit_circle, resample
from .learned import LearnedMotionModel, build_input

C = ObjectClass

# Expected manoeuvre intensity (m/s^1.5) for open-loop uncertainty growth.
PRED_Q = {
    C.DELIVERY_DRONE: 0.35, C.RECREATIONAL_DRONE: 0.8, C.INDUSTRIAL_DRONE: 0.4, C.BIRD: 0.8, C.HELICOPTER: 0.5,
    C.COMMERCIAL_AIRCRAFT: 0.3, C.EMERGENCY_VEHICLE: 0.6, C.WEATHER_BALLOON: 0.15, C.UNKNOWN: 0.7,
}
CAN_HOVER = (C.DELIVERY_DRONE, C.RECREATIONAL_DRONE, C.INDUSTRIAL_DRONE, C.HELICOPTER, C.EMERGENCY_VEHICLE)
LABELS = {
    "cv": "Continue straight", "turn": "Continue turn", "accel": "Continue accelerating", "stop": "Slow to hover",
    "route": "Follow corridor", "orbit": "Continue orbit", "learned": "Learned pattern",
}
EVAL_HORIZONS = (5.0, 10.0, 20.0, 30.0, 60.0)


@dataclass
class Hypothesis:
    kind: str
    prior: float
    times: np.ndarray
    means: np.ndarray  # (H, 3)
    covs: np.ndarray  # (H, 3, 3)
    weight: float = 0.0
    note: str = ""

    def at_many(self, dts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i0, i1, f = _interp_index(self.times, np.asarray(dts, dtype=float))
        m = self.means[i0] * (1 - f)[:, None] + self.means[i1] * f[:, None]
        c = self.covs[i0] * (1 - f)[:, None, None] + self.covs[i1] * f[:, None, None]
        return m, c

    def at(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated mean / covariance ``dt`` seconds after the prediction time."""
        ts = self.times
        if dt <= ts[0]:
            return self.means[0], self.covs[0]
        i = int(np.searchsorted(ts, dt))
        if i == 0:
            i = 1
        if i >= len(ts):
            return self.means[-1], self.covs[-1]
        f = (dt - ts[i - 1]) / (ts[i] - ts[i - 1])
        return self.means[i - 1] * (1 - f) + self.means[i] * f, self.covs[i - 1] * (1 - f) + self.covs[i] * f


_INTERP_CACHE: dict = {}


def _interp_index(times: np.ndarray, dts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracketing indices and fractions for linear interpolation (cached per grid)."""
    key = (times.tobytes(), dts.tobytes())
    hit = _INTERP_CACHE.get(key)
    if hit is None:
        d = np.clip(dts, times[0], times[-1])
        i1 = np.clip(np.searchsorted(times, d), 1, len(times) - 1)
        i0 = i1 - 1
        f = (d - times[i0]) / (times[i1] - times[i0])
        hit = (i0, i1, f)
        if len(_INTERP_CACHE) < 256:
            _INTERP_CACHE[key] = hit
    return hit


def _gauss_logpdf(x: np.ndarray, m: np.ndarray, S: np.ndarray) -> float:
    d = x - m
    try:
        sol = np.linalg.solve(S, d)
    except np.linalg.LinAlgError:
        return -50.0
    _, logdet = np.linalg.slogdet(S)
    return float(-0.5 * (d @ sol) - 0.5 * logdet - 1.5 * math.log(2 * math.pi))


def _propagate_linear(x6: np.ndarray, P6: np.ndarray, times: np.ndarray, q: float, accel: Optional[np.ndarray] = None,
                      accel_time: float = 0.0, decay_tau: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form open-loop propagation (vectorised over the time grid).

    Continuous white-noise acceleration of spectral density ``q**2``; optional
    constant acceleration applied for ``accel_time`` seconds, or exponential
    velocity decay with time constant ``decay_tau`` (deceleration to hover).
    """
    t = np.asarray(times, dtype=float)
    p0, v0 = x6[0:3], x6[3:6]
    if decay_tau is not None:
        g = decay_tau * (1 - np.exp(-t / decay_tau))  # d pos / d v0
    else:
        g = t
    means = p0[None, :] + g[:, None] * v0[None, :]
    if accel is not None:
        ta = np.minimum(t, accel_time)
        means = means + (0.5 * ta**2 + accel_time * np.maximum(t - accel_time, 0.0))[:, None] * accel[None, :]
    Ppp, Ppv, Pvv = P6[0:3, 0:3], P6[0:3, 3:6], P6[3:6, 3:6]
    covs = (Ppp[None] + g[:, None, None] * (Ppv + Ppv.T)[None] + (g**2)[:, None, None] * Pvv[None])
    qn = (q**2) * t**3 / 3.0
    covs = covs + qn[:, None, None] * np.diag([1.0, 1.0, 0.5])[None]
    return means, covs


class TrajectoryPredictor:
    def __init__(self, cfg: dict[str, Any], learned: Optional[LearnedMotionModel] = None, geography: Optional[dict] = None) -> None:
        self.cfg = cfg
        self.horizon = float(cfg.get("horizon", 60))
        step = float(cfg.get("step", 2))
        self.times = np.unique(np.r_[0.0, np.arange(1.0, 5.0), np.arange(step, self.horizon + 1e-9, step)])
        self.prune = float(cfg.get("hypothesis_prune", 0.04))
        self.inertia = float(cfg.get("weight_inertia", 0.75))
        self.max_h = int(cfg.get("max_hypotheses", 5))
        self.learned = learned
        self.geo = geography or {}
        # Online accuracy: horizon -> [sum_err_top, sum_err_mix, sum_err_cv, n, inside_2sigma]
        self.acc = {h: np.zeros(5) for h in EVAL_HORIZONS}

    # ------------------------------------------------------------ helpers
    def _q(self, class_probs: dict[str, float]) -> float:
        return sum(PRED_Q[C(k)] * v for k, v in class_probs.items())

    def _route_hypothesis(self, pos, vel, speed, q) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        corridors = self.geo.get("corridors") or []
        if not corridors or speed < 3:
            return None
        d, li, si, u = distance_to_polylines(pos, corridors)
        if d > 150:
            return None
        line = [np.asarray(p[:2], float) for p in corridors[li]]
        seg = line[si + 1] - line[si]
        fwd = float(seg @ vel[:2]) >= 0
        pts = line if fwd else line[::-1]
        si_dir = si if fwd else len(line) - 2 - si
        heading = vel[:2] / max(np.linalg.norm(vel[:2]), 1e-6)
        segd = pts[si_dir + 1] - pts[si_dir]
        if float(segd @ heading) / max(np.linalg.norm(segd), 1e-6) < 0.5:  # not aligned with the corridor
            return None
        # Walk the polyline from the projection point at current speed.
        proj = line[si] + u * (line[si + 1] - line[si])
        path = [proj] + pts[si_dir + 1:]
        means = np.zeros((len(self.times), 3))
        covs = np.zeros((len(self.times), 3, 3))
        for k, t in enumerate(self.times):
            dist = speed * t
            p = path[0]
            remaining = dist
            for nxt in path[1:]:
                L = float(np.linalg.norm(nxt - p))
                if remaining <= L:
                    p = p + (nxt - p) * (remaining / max(L, 1e-6))
                    remaining = 0
                    break
                remaining -= L
                p = nxt
            # Blend from the current (possibly offset) position onto the corridor.
            blend = min(1.0, t / 10.0)
            xy = pos[:2] + (p - proj) * 1.0 + (proj - pos[:2]) * blend
            means[k] = [xy[0], xy[1], pos[2] + vel[2] * min(t, 5.0)]
            along = (0.15 * speed * t) ** 2 + (q * t**1.5 * 0.3) ** 2 + 25
            cross = 15**2 + (0.5 * t) ** 2
            covs[k] = np.diag([along * 0.5 + cross * 0.5, along * 0.5 + cross * 0.5, 9 + (0.5 * t) ** 2])
        return means, covs, f"corridor {li}"

    def _orbit_hypothesis(self, track, now, pos, vel, q) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        s = resample(track.history, now, 120.0, dt=1.0)
        if len(s) < 25:
            return None
        c, r, resid = fit_circle(s.pos[:, :2])
        if not (15 < r < 4000) or resid > 0.2 * r:
            return None
        ang = np.unwrap(np.arctan2(s.pos[:, 1] - c[1], s.pos[:, 0] - c[0]))
        sweep = abs(ang[-1] - ang[0])
        if sweep < math.radians(150):
            return None
        rel = pos[:2] - c
        cross = rel[0] * vel[1] - rel[1] * vel[0]
        direction = 1.0 if cross > 0 else -1.0
        speed = float(np.linalg.norm(vel[:2]))
        omega = direction * speed / r
        a0 = math.atan2(rel[1], rel[0])
        rr = float(np.linalg.norm(rel))
        means = np.zeros((len(self.times), 3))
        covs = np.zeros((len(self.times), 3, 3))
        for k, t in enumerate(self.times):
            a = a0 + omega * t
            rad = rr + (r - rr) * min(1.0, t / 10)
            means[k] = [c[0] + rad * math.cos(a), c[1] + rad * math.sin(a), pos[2]]
            sig = resid + 3 + 0.05 * speed * t
            covs[k] = np.diag([sig**2, sig**2, 4 + (0.3 * t) ** 2])
        return means, covs, f"r={r:.0f}m, {'ccw' if direction > 0 else 'cw'}"

    # -------------------------------------------------------------- predict
    def predict(self, track, now: float, class_probs: dict[str, float], intent_probs: dict[str, float]) -> list[Hypothesis]:
        imm = track.imm
        t_ref = track.t
        q = self._q(class_probs)
        mu = imm.mode_probabilities()
        xs = {m: imm.x[i] for i, m in enumerate(("cv", "ca", "ct"))}
        Ps = {m: imm.P[i] for i, m in enumerate(("cv", "ca", "ct"))}
        state = track.state
        pos, vel = state[0:3], state[3:6]
        speed = float(np.linalg.norm(vel[:2]))
        hover_p = sum(class_probs.get(c.value, 0) for c in CAN_HOVER)
        hyps: list[Hypothesis] = []

        m, c = _propagate_linear(xs["cv"][:6], Ps["cv"][:6, :6], self.times, q)
        hyps.append(Hypothesis("cv", 0.15 + mu["cv"], self.times, m, c))

        w = float(xs["ct"][9])
        if abs(w) > 0.012 and speed > 2:
            m = np.zeros((len(self.times), 3))
            vx, vy = xs["ct"][3], xs["ct"][4]
            for k, t in enumerate(self.times):
                sw, cw = math.sin(w * t), math.cos(w * t)
                m[k] = xs["ct"][0:3] + np.array([(sw * vx - (1 - cw) * vy) / w, ((1 - cw) * vx + sw * vy) / w, xs["ct"][5] * t])
            _, cc = _propagate_linear(xs["ct"][:6], Ps["ct"][:6, :6], self.times, q)
            sw2 = Ps["ct"][9, 9]
            cc = cc + np.einsum("h,ab->hab", (math.sqrt(sw2) * speed * self.times**2 / 2) ** 2, np.diag([1.0, 1.0, 0.0]))
            hyps.append(Hypothesis("turn", mu["ct"] + 0.05, self.times, m, cc, note=f"{math.degrees(w):+.1f} deg/s"))

        a = xs["ca"][6:9]
        if np.linalg.norm(a) > 0.4:
            m, c = _propagate_linear(xs["ca"][:6], Ps["ca"][:6, :6], self.times, q * 1.3, accel=a, accel_time=6.0)
            hyps.append(Hypothesis("accel", 0.6 * mu["ca"], self.times, m, c, note=f"|a|={np.linalg.norm(a):.1f} m/s2"))

        if hover_p > 0.3 and speed > 1.5:
            m, c = _propagate_linear(state[:6], track.covariance[:6, :6], self.times, q, decay_tau=4.0)
            hyps.append(Hypothesis("stop", 0.12 * hover_p + 0.2 * intent_probs.get("hovering", 0), self.times, m, c))

        route = self._route_hypothesis(pos, vel, speed, q)
        if route is not None:
            p_del = class_probs.get("delivery_drone", 0)
            hyps.append(Hypothesis("route", 0.1 + 0.5 * p_del + 0.4 * intent_probs.get("delivery_route", 0), self.times, route[0], route[1], note=route[2]))

        orbit = self._orbit_hypothesis(track, t_ref, pos, vel, q)
        if orbit is not None:
            p_orb = intent_probs.get("inspection", 0) + intent_probs.get("holding_pattern", 0) + 0.5 * intent_probs.get("emergency_response", 0)
            hyps.append(Hypothesis("orbit", 0.15 + 0.6 * p_orb, self.times, orbit[0], orbit[1], note=orbit[2]))

        if self.learned is not None and self.learned.ready:
            cp = np.array([class_probs.get(c.value, 0.0) for c in OBJECT_CLASSES])
            mp = np.array([mu["cv"], mu["ca"], mu["ct"]])
            built = build_input(track.history, t_ref, pos, vel, cp, mp)
            if built is not None:
                m, c = self.learned.predict(built[0], built[1], self.times)
                c = c + track.covariance[0:3, 0:3][None, :, :]
                hyps.append(Hypothesis("learned", 0.45, self.times, m, c))

        for h in hyps:  # ground constraint
            np.maximum(h.means[:, 2], 0.0, out=h.means[:, 2])
        self._update_weights(track, t_ref, hyps)
        self._score_accuracy(track, t_ref, hyps)
        return hyps

    # ------------------------------------------------------ Bayesian weights
    def _update_weights(self, track, t_ref: float, hyps: list[Hypothesis]) -> None:
        st = track.analytics.setdefault("_pred", {"w": {}, "past": deque(maxlen=12)})
        prior_tot = sum(h.prior for h in hyps)
        priors = {h.kind: h.prior / prior_tot for h in hyps}
        w_prev = st["w"]
        # Score the prediction issued ~3 s ago against the current estimate.
        lik: dict[str, float] = {}
        ref = None
        for t0, past in st["past"]:
            if t_ref - t0 >= 2.5:
                ref = (t0, past)
        if ref is not None:
            t0, past = ref
            z = track.position
            Pz = track.covariance[0:3, 0:3]
            for kind, hp in past.items():
                m, S = hp.at(t_ref - t0)
                lik[kind] = _gauss_logpdf(z, m, S + Pz)
        new_w = {}
        if lik:
            med = float(np.median(list(lik.values())))
            for h in hyps:
                lw = math.log(max(w_prev.get(h.kind, priors[h.kind]), 1e-6)) + 0.5 * lik.get(h.kind, med)
                new_w[h.kind] = lw
            mx = max(new_w.values())
            post = {k: math.exp(v - mx) for k, v in new_w.items()}
            tot = sum(post.values())
            post = {k: v / tot for k, v in post.items()}
        else:
            post = {h.kind: w_prev.get(h.kind, priors[h.kind]) for h in hyps}
            tot = sum(post.values())
            post = {k: v / tot for k, v in post.items()}
        for h in hyps:
            h.weight = self.inertia * post[h.kind] + (1 - self.inertia) * priors[h.kind]
        tot = sum(h.weight for h in hyps)
        for h in hyps:
            h.weight /= tot
        st["w"] = {h.kind: h.weight for h in hyps}
        if not st["past"] or t_ref - st["past"][-1][0] >= 0.9:
            st["past"].append((t_ref, {h.kind: h for h in hyps}))

    # --------------------------------------------------------- evaluation
    def _score_accuracy(self, track, t_ref: float, hyps: list[Hypothesis]) -> None:
        """Compare predictions made ``h`` seconds ago with the current estimate."""
        st = track.analytics.setdefault("_acc", {"pending": deque(maxlen=80), "err": {}})
        z = track.position
        keep = deque(maxlen=80)
        for item in st["pending"]:
            t0, hz, top, mix, cv, mixcov = item
            age = t_ref - t0
            if age >= hz - 0.5:
                if age <= hz + 1.5:
                    e_top = float(np.linalg.norm(top - z))
                    e_mix = float(np.linalg.norm(mix - z))
                    e_cv = float(np.linalg.norm(cv - z))
                    d = (z - mix)[:2]
                    inside = float(d @ np.linalg.solve(mixcov[:2, :2] + np.eye(2), d) <= 6.18)  # 2-sigma (95 %) region
                    self.acc[hz] += np.array([e_top, e_mix, e_cv, 1.0, inside])
                    st["err"][hz] = e_mix
                continue
            keep.append(item)
        st["pending"] = keep
        last = st.get("last_issue", -1e9)
        if t_ref - last >= 2.0:
            st["last_issue"] = t_ref
            top = max(hyps, key=lambda h: h.weight)
            cvh = next(h for h in hyps if h.kind == "cv")
            hzs = np.array([h for h in EVAL_HORIZONS if h <= self.horizon])
            mix, mixcov = mixture_at(hyps, hzs)
            topm, _ = top.at_many(hzs)
            cvm, _ = cvh.at_many(hzs)
            for k, hz in enumerate(hzs):
                st["pending"].append((t_ref, float(hz), topm[k], mix[k], cvm[k], mixcov[k]))

    def accuracy_summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for hz, a in self.acc.items():
            if a[3] > 0:
                out[f"{int(hz)}s"] = {"top": a[0] / a[3], "mixture": a[1] / a[3], "cv_baseline": a[2] / a[3],
                                      "coverage_2sigma": a[4] / a[3], "n": int(a[3])}
        return out


def mixture_at(hyps: list[Hypothesis], dt) -> tuple[np.ndarray, np.ndarray]:
    """Moment-matched Gaussian of the weighted hypothesis mixture at time offset(s) ``dt``."""
    scalar = np.isscalar(dt)
    dts = np.atleast_1d(np.asarray(dt, dtype=float))
    ws = np.array([h.weight for h in hyps])
    ws = ws / max(ws.sum(), 1e-12)
    M, S = zip(*(h.at_many(dts) for h in hyps))
    M = np.stack(M)  # (k, n, 3)
    S = np.stack(S)  # (k, n, 3, 3)
    mean = np.einsum("k,kna->na", ws, M)
    d = M - mean[None]
    cov = np.einsum("k,knab->nab", ws, S) + np.einsum("k,kna,knb->nab", ws, d, d)
    if scalar:
        return mean[0], cov[0]
    return mean, cov


UI_TIMES = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60], dtype=float)


def _pack(ts, m, S) -> list[list[float]]:
    out = []
    for t, mm, SS in zip(ts, m, S):
        out.append([round(float(t), 1), round(float(mm[0]), 1), round(float(mm[1]), 1), round(float(mm[2]), 1),
                    round(float(SS[0, 0]), 1), round(float(SS[0, 1]), 1), round(float(SS[1, 1]), 1), round(math.sqrt(max(SS[2, 2], 0)), 1)])
    return out


def serialize(hyps: list[Hypothesis], t_ref: float, prune: float, max_h: int, ui_times=UI_TIMES) -> dict:
    """Compact JSON representation for the dashboard / storage."""
    shown = sorted([h for h in hyps if h.weight >= prune], key=lambda h: -h.weight)[:max_h]
    ts = ui_times[ui_times <= hyps[0].times[-1] + 1e-9]
    out = []
    for h in shown:
        m, S = h.at_many(ts)
        out.append({"kind": h.kind, "label": LABELS.get(h.kind, h.kind), "note": h.note, "weight": round(h.weight, 3), "points": _pack(ts, m, S)})
    m, S = mixture_at(hyps, ts)
    return {"t_ref": round(t_ref, 2), "hypotheses": out, "mixture": _pack(ts, m, S), "discarded": len(hyps) - len(shown)}
