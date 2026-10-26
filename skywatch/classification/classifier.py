"""Probabilistic object classification with recursive Bayesian smoothing.

The learned model (an MLP trained on simulated tracks) outputs a calibrated
distribution p(class | features). Because consecutive feature windows
overlap heavily, they are not independent evidence; the track posterior is
therefore updated with a tempered (geometric) Bayesian update that is robust
to that correlation. Out-of-distribution feature vectors shift probability
mass toward ``unknown`` rather than producing an over-confident label.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.types import OBJECT_CLASSES, ObjectClass
from .features import FEATURE_NAMES, extract
from .nn import MLP

N_CLS = len(OBJECT_CLASSES)
UNKNOWN_IDX = OBJECT_CLASSES.index(ObjectClass.UNKNOWN)


class ObjectClassifier:
    def __init__(self, cfg: dict, model_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.alpha = float(cfg.get("smoothing", 0.15))
        self.unknown_threshold = float(cfg.get("unknown_threshold", 0.4))
        self.model: Optional[MLP] = None
        self.ood_mean: Optional[np.ndarray] = None
        self.ood_std: Optional[np.ndarray] = None
        self.ood_scale = 1.0
        self.class_stats: dict = {}
        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self, path) -> None:
        d = np.load(path, allow_pickle=False)
        self.model = MLP.from_state(d, "clf_")
        self.ood_mean = d["ood_mean"]
        self.ood_std = d["ood_std"]
        self.ood_scale = float(d["ood_scale"])

    # ---------------------------------------------------------------- infer
    def likelihood(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Model class distribution and out-of-distribution score (0..1)."""
        if self.model is None:
            return self._heuristic(x), 0.0
        p = self.model.predict_proba(x[None, :])[0]
        z = (x - self.ood_mean) / self.ood_std
        d = float(np.sqrt(np.mean(np.minimum(z**2, 100.0))))
        ood = 1.0 / (1.0 + math.exp(-(d - self.ood_scale) * 3.0))
        return p, ood

    def _heuristic(self, x: np.ndarray) -> np.ndarray:
        """Physics-only fallback used before a model has been trained."""
        f = dict(zip(FEATURE_NAMES, x))
        speed = math.expm1(f["log_speed_mean"])
        alt = math.expm1(f["log_alt_mean"])
        s = np.full(N_CLS, 0.2)
        idx = {c: i for i, c in enumerate(OBJECT_CLASSES)}
        if speed > 100 or alt > 3000:
            s[idx[ObjectClass.COMMERCIAL_AIRCRAFT]] += 3
        if 30 < speed <= 100:
            s[idx[ObjectClass.HELICOPTER]] += 2
            s[idx[ObjectClass.EMERGENCY_VEHICLE]] += 1 + 3 * f["priority_status"]
        if speed <= 25 and f["flap_mean"] > 0.6:
            s[idx[ObjectClass.BIRD]] += 2
        if speed <= 25 and f["rotor_mean"] > 0.6:
            s[idx[ObjectClass.DELIVERY_DRONE]] += 1
            s[idx[ObjectClass.RECREATIONAL_DRONE]] += 1
            s[idx[ObjectClass.INDUSTRIAL_DRONE]] += 1
        if f["vz_mean"] > 3 and speed < 20:
            s[idx[ObjectClass.WEATHER_BALLOON]] += 2
        return s / s.sum()

    def update(self, track, now: float) -> dict:
        """Update the track's class posterior and return the analytics dict."""
        x = extract(track, now)
        p, ood = self.likelihood(x)
        prev = track.analytics.get("_cls_log")
        logp = np.log(np.clip(p, 1e-6, 1.0))
        if prev is None:
            post_log = logp
        else:
            # Tempered recursive update; faster adaptation early in the track's life.
            age = now - track.created
            a = max(self.alpha, 1.0 / (1.0 + age / 2.0))
            post_log = (1 - a) * prev + a * logp
        track.analytics["_cls_log"] = post_log
        post = np.exp(post_log - post_log.max())
        post /= post.sum()
        # Out-of-distribution evidence moves mass toward 'unknown'.
        if ood > 0.5:
            moved = post * (ood - 0.5)
            post = post - moved
            post[UNKNOWN_IDX] += moved.sum()
        best = int(np.argmax(post))
        label = OBJECT_CLASSES[best]
        if post[best] < self.unknown_threshold:
            label = ObjectClass.UNKNOWN
        entropy = float(-(post * np.log(post + 1e-12)).sum() / math.log(N_CLS))
        return {
            "label": label.value,
            "probs": {c.value: float(v) for c, v in zip(OBJECT_CLASSES, post)},
            "confidence": float(post[best]) * (1 - 0.5 * ood),
            "entropy": entropy,
            "ood": ood,
            "features": x,
        }
