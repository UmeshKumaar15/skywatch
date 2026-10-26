"""Interacting Multiple Model (IMM) extended Kalman filter.

Shared 10-D state: [x, y, z, vx, vy, vz, ax, ay, az, omega]
  * CV - constant velocity            (smooth cruise)
  * CA - constant acceleration        (speed changes, climbs, stops)
  * CT - coordinated horizontal turn  (turn rate omega estimated online)

Mode probabilities give an interpretable "how is it moving right now" signal
that the trajectory predictor uses as prior hypothesis weights.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

NX = 10
MODELS = ("cv", "ca", "ct")
POS = slice(0, 3)
VEL = slice(3, 6)
ACC = slice(6, 9)


def _cv_f(x: np.ndarray, T: float) -> np.ndarray:
    y = x.copy()
    y[POS] = x[POS] + x[VEL] * T
    y[ACC] = 0.0
    y[9] = 0.0
    return y


def _ca_f(x: np.ndarray, T: float) -> np.ndarray:
    y = x.copy()
    y[POS] = x[POS] + x[VEL] * T + 0.5 * x[ACC] * T * T
    y[VEL] = x[VEL] + x[ACC] * T
    y[9] = 0.0
    return y


def _ct_f(x: np.ndarray, T: float) -> np.ndarray:
    y = x.copy()
    w = x[9]
    vx, vy = x[3], x[4]
    if abs(w) < 1e-5:
        s, c1 = T, 0.5 * w * T * T  # sin(wT)/w, (1-cos wT)/w  (series)
        cw, sw = 1.0, w * T
    else:
        sw, cw = math.sin(w * T), math.cos(w * T)
        s, c1 = sw / w, (1 - cw) / w
    y[0] = x[0] + s * vx - c1 * vy
    y[1] = x[1] + c1 * vx + s * vy
    y[2] = x[2] + x[5] * T
    y[3] = cw * vx - sw * vy
    y[4] = sw * vx + cw * vy
    y[ACC] = 0.0
    return y


TRANSITIONS = {"cv": _cv_f, "ca": _ca_f, "ct": _ct_f}


def _linear_F(model: str, T: float) -> np.ndarray:
    F = np.eye(NX)
    F[0:3, 3:6] = np.eye(3) * T
    if model == "ca":
        F[0:3, 6:9] = np.eye(3) * 0.5 * T * T
        F[3:6, 6:9] = np.eye(3) * T
    else:
        F[6:9, 6:9] = 0.0
    F[9, 9] = 0.0
    return F


def numerical_jacobian(f, x: np.ndarray, T: float) -> np.ndarray:
    if f is _cv_f:
        return _linear_F("cv", T)
    if f is _ca_f:
        return _linear_F("ca", T)
    J = np.empty((NX, NX))
    fx = f(x, T)
    for i in range(NX):
        h = 1e-4 * max(1.0, abs(x[i]))
        dx = x.copy()
        dx[i] += h
        J[:, i] = (f(dx, T) - fx) / h
    return J


def _wna_block(T: float, q: float) -> np.ndarray:
    """Discrete white-noise acceleration covariance for one axis (pos, vel)."""
    return q**2 * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])


def _wnj_block(T: float, q: float) -> np.ndarray:
    """Discrete white-noise jerk covariance for one axis (pos, vel, acc)."""
    return q**2 * np.array(
        [[T**5 / 20, T**4 / 8, T**3 / 6], [T**4 / 8, T**3 / 3, T**2 / 2], [T**3 / 6, T**2 / 2, T]]
    )


_I3 = np.arange(3)


def process_noise(model: str, T: float, q: dict[str, float]) -> np.ndarray:
    Q = np.zeros((NX, NX))
    if model == "ca":
        b = _wnj_block(T, q["ca"])
        for r in range(3):
            for c in range(3):
                Q[_I3 + 3 * r, _I3 + 3 * c] = b[r, c]
        Q[9, 9] = 1e-4
        return Q
    qa = (q["cv"] if model == "cv" else q["ct"]) ** 2 * np.array([1.0, 1.0, 0.49])
    Q[_I3, _I3] = qa * T**4 / 4
    Q[_I3, _I3 + 3] = Q[_I3 + 3, _I3] = qa * T**3 / 2
    Q[_I3 + 3, _I3 + 3] = qa * T**2
    # Unused acceleration / turn-rate states are reset to zero with a
    # nominal spread so that mixing into CA / CT remains well conditioned.
    Q[_I3 + 6, _I3 + 6] = 1.0 if model == "cv" else 0.5
    Q[9, 9] = (q["turn_rate"] ** 2) * T if model == "ct" else 0.01
    return Q


class IMMFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, q: dict[str, float], p_stay: float = 0.95) -> None:
        self.q = q
        m = len(MODELS)
        self.x = np.tile(x0, (m, 1)).astype(float)
        self.P = np.tile(P0, (m, 1, 1)).astype(float)
        self.mu = np.array([0.6, 0.2, 0.2])
        off = (1 - p_stay) / (m - 1)
        self.PI = np.full((m, m), off) + np.eye(m) * (p_stay - off)
        self.last_likelihood = np.ones(m)
        self.last_nis = 0.0
        self._cache: Optional[tuple[np.ndarray, np.ndarray]] = None

    # -------------------------------------------------------------- combined
    def _combined(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is None:
            xc = self.mu @ self.x
            d = self.x - xc
            P = np.einsum("j,jab->ab", self.mu, self.P) + np.einsum("j,ja,jb->ab", self.mu, d, d)
            self._cache = (xc, P)
        return self._cache

    @property
    def state(self) -> np.ndarray:
        return self._combined()[0]

    @property
    def covariance(self) -> np.ndarray:
        return self._combined()[1]

    def mode_probabilities(self) -> dict[str, float]:
        return {m: float(p) for m, p in zip(MODELS, self.mu)}

    # ------------------------------------------------------------------ steps
    def predict(self, T: float) -> None:
        if T <= 1e-6:
            return
        self._cache = None
        m = len(MODELS)
        # Interaction / mixing
        cbar = self.PI.T @ self.mu
        cbar = np.maximum(cbar, 1e-12)
        w = (self.PI * self.mu[:, None]) / cbar[None, :]  # w[i, j] = P(i | j)
        xm = w.T @ self.x
        d = self.x[:, None, :] - xm[None, :, :]  # d[i, j] = x_i - xm_j
        Pm = np.einsum("ij,iab->jab", w, self.P) + np.einsum("ij,ija,ijb->jab", w, d, d)
        self.mu = cbar
        # Model-conditioned prediction
        for j, name in enumerate(MODELS):
            f = TRANSITIONS[name]
            F = numerical_jacobian(f, xm[j], T)
            self.x[j] = f(xm[j], T)
            self.P[j] = F @ Pm[j] @ F.T + process_noise(name, T, self.q)
            self.P[j] = 0.5 * (self.P[j] + self.P[j].T)

    def innovation(self, z: np.ndarray, R: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mixture-level innovation and covariance used for gating / association."""
        xc = self.state
        S = H @ self.covariance @ H.T + R
        return z - H @ xc, S

    def update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray) -> float:
        self._cache = None
        m = len(MODELS)
        lik = np.zeros(m)
        for j in range(m):
            y = z - H @ self.x[j]
            S = H @ self.P[j] @ H.T + R
            Si = np.linalg.inv(S)
            K = self.P[j] @ H.T @ Si
            self.x[j] = self.x[j] + K @ y
            IKH = np.eye(NX) - K @ H
            self.P[j] = IKH @ self.P[j] @ IKH.T + K @ R @ K.T  # Joseph form
            d2 = float(y @ Si @ y)
            _, logdet = np.linalg.slogdet(S)
            lik[j] = math.exp(-0.5 * min(d2, 200.0) - 0.5 * logdet - 0.5 * len(z) * math.log(2 * math.pi))
        c = self.mu * lik
        tot = c.sum()
        if tot > 1e-300:
            self.mu = c / tot
        self.mu = np.clip(self.mu, 1e-4, None)
        self.mu /= self.mu.sum()
        self.last_likelihood = lik
        return tot

    # ---------------------------------------------------------- propagation
    def propagate(self, model: str, T: float, x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None,
                  q_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Open-loop prediction of a single model (used by the trajectory predictor)."""
        j = MODELS.index(model)
        x = self.x[j] if x is None else x
        P = self.P[j] if P is None else P
        f = TRANSITIONS[model]
        F = numerical_jacobian(f, x, T)
        q = {k: v * q_scale for k, v in self.q.items()}
        return f(x, T), F @ P @ F.T + process_noise(model, T, q)


def measurement_matrices(has_velocity: bool) -> np.ndarray:
    H = np.zeros((6 if has_velocity else 3, NX))
    H[0:3, 0:3] = np.eye(3)
    if has_velocity:
        H[3:6, 3:6] = np.eye(3)
    return H
