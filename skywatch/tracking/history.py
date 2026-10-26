"""Helpers to turn irregular multi-sensor track histories into regular series."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HistoryPoint:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    pos_sigma: float
    sensors: str = ""


class TrackHistory:
    """Append-only per-track history backed by NumPy arrays.

    Keeps the most recent ``maxlen`` points; time-window queries are binary
    searches instead of scans, which matters with hundreds of tracks.
    Iteration / indexing yield ``HistoryPoint`` objects for convenience.
    """

    def __init__(self, maxlen: int = 2400) -> None:
        self.maxlen = maxlen
        cap = 256
        self._t = np.empty(cap)
        self._p = np.empty((cap, 3))
        self._v = np.empty((cap, 3))
        self._s = np.empty(cap)
        self._src: list[str] = []
        self._a = 0  # start index of live window
        self._b = 0  # end index (exclusive)

    def append(self, t: float, pos: np.ndarray, vel: np.ndarray, sigma: float, src: str = "") -> None:
        if self._b == len(self._t):
            n = self._b - self._a
            if len(self._t) >= 2 * self.maxlen:  # compact: move live window to the front
                for arr in (self._t, self._p, self._v, self._s):
                    arr[:n] = arr[self._a:self._b]
                self._src = self._src[self._a:self._b]
            else:  # grow
                cap = len(self._t) * 2
                self._t = np.r_[self._t[self._a:self._b], np.empty(cap - n)]
                self._p = np.vstack([self._p[self._a:self._b], np.empty((cap - n, 3))])
                self._v = np.vstack([self._v[self._a:self._b], np.empty((cap - n, 3))])
                self._s = np.r_[self._s[self._a:self._b], np.empty(cap - n)]
                self._src = self._src[self._a:self._b]
            self._a, self._b = 0, n
        i = self._b
        self._t[i], self._p[i], self._v[i], self._s[i] = t, pos, vel, sigma
        self._src.append(src)
        self._b += 1
        if self._b - self._a > self.maxlen:
            self._a += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._t[self._a:self._b], self._p[self._a:self._b], self._v[self._a:self._b]

    def __len__(self) -> int:
        return self._b - self._a

    def __getitem__(self, k: int) -> HistoryPoint:
        n = len(self)
        if k < 0:
            k += n
        if not 0 <= k < n:
            raise IndexError(k)
        i = self._a + k
        return HistoryPoint(float(self._t[i]), self._p[i].copy(), self._v[i].copy(), float(self._s[i]), self._src[i])

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]


@dataclass
class Series:
    t: np.ndarray  # (n,)
    pos: np.ndarray  # (n, 3)
    vel: np.ndarray  # (n, 3)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel[:, :2], axis=1)

    @property
    def heading(self) -> np.ndarray:
        return np.arctan2(self.vel[:, 1], self.vel[:, 0])


def resample(history, now: float, window: float, dt: float = 1.0) -> Series:
    """Linearly interpolate a track history onto a regular grid ending at ``now``.

    ``history`` is a ``TrackHistory`` (fast path) or an iterable of objects with
    ``t``, ``pos`` and ``vel``. Points before the start of the history are not
    extrapolated.
    """
    if isinstance(history, TrackHistory):
        T, Pa, Va = history.arrays()
        i0 = int(np.searchsorted(T, now - window - dt))
        ts, P, V = T[i0:], Pa[i0:], Va[i0:]
    else:
        pts = [h for h in history if h.t >= now - window - dt]
        ts = np.array([h.t for h in pts])
        P = np.array([h.pos for h in pts]).reshape(-1, 3)
        V = np.array([h.vel for h in pts]).reshape(-1, 3)
    if len(ts) < 2:
        if len(ts):
            return Series(ts[-1:].copy(), P[-1:].copy(), V[-1:].copy())
        return Series(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    t0 = max(ts[0], now - window)
    grid = np.arange(now, t0 - 1e-9, -dt)[::-1]
    grid = grid[grid <= ts[-1] + dt * 3]  # don't invent data far beyond the last update
    if len(grid) == 0:
        grid = np.array([ts[-1]])
    pos = np.stack([np.interp(grid, ts, P[:, k]) for k in range(3)], axis=1)
    vel = np.stack([np.interp(grid, ts, V[:, k]) for k in range(3)], axis=1)
    return Series(grid, pos, vel)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def turn_rates(s: Series, min_speed: float = 1.0) -> np.ndarray:
    if len(s) < 2:
        return np.zeros(0)
    h = s.heading
    dh = wrap_angle(np.diff(h)) / np.maximum(np.diff(s.t), 1e-6)
    moving = (s.speed[1:] > min_speed) & (s.speed[:-1] > min_speed)
    return dh[moving]


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic (Kasa) circle fit. Returns centre, radius, RMS residual."""
    if len(xy) < 5:
        return xy.mean(axis=0) if len(xy) else np.zeros(2), math.inf, math.inf
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return xy.mean(axis=0), math.inf, math.inf
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    resid = float(np.sqrt(np.mean((np.linalg.norm(xy - c, axis=1) - r) ** 2)))
    return c, r, resid


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from p to segment ab and the projection parameter in [0, 1]."""
    ab = b - a
    L2 = float(ab @ ab)
    u = 0.0 if L2 < 1e-9 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab))), u


_SEG_CACHE: dict[int, tuple] = {}


def _segments(polylines: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = id(polylines)
    hit = _SEG_CACHE.get(key)
    if hit is not None and hit[0] is polylines:
        return hit[1]
    A, B, L, S = [], [], [], []
    for li, line in enumerate(polylines):
        for si in range(len(line) - 1):
            A.append(line[si][:2])
            B.append(line[si + 1][:2])
            L.append(li)
            S.append(si)
    arrs = (np.asarray(A, float).reshape(-1, 2), np.asarray(B, float).reshape(-1, 2), np.asarray(L), np.asarray(S))
    _SEG_CACHE[key] = (polylines, arrs)
    return arrs


def distance_to_polylines(p: np.ndarray, polylines: list) -> tuple[float, int, int, float]:
    """Min horizontal distance to any polyline -> (distance, line idx, segment idx, u)."""
    A, B, L, S = _segments(polylines)
    if len(A) == 0:
        return math.inf, -1, -1, 0.0
    AB = B - A
    L2 = np.maximum((AB * AB).sum(axis=1), 1e-9)
    u = np.clip(((p[:2] - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
    d = np.linalg.norm(p[:2] - (A + u[:, None] * AB), axis=1)
    k = int(np.argmin(d))
    return float(d[k]), int(L[k]), int(S[k]), float(u[k])


def polyline_distances(points: np.ndarray, polylines: list) -> np.ndarray:
    """Horizontal distance from each of ``points`` (n, >=2) to the nearest polyline."""
    A, B, _, _ = _segments(polylines)
    if len(A) == 0:
        return np.full(len(points), math.inf)
    P = np.asarray(points, float)[:, None, :2]  # (n, 1, 2)
    AB = (B - A)[None]
    L2 = np.maximum((AB * AB).sum(-1), 1e-9)
    u = np.clip(((P - A[None]) * AB).sum(-1) / L2, 0.0, 1.0)
    d = np.linalg.norm(P - (A[None] + u[..., None] * AB), axis=-1)
    return d.min(axis=1)
