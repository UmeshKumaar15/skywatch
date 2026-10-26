"""Flight behaviours for simulated agents.

Each behaviour outputs a commanded ground velocity (ENU, m/s). The agent's
dynamics model then tracks that command subject to acceleration limits and
disturbances. A behaviour reports ``done`` when its leg is complete, so
missions can be composed as sequences of legs. Every leg carries the ground
truth *intent* label so intent inference can be evaluated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.types import Intent


def _toward(pos: np.ndarray, target: np.ndarray, speed: float, slow_radius: float = 30.0) -> np.ndarray:
    d = target - pos
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(3)
    s = speed * min(1.0, dist / max(slow_radius, 1e-3))
    return d / dist * s


class Behavior:
    intent: Intent = Intent.TRANSIT
    done: bool = False

    def command(self, agent, t: float, dt: float, world) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Waypoints(Behavior):
    points: list
    speed: float
    intent: Intent = Intent.TRANSIT
    tolerance: float = 15.0
    vertical_speed: float = 4.0
    hover_at_end: float = 0.0
    done: bool = False
    _i: int = 0
    _hover_left: Optional[float] = None

    def command(self, agent, t, dt, world):
        if self._i >= len(self.points):
            if self._hover_left is None:
                self._hover_left = self.hover_at_end
            self._hover_left -= dt
            if self._hover_left <= 0:
                self.done = True
            return np.zeros(3)
        target = np.asarray(self.points[self._i], dtype=float)
        pos = agent.pos
        if np.linalg.norm(target[:2] - pos[:2]) < self.tolerance and abs(target[2] - pos[2]) < max(self.tolerance, 8.0):
            self._i += 1
            return agent.vel * 0.5
        horiz = _toward(np.r_[pos[:2], 0.0], np.r_[target[:2], 0.0], self.speed, slow_radius=self.speed * 3)
        dz = target[2] - pos[2]
        horiz[2] = float(np.clip(dz * 0.5, -self.vertical_speed, self.vertical_speed))
        return horiz


@dataclass
class Hover(Behavior):
    duration: float
    intent: Intent = Intent.HOVERING
    done: bool = False
    _anchor: Optional[np.ndarray] = None
    _left: Optional[float] = None

    def command(self, agent, t, dt, world):
        if self._anchor is None:
            self._anchor = agent.pos.copy()
            self._left = self.duration
        self._left -= dt
        if self._left <= 0:
            self.done = True
        return (self._anchor - agent.pos) * 0.4


@dataclass
class Orbit(Behavior):
    """Circular orbit (inspection of a structure, or a loiter)."""

    center: np.ndarray
    radius: float
    speed: float
    altitude: float
    duration: float
    clockwise: bool = False
    intent: Intent = Intent.INSPECTION
    done: bool = False
    _left: Optional[float] = None

    def command(self, agent, t, dt, world):
        if self._left is None:
            self._left = self.duration
        self._left -= dt
        if self._left <= 0:
            self.done = True
        c = np.asarray(self.center, dtype=float)
        rel = agent.pos[:2] - c[:2]
        r = float(np.linalg.norm(rel)) or 1e-3
        radial = rel / r
        tangent = np.array([-radial[1], radial[0]]) * (-1 if self.clockwise else 1)
        # Blend tangential motion with radial correction toward the circle.
        v = tangent * self.speed + radial * np.clip(self.radius - r, -self.speed, self.speed) * 0.5
        vz = np.clip((self.altitude - agent.pos[2]) * 0.4, -3, 3)
        return np.array([v[0], v[1], vz])


@dataclass
class Racetrack(Behavior):
    """Holding pattern: two straight legs joined by semicircles."""

    center: np.ndarray
    heading_deg: float
    leg_length: float
    radius: float
    speed: float
    altitude: float
    duration: float
    intent: Intent = Intent.HOLDING_PATTERN
    done: bool = False
    _left: Optional[float] = None

    def command(self, agent, t, dt, world):
        if self._left is None:
            self._left = self.duration
        self._left -= dt
        if self._left <= 0:
            self.done = True
        h = math.radians(self.heading_deg)
        u = np.array([math.sin(h), math.cos(h)])  # along-leg unit vector
        n = np.array([u[1], -u[0]])
        c = np.asarray(self.center, dtype=float)[:2]
        rel = agent.pos[:2] - c
        a = float(rel @ u)
        half = self.leg_length / 2
        if abs(a) <= half:
            # On a straight leg: fly along u on the +n side, -u on the -n side.
            side = 1.0 if rel @ n >= 0 else -1.0
            target_line = side * self.radius
            v2 = u * self.speed * side + n * np.clip(target_line - rel @ n, -10, 10) * 0.3
        else:
            end = c + u * math.copysign(half, a)
            rr = agent.pos[:2] - end
            r = float(np.linalg.norm(rr)) or 1e-3
            radial = rr / r
            tangent = np.array([radial[1], -radial[0]])
            v2 = tangent * self.speed + radial * np.clip(self.radius - r, -10, 10) * 0.3
        vz = np.clip((self.altitude - agent.pos[2]) * 0.2, -5, 5)
        return np.array([v2[0], v2[1], vz])


@dataclass
class Lawnmower(Behavior):
    """Survey / mapping pattern over a rectangle."""

    origin: np.ndarray  # (x, y) of the area corner
    width: float
    height: float
    spacing: float
    speed: float
    altitude: float
    intent: Intent = Intent.MAPPING
    done: bool = False
    _wp: Optional[Waypoints] = None

    def command(self, agent, t, dt, world):
        if self._wp is None:
            pts = []
            x0, y0 = float(self.origin[0]), float(self.origin[1])
            n = max(2, int(self.width / self.spacing) + 1)
            for i in range(n):
                x = x0 + i * self.spacing
                ys = (y0, y0 + self.height) if i % 2 == 0 else (y0 + self.height, y0)
                pts.append((x, ys[0], self.altitude))
                pts.append((x, ys[1], self.altitude))
            self._wp = Waypoints(pts, self.speed, intent=self.intent, tolerance=8.0)
        v = self._wp.command(agent, t, dt, world)
        self.done = self._wp.done
        return v


@dataclass
class Wander(Behavior):
    """Recreational flying: random targets in an area, speed changes, hovers."""

    center: np.ndarray
    radius: float
    speed_range: tuple
    alt_range: tuple
    duration: float
    hover_prob: float = 0.25
    intent: Intent = Intent.RECREATIONAL
    done: bool = False
    _left: Optional[float] = None
    _target: Optional[np.ndarray] = None
    _speed: float = 5.0
    _hover: float = 0.0

    def command(self, agent, t, dt, world):
        rng = world.rng
        if self._left is None:
            self._left = self.duration
        self._left -= dt
        if self._left <= 0:
            self.done = True
        if self._hover > 0:
            self._hover -= dt
            return -agent.vel * 0.5
        if self._target is None or np.linalg.norm(self._target - agent.pos) < 10:
            if self._target is not None and rng.random() < self.hover_prob:
                self._hover = rng.uniform(3, 15)
            ang = rng.uniform(0, 2 * math.pi)
            r = self.radius * math.sqrt(rng.random())
            c = np.asarray(self.center, dtype=float)
            self._target = np.array([c[0] + r * math.cos(ang), c[1] + r * math.sin(ang), rng.uniform(*self.alt_range)])
            self._speed = rng.uniform(*self.speed_range)
        return _toward(agent.pos, self._target, self._speed, slow_radius=8)


@dataclass
class BirdFlight(Behavior):
    """Correlated random walk with occasional thermal soaring circles."""

    speed: float
    alt_range: tuple
    duration: float
    intent: Intent = Intent.TRANSIT
    done: bool = False
    _left: Optional[float] = None
    _heading: Optional[float] = None
    _soar: float = 0.0
    _turn: float = 0.0

    def command(self, agent, t, dt, world):
        rng = world.rng
        if self._left is None:
            self._left = self.duration
            self._heading = math.atan2(agent.vel[1], agent.vel[0]) if np.linalg.norm(agent.vel) > 0.1 else rng.uniform(0, 2 * math.pi)
        self._left -= dt
        if self._left <= 0:
            self.done = True
        if self._soar > 0:
            self._soar -= dt
            self._heading += self._turn * dt
        else:
            self._heading += rng.normal(0, 0.35) * math.sqrt(dt)
            if rng.random() < 0.004:
                self._soar = rng.uniform(15, 40)
                self._turn = rng.choice([-1, 1]) * rng.uniform(0.25, 0.45)
        s = self.speed * (1 + 0.15 * math.sin(t * 0.7 + agent.phase))
        lo, hi = self.alt_range
        vz = rng.normal(0, 0.8) + (lo - agent.pos[2]) * 0.1 * (agent.pos[2] < lo) + (hi - agent.pos[2]) * 0.1 * (agent.pos[2] > hi)
        return np.array([s * math.cos(self._heading), s * math.sin(self._heading), vz])


@dataclass
class Drift(Behavior):
    """Passive drift with the wind plus constant ascent (weather balloon)."""

    ascent_rate: float
    duration: float
    intent: Intent = Intent.DRIFTING
    done: bool = False
    _left: Optional[float] = None

    def command(self, agent, t, dt, world):
        if self._left is None:
            self._left = self.duration
        self._left -= dt
        if self._left <= 0:
            self.done = True
        w = world.wind_at(agent.pos)
        return np.array([w[0], w[1], self.ascent_rate])


@dataclass
class Erratic(Behavior):
    """Abrupt manoeuvring: random hard accelerations (anomaly injection)."""

    duration: float
    max_speed: float
    intent: Intent = Intent.ANOMALOUS
    done: bool = False
    _left: Optional[float] = None
    _cmd: Optional[np.ndarray] = None
    _switch: float = 0.0

    def command(self, agent, t, dt, world):
        rng = world.rng
        if self._left is None:
            self._left = self.duration
        self._left -= dt
        self._switch -= dt
        if self._left <= 0:
            self.done = True
        if self._cmd is None or self._switch <= 0:
            ang = rng.uniform(0, 2 * math.pi)
            s = rng.uniform(0.5, 1.0) * self.max_speed
            self._cmd = np.array([s * math.cos(ang), s * math.sin(ang), rng.uniform(-4, 4)])
            self._switch = rng.uniform(1.5, 4.0)
        return self._cmd
