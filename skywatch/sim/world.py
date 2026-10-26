"""Simulated airspace: agents, missions, wind, disturbances and anomaly injection.

Everything random is drawn from a single ``numpy.random.Generator`` seeded
from the scenario, so a (scenario, seed) pair reproduces exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.types import Intent, ObjectClass
from . import behaviors as B

C = ObjectClass

# Physical / signature profile per object class. ``rcs`` in dBsm, ``rotor`` and
# ``flap`` are the nominal micro-Doppler cue strengths a radar would observe.
PROFILES: dict[ObjectClass, dict[str, float]] = {
    C.DELIVERY_DRONE: dict(max_acc=4.0, max_speed=20, tau=1.0, rcs=-13, rotor=0.80, flap=0.10, size=1.3, gust=0.6),
    C.RECREATIONAL_DRONE: dict(max_acc=7.0, max_speed=22, tau=0.6, rcs=-19, rotor=0.85, flap=0.10, size=0.4, gust=0.8),
    C.INDUSTRIAL_DRONE: dict(max_acc=4.0, max_speed=16, tau=0.8, rcs=-15, rotor=0.85, flap=0.08, size=1.0, gust=0.5),
    C.BIRD: dict(max_acc=5.0, max_speed=18, tau=0.8, rcs=-24, rotor=0.12, flap=0.85, size=0.5, gust=1.0),
    C.HELICOPTER: dict(max_acc=3.0, max_speed=75, tau=2.0, rcs=9, rotor=0.95, flap=0.05, size=13, gust=0.4),
    C.COMMERCIAL_AIRCRAFT: dict(max_acc=2.0, max_speed=260, tau=4.0, rcs=24, rotor=0.15, flap=0.02, size=40, gust=0.3),
    C.EMERGENCY_VEHICLE: dict(max_acc=3.5, max_speed=80, tau=1.8, rcs=8, rotor=0.95, flap=0.05, size=12, gust=0.4),
    C.WEATHER_BALLOON: dict(max_acc=1.0, max_speed=40, tau=3.0, rcs=-6, rotor=0.02, flap=0.05, size=2.0, gust=0.6),
    C.UNKNOWN: dict(max_acc=3.0, max_speed=15, tau=1.5, rcs=-10, rotor=0.45, flap=0.45, size=1.5, gust=1.0),
}


@dataclass
class Geography:
    radius: float
    depots: list
    destinations: list
    corridors: list  # list of polylines [[x, y], ...]
    inspection_sites: list
    mapping_areas: list  # [x, y, w, h]
    parks: list  # [x, y, r]
    hospital: list
    met_station: list
    airport: list  # runway threshold [x, y] + heading
    nofly: list  # [x, y, r, name]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @staticmethod
    def generate(rng: np.random.Generator, radius: float) -> "Geography":
        def rpoint(rmin, rmax):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(rmin, rmax)
            return [round(r * math.cos(a), 1), round(r * math.sin(a), 1)]

        depots = [rpoint(300, 1800) for _ in range(2)]
        destinations = [rpoint(800, radius * 0.6) for _ in range(8)]
        corridors = []
        for i, dst in enumerate(destinations):
            d = depots[i % len(depots)]
            # A dog-leg corridor: depot -> intermediate -> destination.
            mid = [(d[0] + dst[0]) / 2 + rng.uniform(-300, 300), (d[1] + dst[1]) / 2 + rng.uniform(-300, 300)]
            corridors.append([d, [round(mid[0], 1), round(mid[1], 1)], dst])
        ang = rng.uniform(0, 2 * math.pi)
        parks = [rpoint(600, radius * 0.45) + [350.0] for _ in range(2)]

        def clear_of_routes(c, r):
            from ..tracking.history import distance_to_polylines
            p = np.array([c[0], c[1], 0.0])
            if distance_to_polylines(p, corridors)[0] < r + 150:
                return False
            return all(math.hypot(c[0] - pk[0], c[1] - pk[1]) > r + pk[2] + 100 for pk in parks)

        nofly = []
        for name, r in (("stadium", 400.0), ("power plant", 300.0)):
            for _ in range(200):
                c = rpoint(1000, radius * 0.7)
                if clear_of_routes(c, r):
                    break
            nofly.append(c + [r, name])
        return Geography(
            radius=radius,
            depots=depots,
            destinations=destinations,
            corridors=corridors,
            inspection_sites=[rpoint(500, radius * 0.55) for _ in range(3)],
            mapping_areas=[rpoint(800, radius * 0.5) + [500.0, 350.0] for _ in range(2)],
            parks=parks,
            hospital=rpoint(1000, radius * 0.5),
            met_station=rpoint(1500, radius * 0.6),
            airport=[round((radius + 3000) * math.cos(ang), 1), round((radius + 3000) * math.sin(ang), 1), round(math.degrees(ang), 1)],
            nofly=nofly,
        )


@dataclass
class Agent:
    id: str
    cls: ObjectClass
    pos: np.ndarray
    vel: np.ndarray
    mission: list
    profile: dict
    adsb_id: Optional[str] = None
    rid_id: Optional[str] = None
    transponder_on: bool = True
    priority_status: int = 0
    phase: float = 0.0
    born: float = 0.0
    alive: bool = True
    gust: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anomalies: list = field(default_factory=list)  # [kind, t_start, t_end|None]
    scheduled: list = field(default_factory=list)  # [(t, kind, params)]
    route_corridor: Optional[int] = None
    rcs_offset: float = 0.0

    @property
    def intent(self) -> Intent:
        return self.mission[0].intent if self.mission else Intent.TRANSIT

    def active_anomaly(self, t: float) -> Optional[str]:
        for kind, t0, t1 in self.anomalies:
            if t >= t0 and (t1 is None or t <= t1):
                return kind
        return None

    def step(self, t: float, dt: float, world: "World") -> None:
        # Scheduled anomaly injections
        for item in list(self.scheduled):
            if t >= item[0]:
                self.scheduled.remove(item)
                world.inject_anomaly(self, item[1], t)
        while self.mission and self.mission[0].done:
            done = self.mission.pop(0)
            kind = getattr(done, "anomaly_kind", None)
            if kind:
                for a in self.anomalies:
                    if a[0] == kind and a[2] is None:
                        a[2] = t
        if not self.mission:
            self.alive = False
            return
        cmd = np.asarray(self.mission[0].command(self, t, dt, world), dtype=float)
        sp = float(np.linalg.norm(cmd))
        if sp > self.profile["max_speed"]:
            cmd *= self.profile["max_speed"] / sp
        acc = (cmd - self.vel) / self.profile["tau"]
        an = float(np.linalg.norm(acc))
        lim = self.profile["max_acc"] * (3.0 if self.active_anomaly(t) == "erratic" else 1.0)
        if an > lim:
            acc *= lim / an
        # Ornstein-Uhlenbeck gust disturbance on velocity
        g = self.profile["gust"] * world.gust_intensity
        self.gust += -self.gust * dt / 3.0 + world.rng.normal(0, g, 3) * math.sqrt(dt) * np.array([1, 1, 0.4])
        self.acc = acc
        self.vel = self.vel + acc * dt
        self.pos = self.pos + (self.vel + self.gust) * dt
        if self.pos[2] < 0:
            self.pos[2] = 0.0
            self.vel[2] = max(self.vel[2], 0.0)


class World:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.seed = int(scenario.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.radius = float(scenario.get("radius", 6000))
        self.t = 0.0
        self.wind = np.array(scenario.get("wind", [3.0, 1.5]) + [0.0], dtype=float)[:3]
        self.gust_intensity = float(scenario.get("gust_intensity", 1.0))
        geo = scenario.get("geography")
        self.geo = Geography(**geo) if geo else Geography.generate(self.rng, self.radius)
        self.agents: dict[str, Agent] = {}
        self.retired: list[Agent] = []
        self._next_id = 1
        self.spawn_rates = {C(k): float(v) for k, v in scenario.get("spawn_rates_per_min", {}).items()}
        self.max_agents = int(scenario.get("max_agents", 60))
        self.anomaly_rate = float(scenario.get("anomaly_rate_per_min", 0.4))
        for cls_name, n in scenario.get("initial", {}).items():
            for _ in range(int(n)):
                self.spawn(C(cls_name), initial=True)
        for f in scenario.get("formations", []):
            self.spawn_formation(int(f.get("count", 5)), float(f.get("spacing", 20)))

    # ------------------------------------------------------------------ wind
    def wind_at(self, pos: np.ndarray) -> np.ndarray:
        shear = 1.0 + min(pos[2], 12000.0) / 1500.0
        return self.wind * shear

    # -------------------------------------------------------------- spawning
    def _new_id(self, prefix: str) -> str:
        i = self._next_id
        self._next_id += 1
        return f"{prefix}{i:04d}"

    def _edge_point(self, alt: float, inward_jitter: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        """Point on the zone boundary and a unit direction pointing across the zone."""
        a = self.rng.uniform(0, 2 * math.pi)
        p = np.array([self.radius * math.cos(a), self.radius * math.sin(a), alt])
        inward = math.atan2(-p[1], -p[0]) + self.rng.uniform(-inward_jitter, inward_jitter)
        return p, np.array([math.cos(inward), math.sin(inward), 0.0])

    def spawn(self, cls: ObjectClass, initial: bool = False) -> Optional[Agent]:
        rng, geo = self.rng, self.geo
        prof = dict(PROFILES[cls])
        aid = self._new_id("O")
        vel = np.zeros(3)
        adsb = rid = None
        prio = 0
        mission: list = []
        corridor = None
        if cls == C.DELIVERY_DRONE:
            corridor = int(rng.integers(len(geo.corridors)))
            route = geo.corridors[corridor]
            cruise = rng.uniform(60, 110)
            spd = rng.uniform(12, 17)
            pts = [(p[0], p[1], cruise) for p in route]
            dst = route[-1]
            outbound = B.Waypoints(pts, spd, intent=Intent.DELIVERY_ROUTE)
            drop = B.Waypoints([(dst[0], dst[1], 25.0)], 4, intent=Intent.DELIVERY_ROUTE, hover_at_end=rng.uniform(8, 20))
            back = B.Waypoints([(p[0], p[1], cruise + 10) for p in reversed(route)] + [(route[0][0], route[0][1], 3.0)], spd, intent=Intent.DELIVERY_ROUTE)
            mission = [outbound, drop, back]
            start = route[0]
            pos = np.array([start[0], start[1], 5.0])
            if initial:  # start somewhere along the outbound leg
                f = rng.uniform(0.0, 0.8)
                pos = np.array([start[0] + f * (dst[0] - start[0]), start[1] + f * (dst[1] - start[1]), cruise])
                outbound.points = [pts[-1]]
            rid = f"RID-{aid}"
        elif cls == C.RECREATIONAL_DRONE:
            park = geo.parks[int(rng.integers(len(geo.parks)))]
            pos = np.array([park[0] + rng.normal(0, 80), park[1] + rng.normal(0, 80), 2.0 if not initial else rng.uniform(20, 80)])
            mission = [B.Wander(np.array(park[:2]), park[2], (3, 16), (15, 110), rng.uniform(240, 700))]
            mission.append(B.Waypoints([(pos[0], pos[1], 1.0)], 6, intent=Intent.RECREATIONAL))
            if rng.random() < 0.5:
                rid = f"RID-{aid}"
        elif cls == C.INDUSTRIAL_DRONE:
            if rng.random() < 0.5:
                site = geo.inspection_sites[int(rng.integers(len(geo.inspection_sites)))]
                pos = np.array([site[0] + rng.normal(0, 200), site[1] + rng.normal(0, 200), 3.0])
                r = rng.uniform(35, 90)
                alt = rng.uniform(40, 120)
                mission = [
                    B.Waypoints([(site[0] + r, site[1], alt)], 8, intent=Intent.TRANSIT),
                    B.Orbit(np.array(site), r, rng.uniform(3, 7), alt, rng.uniform(150, 400), bool(rng.random() < 0.5)),
                    B.Waypoints([(pos[0], pos[1], 2.0)], 8, intent=Intent.TRANSIT),
                ]
            else:
                area = geo.mapping_areas[int(rng.integers(len(geo.mapping_areas)))]
                pos = np.array([area[0] - 60, area[1] - 60, 3.0])
                mission = [B.Lawnmower(np.array(area[:2]), area[2], area[3], rng.uniform(45, 70), rng.uniform(8, 12), rng.uniform(80, 120))]
                mission.append(B.Waypoints([(pos[0], pos[1], 2.0)], 10, intent=Intent.TRANSIT))
            if initial:
                pos[2] = 60.0
            rid = f"RID-{aid}"
        elif cls == C.BIRD:
            a = rng.uniform(0, 2 * math.pi)
            r = self.radius * 0.8 * math.sqrt(rng.random())
            pos = np.array([r * math.cos(a), r * math.sin(a), rng.uniform(20, 180)])
            h = rng.uniform(0, 2 * math.pi)
            spd = rng.uniform(8, 14)
            vel = np.array([spd * math.cos(h), spd * math.sin(h), 0.0])
            mission = [B.BirdFlight(spd, (15, 250), rng.uniform(300, 1200))]
        elif cls == C.HELICOPTER:
            alt = rng.uniform(250, 600)
            pos, d = self._edge_point(alt)
            spd = rng.uniform(45, 65)
            vel = d * spd
            if rng.random() < 0.45:
                c = pos + d * self.radius * rng.uniform(0.6, 0.9)
                mission = [
                    B.Waypoints([tuple(c)], spd, intent=Intent.TRANSIT, tolerance=60),
                    B.Racetrack(c, rng.uniform(0, 360), rng.uniform(800, 1500), rng.uniform(250, 400), spd * 0.7, alt, rng.uniform(150, 360)),
                ]
                exit_dir = d
                mission.append(B.Waypoints([tuple(c + exit_dir * self.radius * 1.6)], spd, intent=Intent.TRANSIT, tolerance=80))
            else:
                mission = [B.Waypoints([tuple(pos + d * self.radius * 2.2)], spd, intent=Intent.TRANSIT, tolerance=80)]
            if rng.random() < 0.9:
                adsb = f"{rng.integers(0x100000, 0xFFFFFF):06X}"
        elif cls == C.COMMERCIAL_AIRCRAFT:
            ap = geo.airport
            if rng.random() < 0.6:
                # Arrival: straight-in approach descending toward the airport.
                hdg = math.radians(ap[2]) + rng.uniform(-0.25, 0.25)
                far = np.array([ap[0], ap[1]]) - np.array([math.cos(hdg), math.sin(hdg)]) * (self.radius * 2 + 3000)
                pos = np.array([far[0], far[1], rng.uniform(3500, 4500)])
                pos[:2] = self._clip_to_zone(pos[:2], np.array(ap[:2]))
                spd = rng.uniform(110, 150)
                mission = [B.Waypoints([(ap[0], ap[1], 300.0)], spd, intent=Intent.TRANSIT, tolerance=200, vertical_speed=7)]
                if rng.random() < 0.2:
                    hold_c = pos + (np.r_[ap[:2], pos[2]] - pos) * 0.35
                    mission.insert(0, B.Racetrack(hold_c, math.degrees(math.atan2(ap[0] - pos[0], ap[1] - pos[1])), 6000, 2000, spd, pos[2], rng.uniform(200, 400)))
            else:
                alt = rng.uniform(8000, 11000)
                pos, d = self._edge_point(alt, 0.6)
                spd = rng.uniform(200, 250)
                mission = [B.Waypoints([tuple(pos + d * self.radius * 2.5)], spd, intent=Intent.TRANSIT, tolerance=300)]
            target = np.asarray(mission[0].points[0] if isinstance(mission[0], B.Waypoints) else mission[0].center, dtype=float)
            dd = target - pos
            dd[2] = 0
            vel = dd / (np.linalg.norm(dd) + 1e-9) * spd
            adsb = f"{rng.integers(0x100000, 0xFFFFFF):06X}"
        elif cls == C.EMERGENCY_VEHICLE:
            hosp = geo.hospital
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(1500, self.radius * 0.7)
            inc = np.array([r * math.cos(a), r * math.sin(a)])
            pos = np.array([hosp[0], hosp[1], 5.0 if not initial else 300.0])
            spd = rng.uniform(65, 78)
            mission = [
                B.Waypoints([(hosp[0], hosp[1], 300.0), (inc[0], inc[1], 250.0)], spd, intent=Intent.EMERGENCY_RESPONSE, tolerance=60, vertical_speed=8),
                B.Orbit(np.r_[inc, 0], 150, 25, 200, rng.uniform(60, 150), intent=Intent.EMERGENCY_RESPONSE),
                B.Waypoints([(hosp[0], hosp[1], 150.0), (hosp[0], hosp[1], 5.0)], spd, intent=Intent.EMERGENCY_RESPONSE, tolerance=60, vertical_speed=8),
            ]
            adsb = f"{rng.integers(0x100000, 0xFFFFFF):06X}"
            prio = 1 if rng.random() < 0.85 else 0
        elif cls == C.WEATHER_BALLOON:
            ms = geo.met_station
            pos = np.array([ms[0], ms[1], 2.0 if not initial else rng.uniform(200, 3000)])
            mission = [B.Drift(rng.uniform(4.0, 6.0), 4000)]
        else:  # UNKNOWN
            a = rng.uniform(0, 2 * math.pi)
            r = self.radius * 0.6 * math.sqrt(rng.random())
            pos = np.array([r * math.cos(a), r * math.sin(a), rng.uniform(30, 400)])
            if rng.random() < 0.5:
                mission = [B.Wander(pos[:2].copy(), 600, (1, 9), (20, 500), rng.uniform(200, 600), hover_prob=0.5, intent=Intent.TRANSIT)]
            else:
                mission = [B.Drift(rng.uniform(-1.0, 2.0), rng.uniform(200, 600))]
            prof["rcs"] = rng.uniform(-20, 0)
            prof["rotor"] = rng.uniform(0.2, 0.7)
            prof["flap"] = rng.uniform(0.2, 0.7)

        agent = Agent(
            id=aid, cls=cls, pos=np.asarray(pos, dtype=float), vel=np.asarray(vel, dtype=float), mission=mission,
            profile=prof, adsb_id=adsb, rid_id=rid, priority_status=prio, phase=rng.uniform(0, 6.28), born=self.t,
            route_corridor=corridor, rcs_offset=rng.normal(0, 2.0),
        )
        self.agents[aid] = agent
        return agent

    def _clip_to_zone(self, p: np.ndarray, toward: np.ndarray) -> np.ndarray:
        """Move p along the line to ``toward`` until it lies on the zone boundary."""
        if np.linalg.norm(p) <= self.radius:
            return p
        d = toward - p
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if np.linalg.norm(p + d * mid) > self.radius:
                lo = mid
            else:
                hi = mid
        return p + d * hi

    def spawn_formation(self, count: int, spacing: float) -> None:
        """A group of drones flying the same route with fixed offsets (show / swarm)."""
        rng = self.rng
        park = self.geo.parks[0]
        alt = rng.uniform(60, 100)
        route = []
        for _ in range(6):
            a = rng.uniform(0, 2 * math.pi)
            route.append((park[0] + 900 * math.cos(a), park[1] + 900 * math.sin(a), alt))
        for i in range(count):
            off = np.array([(i % 3) * spacing, (i // 3) * spacing, 0.0])
            pts = [tuple(np.asarray(p) + off) for p in route]
            aid = self._new_id("O")
            prof = dict(PROFILES[C.INDUSTRIAL_DRONE])
            prof["gust"] = 0.15  # tight station-keeping
            self.agents[aid] = Agent(
                id=aid, cls=C.INDUSTRIAL_DRONE, pos=np.array([park[0], park[1], alt]) + off, vel=np.zeros(3),
                mission=[B.Waypoints(pts, 9.0, intent=Intent.TRANSIT, tolerance=12)], profile=prof,
                rid_id=f"RID-{aid}", born=self.t, rcs_offset=rng.normal(0, 1.0),
            )

    # ------------------------------------------------------------ anomalies
    def schedule_anomaly(self, agent: Agent, t: float, kind: str) -> None:
        agent.scheduled.append((t, kind, None))

    def inject_anomaly(self, agent: Agent, kind: str, t: float) -> None:
        rng = self.rng
        if kind == "altitude_jump":
            dz = rng.choice([-1, 1]) * rng.uniform(70, 130)
            tgt = max(10.0, agent.pos[2] + dz)
            b = B.Waypoints([(agent.pos[0], agent.pos[1], tgt)], 2.0, intent=Intent.ANOMALOUS, tolerance=6, vertical_speed=9)
        elif kind == "erratic":
            b = B.Erratic(rng.uniform(20, 40), agent.profile["max_speed"])
        elif kind == "long_hover":
            b = B.Hover(rng.uniform(120, 200), intent=Intent.ANOMALOUS)
        elif kind == "route_deviation":
            a = rng.uniform(0, 2 * math.pi)
            off = agent.pos[:2] + rng.uniform(900, 1500) * np.array([math.cos(a), math.sin(a)])
            off = off * min(1.0, self.radius * 0.85 / max(np.linalg.norm(off), 1))
            b = B.Waypoints([(off[0], off[1], agent.pos[2])], agent.profile["max_speed"] * 0.8, intent=Intent.ANOMALOUS, tolerance=30)
        elif kind == "transponder_off":
            agent.transponder_on = False
            dur = rng.uniform(40, 90)
            agent.anomalies.append([kind, t, t + dur])
            agent.scheduled.append((t + dur, "_transponder_on", None))
            return
        elif kind == "_transponder_on":
            agent.transponder_on = True
            return
        else:
            return
        b.anomaly_kind = kind
        agent.mission.insert(0, b)
        agent.anomalies.append([kind, t, None])

    def _maybe_random_anomaly(self, dt: float) -> None:
        if self.anomaly_rate <= 0 or self.rng.random() > self.anomaly_rate / 60.0 * dt:
            return
        eligible = [a for a in self.agents.values() if a.cls not in (C.BIRD, C.WEATHER_BALLOON, C.COMMERCIAL_AIRCRAFT)
                    and not a.anomalies and self.t - a.born > 40 and a.pos[2] > 20]
        if not eligible:
            return
        agent = eligible[int(self.rng.integers(len(eligible)))]
        kinds = ["altitude_jump", "erratic", "long_hover", "route_deviation"]
        if agent.adsb_id or agent.rid_id:
            kinds.append("transponder_off")
        self.inject_anomaly(agent, kinds[int(self.rng.integers(len(kinds)))], self.t)

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> None:
        self.t += dt
        n = len(self.agents)
        for cls, rate in self.spawn_rates.items():
            if n < self.max_agents and self.rng.random() < rate / 60.0 * dt:
                self.spawn(cls)
                n += 1
        self._maybe_random_anomaly(dt)
        for a in list(self.agents.values()):
            a.step(self.t, dt, self)
            if np.linalg.norm(a.pos[:2]) > self.radius * 1.35 or a.pos[2] > 15000:
                a.alive = False
            if not a.alive:
                for an in a.anomalies:
                    if an[2] is None:
                        an[2] = self.t
                self.retired.append(self.agents.pop(a.id))

    def truth_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "id": a.id,
                "cls": a.cls.value,
                "pos": [round(float(v), 1) for v in a.pos],
                "vel": [round(float(v), 2) for v in a.vel],
                "intent": a.intent.value,
                "anomaly": a.active_anomaly(self.t),
            }
            for a in self.agents.values()
        ]
