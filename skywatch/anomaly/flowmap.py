"""Learned airspace flow map.

A spatial model of *normal* traffic, learned online from observed tracks:
for each grid cell, altitude band and object group it accumulates a
histogram of motion axes (4 axial heading bins, i.e. direction modulo 180
degrees because routes are flown both ways, + hover). The surprise of a
track's current motion under this model flags unusual flight paths (wrong
way along a route, flying where that type of object never flies, ...).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

ALT_BANDS = np.array([150.0, 1000.0, 5000.0])
GROUPS = {"delivery_drone": 0, "recreational_drone": 0, "industrial_drone": 0, "bird": 1, "helicopter": 2,
          "commercial_aircraft": 2, "emergency_vehicle": 2, "weather_balloon": 3, "unknown": 3}
N_BINS = 5  # 4 axial headings + hover


class FlowMap:
    def __init__(self, radius: float, cell: float = 400.0) -> None:
        self.cell = cell
        self.extent = radius * 1.4
        self.n = int(math.ceil(2 * self.extent / cell))
        self.counts = np.zeros((4, len(ALT_BANDS) + 1, self.n, self.n, N_BINS), dtype=np.float32)
        self.total_updates = 0

    def _index(self, pos: np.ndarray) -> Optional[tuple[int, int, int]]:
        cx = int((pos[0] + self.extent) // self.cell)
        cy = int((pos[1] + self.extent) // self.cell)
        if not (0 <= cx < self.n and 0 <= cy < self.n):
            return None
        band = int(np.searchsorted(ALT_BANDS, pos[2]))
        return band, cx, cy

    @staticmethod
    def _bin(vel: np.ndarray) -> int:
        sp = float(np.linalg.norm(vel[:2]))
        if sp < 1.5:
            return 4
        a = math.atan2(vel[1], vel[0]) % math.pi
        return int(a // (math.pi / 4)) % 4

    def update(self, label: str, pos: np.ndarray, vel: np.ndarray, weight: float = 1.0) -> None:
        idx = self._index(pos)
        if idx is None:
            return
        g = GROUPS.get(label, 3)
        b = self._bin(vel)
        band, cx, cy = idx
        self.counts[g, band, cx, cy, b] += weight
        # Light spatial smoothing so the model generalises to adjacent cells.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = cx + dx, cy + dy
            if 0 <= x < self.n and 0 <= y < self.n:
                self.counts[g, band, x, y, b] += 0.25 * weight
        self.total_updates += 1

    def surprise(self, label: str, pos: np.ndarray, vel: np.ndarray) -> tuple[float, dict[str, float]]:
        """Return an unusualness score in [0, 1] plus evidence."""
        idx = self._index(pos)
        if idx is None:
            return 0.0, {}
        g = GROUPS.get(label, 3)
        band, cx, cy = idx
        group_total = float(self.counts[g].sum())
        if group_total < 20000:  # still learning
            return 0.0, {"learning": group_total}
        cell = self.counts[g, band, cx, cy]
        tot = float(cell.sum())
        alpha = 0.5
        p = (cell[self._bin(vel)] + alpha) / (tot + alpha * N_BINS)
        dir_surprise = -math.log(p)
        occupied = self.counts[g, band].sum(axis=-1)
        mean_occ = float(occupied[occupied > 0].mean()) if (occupied > 0).any() else 0.0
        rarity = 1.0 - min(1.0, tot / max(0.02 * mean_occ, 1.0))
        # Evidence-weighted: a cell must have seen plenty of traffic before a rare
        # direction there is considered meaningful.
        dir_score = 1 / (1 + math.exp(-(dir_surprise - 3.4) / 0.4)) * min(1.0, tot / 400.0) if tot >= 150 else 0.0
        score = max(dir_score, 0.6 * rarity)
        return float(score), {"direction_surprise": dir_surprise, "cell_samples": tot, "rarity": rarity}

    def save(self, path: Path) -> None:
        np.savez_compressed(path, counts=self.counts, cell=self.cell, extent=self.extent)

    def load(self, path: Path) -> bool:
        if not Path(path).exists():
            return False
        d = np.load(path)
        if d["counts"].shape != self.counts.shape:
            return False
        self.counts = d["counts"].astype(np.float32)
        self.total_updates = int(self.counts.sum() / 2.0)
        return True
