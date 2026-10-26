"""Example external sensor gateway.

Runs as an independent process and streams detections of a synthetic target
to a SkyWatch instance started with ``python -m skywatch serve --source external``.
Replace ``read_sensor()`` with your radar / LiDAR / camera driver: anything
that can produce a position (and ideally a covariance) can feed the platform.

    python examples/external_gateway.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request


def read_sensor(t: float) -> list[dict]:
    """Synthetic radar: one drone orbiting at 120 m radius, 80 m altitude."""
    a = 0.05 * t
    x, y = 120 * math.cos(a) + random.gauss(0, 4), 120 * math.sin(a) + random.gauss(0, 4)
    return [{
        "timestamp": t,
        "sensor_id": "gateway-radar-1",
        "sensor_type": "radar",
        "position": [x, y, 80 + random.gauss(0, 3)],
        "sigma": [4, 4, 3],
        "features": {"rcs_dbsm": -14 + random.gauss(0, 2), "rotor_score": 0.8, "flap_score": 0.1},
    }]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rate", type=float, default=2.0, help="scans per second")
    args = ap.parse_args()
    while True:
        batch = read_sensor(time.time())
        req = urllib.request.Request(f"{args.url}/api/ingest", data=json.dumps(batch).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
        time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
