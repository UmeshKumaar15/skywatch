import math

import numpy as np

from skywatch.core.geo import LocalFrame
from skywatch.tracking.imm import IMMFilter, measurement_matrices

Q = {"cv": 0.6, "ca": 3.0, "ct": 1.0, "turn_rate": 0.04}


def _filter(x0):
    P0 = np.diag([100, 100, 100, 400, 400, 100, 4, 4, 4, 0.0025])
    return IMMFilter(np.r_[x0, np.zeros(10 - len(x0))], P0, Q)


def test_imm_converges_on_constant_velocity_target():
    rng = np.random.default_rng(0)
    f = _filter([0, 0, 100])
    H = measurement_matrices(False)
    R = np.eye(3) * 25
    for k in range(1, 60):
        truth = np.array([12.0 * k, -5.0 * k, 100.0])
        f.predict(1.0)
        f.update(truth + rng.normal(0, 5, 3), R, H)
    assert np.allclose(f.state[3:5], [12, -5], atol=1.0)
    assert np.linalg.norm(f.state[0:3] - truth) < 10
    assert f.mode_probabilities()["cv"] > 0.4


def test_imm_detects_turn():
    rng = np.random.default_rng(1)
    f = _filter([0, 0, 100])
    H = measurement_matrices(False)
    R = np.eye(3) * 4
    w, v, r = 0.1, 20.0, 200.0
    for k in range(0, 80):
        a = w * k
        truth = np.array([r * math.sin(a), r - r * math.cos(a), 100.0])
        f.predict(1.0)
        f.update(truth + rng.normal(0, 2, 3), R, H)
    assert f.mode_probabilities()["ct"] > 0.5
    assert abs(f.x[2][9] - w) < 0.03


def test_local_frame_roundtrip():
    fr = LocalFrame(52.37, 4.90, 0.0)
    enu = fr.to_enu(52.38, 4.92, 350.0)
    lat, lon, alt = fr.to_geodetic(enu)
    assert abs(lat - 52.38) < 1e-7 and abs(lon - 4.92) < 1e-7 and abs(alt - 350) < 1e-3
    assert 1300 < enu[0] < 1400 and 1100 < enu[1] < 1150
