import math

import numpy as np

from skywatch.core.types import Observation, SensorType
from skywatch.prediction.predictor import TrajectoryPredictor, mixture_at, serialize
from skywatch.tracking.tracker import MultiObjectTracker

CLS = {"delivery_drone": 0.9, "unknown": 0.1}


def _track(points, dt=1.0):
    trk = MultiObjectTracker({})
    for k, p in enumerate(points):
        trk.process([Observation("r", SensorType.LIDAR, k * dt, np.asarray(p, float), np.eye(3) * 1.0)])
    return trk.confirmed()[0]


def test_straight_track_prefers_constant_velocity():
    tr = _track([[15 * k, 0, 80] for k in range(40)])
    pred = TrajectoryPredictor({"horizon": 60, "step": 2})
    for _ in range(3):
        hyps = pred.predict(tr, tr.t, CLS, {})
    cv = next(h for h in hyps if h.kind == "cv")
    m, S = cv.at(10.0)
    assert np.allclose(m, [15 * 39 + 150, 0, 80], atol=8)
    assert abs(sum(h.weight for h in hyps) - 1) < 1e-9
    # Uncertainty grows with horizon
    assert np.trace(cv.at(60.0)[1]) > np.trace(cv.at(10.0)[1]) > np.trace(cv.at(1.0)[1])


def test_orbit_hypothesis_for_circling_track():
    r, w = 80.0, 0.08
    tr = _track([[r * math.cos(w * k), r * math.sin(w * k), 60] for k in range(90)])
    pred = TrajectoryPredictor({"horizon": 60, "step": 2})
    hyps = pred.predict(tr, tr.t, CLS, {"inspection": 0.8})
    orbit = next((h for h in hyps if h.kind == "orbit"), None)
    assert orbit is not None
    m, _ = orbit.at(20.0)
    assert abs(np.hypot(m[0], m[1]) - r) < 15


def test_predictions_never_go_below_ground():
    tr = _track([[5 * k, 0, max(100 - 6 * k, 1)] for k in range(18)])
    hyps = TrajectoryPredictor({"horizon": 60, "step": 2}).predict(tr, tr.t, CLS, {})
    assert all((h.means[:, 2] >= 0).all() for h in hyps)
    out = serialize(hyps, tr.t, 0.0, 10)
    assert out["mixture"] and out["hypotheses"]
    m, S = mixture_at(hyps, np.array([10.0, 20.0]))
    assert m.shape == (2, 3) and S.shape == (2, 3, 3)
