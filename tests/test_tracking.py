import numpy as np

from skywatch.core.types import Observation, SensorType, TrackStatus
from skywatch.tracking.tracker import MultiObjectTracker

CFG = {"confirm_hits": 3, "max_coast_time": 8.0}


def radar(t, p, sigma=5.0, ident=None, stype=SensorType.RADAR, truth=None):
    return Observation("r1", stype, t, np.asarray(p, float), np.eye(3) * sigma**2, identity=ident, truth_id=truth)


def test_crossing_targets_keep_identity():
    trk = MultiObjectTracker(CFG)
    rng = np.random.default_rng(3)
    for k in range(60):
        t = float(k)
        a = np.array([-600 + 20 * k, 0.0, 100.0])
        b = np.array([0.0, -600 + 20 * k, 120.0])
        trk.process([radar(t, a + rng.normal(0, 5, 3), truth="A"), radar(t, b + rng.normal(0, 5, 3), truth="B")])
    conf = trk.confirmed()
    assert len(conf) == 2
    for tr in conf:
        assert len(tr.truth_votes) == 1, tr.truth_votes  # never swapped


def test_clutter_does_not_confirm_tracks():
    trk = MultiObjectTracker(CFG)
    rng = np.random.default_rng(4)
    for k in range(40):
        obs = [Observation("r1", SensorType.RADAR, float(k), rng.uniform(-5000, 5000, 3), np.eye(3) * 100, quality=0.5) for _ in range(3)]
        trk.process(obs)
    assert trk.confirmed() == []


def test_identity_conflict_blocks_association_and_coasting():
    trk = MultiObjectTracker(CFG)
    for k in range(5):
        trk.process([radar(float(k), [10 * k, 0, 50], ident="RID-1", stype=SensorType.REMOTE_ID)])
    tr = trk.confirmed()[0]
    assert tr.identity == "RID-1"
    # A different identity at the same place must not be absorbed
    trk.process([radar(5.0, [50, 0, 50], ident="RID-2", stype=SensorType.REMOTE_ID)])
    assert trk.tracks[tr.id].identity == "RID-1"
    assert any(t.identity == "RID-2" for t in trk.tracks.values())
    # Silence -> coasting -> deleted
    trk.advance(9.0)
    assert trk.tracks[tr.id].status == TrackStatus.COASTING
    trk.advance(20.0)
    assert tr.id not in trk.tracks


def test_duplicate_tracks_are_merged():
    trk = MultiObjectTracker(CFG)
    for k in range(6):
        trk.process([radar(float(k), [10 * k, 0, 50])])
    first = trk.confirmed()[0]
    # Inject a duplicate track of the same object (as a late re-initiation would create)
    dup = trk._new_track(radar(5.0, [50, 0, 50]))
    dup.imm.x[:, 3] = 10.0
    dup.status = TrackStatus.CONFIRMED
    dup.update_batches.clear()  # it was never updated in the same scan as the original
    trk.tracks[dup.id] = dup
    assert len(trk.confirmed()) == 2
    for k in range(6, 20):
        trk.process([radar(float(k), [10 * k, 0, 50])])
    assert [t.id for t in trk.confirmed()] == [first.id]
    assert trk.stats.merged >= 1


def test_close_formation_is_not_merged():
    trk = MultiObjectTracker(CFG)
    rng = np.random.default_rng(5)
    for k in range(40):
        a = [8 * k, 0, 60]
        b = [8 * k, 25, 60]
        trk.process([radar(float(k), np.add(a, rng.normal(0, 2, 3)), sigma=2), radar(float(k), np.add(b, rng.normal(0, 2, 3)), sigma=2)])
    assert len(trk.confirmed()) == 2
