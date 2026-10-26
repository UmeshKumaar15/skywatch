from skywatch.config import load_config, load_scenario
from skywatch.sim.engine import SimulationEngine


def _run(seed, steps=300):
    cfg = load_config()
    eng = SimulationEngine(load_scenario("urban_mixed", seed=seed), cfg["sensors"])
    out = []
    for _ in range(steps):
        for b in eng.step():
            out.extend((o.sensor_id, round(o.timestamp, 3), tuple(round(float(v), 3) for v in o.position)) for o in b)
    return out, eng


def test_simulation_is_deterministic_per_seed():
    a, _ = _run(11)
    b, _ = _run(11)
    c, _ = _run(12)
    assert a == b
    assert a != c


def test_all_modalities_produce_observations():
    obs, eng = _run(42, steps=200)
    sensors = {s for s, *_ in obs}
    assert {"radar-lr", "adsb-rx", "rid-central"} <= sensors
    assert len(eng.world.agents) > 10


def test_anomaly_injection_records_ground_truth():
    _, eng = _run(5, steps=10)
    agent = next(a for a in eng.world.agents.values() if a.cls.value == "delivery_drone")
    eng.world.inject_anomaly(agent, "long_hover", eng.world.t)
    assert agent.anomalies and agent.anomalies[0][0] == "long_hover"
    assert agent.active_anomaly(eng.world.t) == "long_hover"
