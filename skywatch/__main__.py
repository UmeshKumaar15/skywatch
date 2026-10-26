"""SkyWatch command line.

    python -m skywatch serve     [--scenario urban_mixed] [--seed 42] [--speed 1] [--source sim|external|sbs://host:port|replay:file.jsonl]
    python -m skywatch train     [--episodes 24] [--duration 600] [--workers 4]
    python -m skywatch bench     [--episodes 6] [--duration 600] [--scenario urban_mixed]
    python -m skywatch simulate  --scenario dense_swarm --duration 900     (headless, recorded for playback)
    python -m skywatch export    --run 3 --out exports/
    python -m skywatch scenarios
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="skywatch", description="Real-time airspace monitoring & trajectory prediction")
    p.add_argument("--config", help="YAML file overriding config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the live service + dashboard")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--scenario")
    s.add_argument("--seed", type=int)
    s.add_argument("--speed", type=float, help="simulation time scale")
    s.add_argument("--source", default="sim", help="sim | external | sbs://host:port | replay:path.jsonl")

    t = sub.add_parser("train", help="generate simulated datasets and train the learned models")
    t.add_argument("--episodes", type=int, default=24)
    t.add_argument("--duration", type=float, default=600)
    t.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--out", default="models")
    t.add_argument("--baselines-only", action="store_true", help="only re-calibrate anomaly baselines")

    b = sub.add_parser("bench", help="benchmark tracking / classification / prediction / anomaly detection")
    b.add_argument("--episodes", type=int, default=4)
    b.add_argument("--duration", type=float, default=600)
    b.add_argument("--scenario", default="urban_mixed")
    b.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    b.add_argument("--seed0", type=int, default=5000)
    b.add_argument("--out", default="reports")

    m = sub.add_parser("simulate", help="run a scenario headless and record it for playback")
    m.add_argument("--scenario", default="urban_mixed")
    m.add_argument("--seed", type=int)
    m.add_argument("--duration", type=float, default=600)
    m.add_argument("--jsonl", help="also write the raw normalised observations to a JSONL file (replayable)")

    e = sub.add_parser("export", help="export a recorded run to CSV")
    e.add_argument("--run", type=int, required=True)
    e.add_argument("--out", default="exports")

    sub.add_parser("scenarios", help="list available scenarios")

    args = p.parse_args(argv)
    from .config import list_scenarios, load_config, load_scenario, resolve_path

    cfg = load_config(args.config)

    if args.cmd == "scenarios":
        for sc in list_scenarios():
            print(f"{sc['name']:<22} seed={sc['seed']:<6} {sc['description']}")
        return 0

    if args.cmd == "train":
        from .training import calibrate_only, train

        if args.baselines_only:
            print(json.dumps(calibrate_only(args.episodes, args.duration, args.workers, args.out), indent=1))
        else:
            train(args.episodes, args.duration, args.workers, args.out, epochs=args.epochs)
        return 0

    if args.cmd == "bench":
        from .benchmark import run_benchmark

        run_benchmark(cfg, args.scenario, args.episodes, args.duration, args.workers, args.seed0, args.out)
        return 0

    if args.cmd == "simulate":
        from .runner import SimRunner
        from .storage.store import Store
        from .core.bus import TOPIC_OBSERVATIONS

        store = Store(resolve_path(cfg["storage"]["path"]))
        sc = load_scenario(args.scenario, seed=args.seed)
        runner = SimRunner(cfg, sc, store=store)
        fh = open(args.jsonl, "w") if args.jsonl else None
        if fh:
            def dump(batch):
                for o in batch:
                    fh.write(json.dumps({"timestamp": o.timestamp, "sensor_id": o.sensor_id, "sensor_type": o.sensor_type.value,
                                         "position": o.position.tolist(), "covariance": o.covariance.tolist(),
                                         "velocity": None if o.velocity is None else o.velocity.tolist(),
                                         "identity": o.identity, "features": o.features}) + "\n")
            runner.bus.subscribe(TOPIC_OBSERVATIONS, dump)
        t0 = time.time()
        asyncio.run(runner.run_headless(args.duration))
        if fh:
            fh.close()
        print(f"Recorded run {runner.run_id} ({args.duration:.0f} s simulated in {time.time() - t0:.1f} s) -> {store.path}")
        return 0

    if args.cmd == "export":
        from .storage.store import Store

        store = Store(resolve_path(cfg["storage"]["path"]))
        for f in store.export_csv(args.run, args.out):
            print(f)
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .server.app import create_app

        srv = cfg.setdefault("server", {})
        for k, v in (("host", args.host), ("port", args.port), ("scenario", args.scenario), ("seed", args.seed), ("time_scale", args.speed)):
            if v is not None:
                srv[k] = v
        srv["source"] = args.source
        app = create_app(cfg)
        uvicorn.run(app, host=srv.get("host", "0.0.0.0"), port=int(srv.get("port", 8000)), log_level="info")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
