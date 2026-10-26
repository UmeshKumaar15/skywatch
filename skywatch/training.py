"""Dataset generation and model training from simulation.

    python -m skywatch train --episodes 24 --duration 600 --workers 4

Episodes are randomised variations of the ``quiet_training`` scenario run
through the *real* tracker, so the learned models see exactly the noisy,
filtered track estimates they will see at inference time. Produces:

  models/classifier.npz     object classifier (MLP + temperature + OOD stats)
  models/motion_model.npz   learned trajectory residual model + uncertainty
  models/baselines.json     class-conditional kinematic baselines (anomaly priors)
  models/training_report.json
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .anomaly.detector import FEATS, kinematic_features
from .classification.features import FEATURE_NAMES, extract
from .classification.nn import MLP
from .config import load_config, load_scenario, resolve_path
from .core.types import OBJECT_CLASSES, TrackStatus
from .prediction.learned import DEFAULT_TIMES, build_input, horizon_scale, targets_from_truth
from .runner import SimRunner

CLS_INDEX = {c.value: i for i, c in enumerate(OBJECT_CLASSES)}


def randomized_scenario(seed: int, duration: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 100_000)
    sc = load_scenario("quiet_training", seed=seed)
    sc["duration"] = duration
    sc["wind"] = [float(rng.normal(0, 4)), float(rng.normal(0, 4))]
    sc["gust_intensity"] = float(rng.uniform(0.5, 1.6))
    sc["initial"] = {k: int(max(0, v + rng.integers(-1, 3))) for k, v in sc["initial"].items()}
    sc["anomaly_rate_per_min"] = float(rng.uniform(0.2, 0.8))
    return sc


def run_episode(args: tuple[int, float, dict]) -> dict[str, Any]:
    seed, duration, cfg = args
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("storage", {})["record"] = False
    scenario = randomized_scenario(seed, duration)
    runner = SimRunner(cfg, scenario, store=None, analytics=False)
    world = runner.engine.world
    truth_traj: dict[str, list] = defaultdict(list)
    cls_X, cls_y = [], []
    motion = []  # (x, frame, truth_id, t_ref, cls)
    base = defaultdict(list)
    state = {"next": 0.0, "traj": 0.0}

    def on_step(r: SimRunner, _snap) -> None:
        t = r.engine.t
        if t >= state["traj"]:
            state["traj"] = t + 0.5
            for a in world.agents.values():
                truth_traj[a.id].append((t, *a.pos))
        if t < state["next"]:
            return
        state["next"] = t + 2.0
        now = r.pipeline.tracker.t
        agents = world.agents
        for tr in list(r.pipeline.tracker.tracks.values()):
            if tr.status not in (TrackStatus.CONFIRMED, TrackStatus.COASTING) or now - tr.created < 3:
                continue
            tot = sum(tr.truth_votes.values())
            if not tot:
                continue
            tid, votes = tr.truth_votes.most_common(1)[0]
            if votes / tot < 0.8 or tid not in agents:
                continue
            agent = agents[tid]
            cls = agent.cls.value
            cls_X.append(extract(tr, now))
            cls_y.append(CLS_INDEX[cls])
            if now - tr.last_update < 1.5:
                onehot = np.full(len(OBJECT_CLASSES), 0.3 / len(OBJECT_CLASSES))
                onehot[CLS_INDEX[cls]] += 0.7
                mp = tr.imm.mu
                built = build_input(tr.history, tr.t, tr.position, tr.velocity, onehot, np.array([mp[0], mp[1], mp[2]]))
                if built is not None:
                    motion.append((built[0], built[1], tid, tr.t))
            if now - tr.created > 20 and agent.active_anomaly(now) is None and not agent.anomalies:
                kin = kinematic_features(tr, now)
                if kin is not None:
                    vz, _dz, acc, turn, nis = kin
                    base[cls].append((vz, acc, turn, nis))

    asyncio.run(runner.run_headless(duration, on_step))

    # Build motion targets from the ground-truth trajectories.
    H = len(DEFAULT_TIMES)
    mX, mY, mM, mScale = [], [], [], []
    traj = {k: np.array(v) for k, v in truth_traj.items()}
    for x, frame, tid, t_ref in motion:
        tj = traj.get(tid)
        if tj is None or len(tj) < 2:
            continue
        ft = t_ref + DEFAULT_TIMES
        valid = ft <= tj[-1, 0]
        if valid.sum() < 4:
            continue
        fut = np.stack([np.interp(ft, tj[:, 0], tj[:, k]) for k in (1, 2, 3)], axis=1)
        y = targets_from_truth(fut, frame, DEFAULT_TIMES)
        mask = np.repeat(valid.astype(float), 3)
        mX.append(x)
        mY.append(y)
        mM.append(mask)
        mScale.append(frame.scale)
    return {
        "seed": seed,
        "cls_X": np.array(cls_X), "cls_y": np.array(cls_y, dtype=int),
        "m_X": np.array(mX), "m_Y": np.array(mY), "m_M": np.array(mM), "m_S": np.array(mScale),
        "base": {k: np.array(v) for k, v in base.items()},
        "H": H,
    }


def _stack(eps: list[dict], key: str) -> np.ndarray:
    arrs = [e[key] for e in eps if len(e[key])]
    return np.concatenate(arrs) if arrs else np.zeros((0,))


def calibrate_baselines(results: list[dict]) -> dict[str, list]:
    """Per class and feature: [mean, std, tail scale] from normal (non-anomalous) traffic."""
    pooled: dict[str, list] = defaultdict(list)
    for e in results:
        for c, arr in e["base"].items():
            if len(arr):
                pooled[c].append(arr)
    baselines = {}
    for c, arrs in pooled.items():
        a = np.concatenate(arrs)
        if len(a) < 30:
            continue
        stats = []
        for i in range(len(FEATS)):
            col = a[:, i]
            m, sd = float(col.mean()), float(max(col.std(), 0.05))
            tail = float(max((np.percentile(col, 99.5) - m) / 2.58, sd * 0.5, 0.05))
            stats.append([round(m, 3), round(sd, 3), round(tail, 3)])
        baselines[c] = stats
    return baselines


def calibrate_only(episodes: int = 8, duration: float = 400.0, workers: int = 4, out_dir: str = "models", seed0: int = 3000) -> dict:
    """Re-calibrate only the anomaly baselines (fast; models unchanged)."""
    cfg = load_config()
    jobs = [(seed0 + i, duration, cfg) for i in range(episodes)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(run_episode, jobs))
    baselines = calibrate_baselines(results)
    out = resolve_path(out_dir)
    (out / "baselines.json").write_text(json.dumps(baselines, indent=1))
    rp = out / "training_report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        rep["baselines"] = {c: dict(zip(FEATS, v)) for c, v in baselines.items()}
        rp.write_text(json.dumps(rep, indent=1))
    return baselines


def train(episodes: int = 24, duration: float = 600.0, workers: int = 4, out_dir: str = "models", seed0: int = 1000,
          epochs: int = 40, verbose: bool = True) -> dict[str, Any]:
    cfg = load_config()
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(seed0 + i, duration, cfg) for i in range(episodes)]
    if verbose:
        print(f"Generating {episodes} episodes x {duration:.0f} s with {workers} workers ...")
    results = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(run_episode, jobs):
                results.append(r)
                if verbose:
                    print(f"  episode seed={r['seed']}: {len(r['cls_y'])} class samples, {len(r['m_X'])} motion samples")
    else:
        for j in jobs:
            r = run_episode(j)
            results.append(r)
            if verbose:
                print(f"  episode seed={r['seed']}: {len(r['cls_y'])} class samples, {len(r['m_X'])} motion samples")
    gen_time = time.time() - t0
    n_val = max(1, episodes // 6)
    train_eps, val_eps = results[:-n_val], results[-n_val:]
    report: dict[str, Any] = {"episodes": episodes, "duration": duration, "generation_s": round(gen_time, 1)}

    # ----------------------------------------------------------- classifier
    Xtr, ytr = _stack(train_eps, "cls_X"), _stack(train_eps, "cls_y").astype(int)
    Xva, yva = _stack(val_eps, "cls_X"), _stack(val_eps, "cls_y").astype(int)
    k = len(OBJECT_CLASSES)
    counts = np.bincount(ytr, minlength=k).astype(float)
    sw = 1.0 / np.maximum(counts[ytr], 1.0)
    sw *= len(ytr) / sw.sum()
    clf = MLP([Xtr.shape[1], 96, 96, k], seed=1, task="classification")
    if verbose:
        print(f"Training classifier on {len(ytr)} samples ({len(yva)} validation) ...")
    clf.fit(Xtr, ytr, epochs=epochs, sample_weight=sw, lr=2e-3, verbose=verbose)
    temp = clf.calibrate_temperature(Xva, yva)
    pva = clf.predict_proba(Xva)
    acc = float((pva.argmax(1) == yva).mean())
    conf = np.zeros((k, k), dtype=int)
    for a, b in zip(yva, pva.argmax(1)):
        conf[a, b] += 1
    per_class = {OBJECT_CLASSES[i].value: round(float(conf[i, i] / max(conf[i].sum(), 1)), 3) for i in range(k)}
    ood_mean, ood_std = Xtr.mean(0), Xtr.std(0) + 1e-6
    d_va = np.sqrt(np.mean(np.minimum(((Xva - ood_mean) / ood_std) ** 2, 100.0), axis=1))
    ood_scale = float(np.percentile(d_va, 97))
    np.savez(out / "classifier.npz", **clf.state_dict("clf_"), ood_mean=ood_mean, ood_std=ood_std, ood_scale=ood_scale,
             feature_names=np.array(FEATURE_NAMES), classes=np.array([c.value for c in OBJECT_CLASSES]))
    report["classifier"] = {"train_samples": int(len(ytr)), "val_samples": int(len(yva)), "val_accuracy": round(acc, 4),
                            "temperature": temp, "per_class_recall": per_class, "confusion": conf.tolist(),
                            "classes": [c.value for c in OBJECT_CLASSES]}
    if verbose:
        print(f"  validation accuracy {acc:.3f} (temperature {temp:.2f})")

    # ---------------------------------------------------------- motion model
    mX, mY, mM = _stack(train_eps, "m_X"), _stack(train_eps, "m_Y"), _stack(train_eps, "m_M")
    vX, vY, vM, vS = _stack(val_eps, "m_X"), _stack(val_eps, "m_Y"), _stack(val_eps, "m_M"), _stack(val_eps, "m_S")
    H = len(DEFAULT_TIMES)
    out_scale = horizon_scale(DEFAULT_TIMES)
    net = MLP([mX.shape[1], 128, 128, H * 3], seed=2, task="regression")
    if verbose:
        print(f"Training motion model on {len(mX)} samples ({len(vX)} validation) ...")
    net.fit(mX, mY / out_scale, mask=mM, epochs=epochs, lr=1.5e-3, weight_decay=1e-5, verbose=verbose)
    pv = net.predict(vX) * out_scale
    res = (pv - vY).reshape(len(vX), H, 3)
    msk = vM.reshape(len(vX), H, 3)
    sigma_norm = np.sqrt((res**2 * msk).sum(0) / np.maximum(msk.sum(0), 1))
    sigma_floor = 2.0 + 0.2 * DEFAULT_TIMES
    np.savez(out / "motion_model.npz", **net.state_dict("mm_"), times=DEFAULT_TIMES, sigma_norm=sigma_norm, sigma_floor=sigma_floor,
             out_scale=out_scale)
    # Displacement errors (metres) vs. the constant-velocity baseline (= zero residual)
    err_l = np.linalg.norm(res, axis=2) * vS[:, None]
    err_cv = np.linalg.norm(vY.reshape(len(vX), H, 3), axis=2) * vS[:, None]
    m2 = msk[:, :, 0] > 0
    horizons = {}
    for hz in (5.0, 10.0, 20.0, 30.0, 60.0):
        j = int(np.argmin(np.abs(DEFAULT_TIMES - hz)))
        sel = m2[:, j]
        if sel.any():
            horizons[f"{int(hz)}s"] = {"learned_m": round(float(np.mean(err_l[sel, j])), 1), "cv_m": round(float(np.mean(err_cv[sel, j])), 1)}
    report["motion_model"] = {"train_samples": int(len(mX)), "val_samples": int(len(vX)), "mean_error_by_horizon": horizons}
    if verbose:
        for k_, v in horizons.items():
            print(f"  {k_:>4}: learned {v['learned_m']:7.1f} m   constant-velocity {v['cv_m']:7.1f} m")

    # ------------------------------------------------------------ baselines
    baselines = calibrate_baselines(results)
    (out / "baselines.json").write_text(json.dumps(baselines, indent=1))
    report["baselines"] = {c: dict(zip(FEATS, v)) for c, v in baselines.items()}
    report["total_s"] = round(time.time() - t0, 1)
    (out / "training_report.json").write_text(json.dumps(report, indent=1))
    if verbose:
        print(f"Saved models to {out} ({report['total_s']} s)")
    return report
