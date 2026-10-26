"""End-to-end benchmark against simulation ground truth.

    python -m skywatch bench --scenario urban_mixed --episodes 4 --duration 600

Metrics
  tracking        MOTA, MOTP, identity switches, coverage, false tracks (CLEAR-MOT style,
                  Hungarian matching every second with a speed-dependent gate)
  classification  accuracy of the track label vs. true class (tracks older than 10 s)
  intent          accuracy of the inferred intent vs. the simulator's mission label
  prediction      mean error at 10/20/30/60 s for: weighted mixture, top hypothesis,
                  best-of-N (oracle), learned model alone, constant velocity baseline;
                  calibration = fraction of truth inside the 95 % mixture region
  anomaly         event recall (injected anomalies flagged by an insight), insight precision,
                  detection delay
  performance     wall-clock cost per simulated second
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import load_scenario, resolve_path
from .core.bus import TOPIC_INSIGHTS
from .core.types import TrackStatus
from .prediction.predictor import mixture_at
from .runner import SimRunner

HORIZONS = (10.0, 20.0, 30.0, 60.0)
ANOMALY_KINDS = {
    "altitude_jump": {"altitude_change", "abrupt_maneuver", "excessive_acceleration"},
    "erratic": {"abrupt_maneuver", "excessive_acceleration", "altitude_change"},
    "long_hover": {"prolonged_hover"},
    "route_deviation": {"route_deviation", "unusual_path", "abrupt_maneuver"},
    "transponder_off": {"comms_loss"},
}


def run_episode(args: tuple[dict, str, int, float]) -> dict[str, Any]:
    cfg, scenario_name, seed, duration = args
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("storage", {})["record"] = False
    sc = load_scenario(scenario_name, seed=seed)
    sc["duration"] = duration
    runner = SimRunner(cfg, sc, store=None, analytics=True)
    world = runner.engine.world
    insights: list = []
    runner.bus.subscribe(TOPIC_INSIGHTS, lambda ins: insights.extend(ins))

    truth_traj: dict[str, list] = defaultdict(list)
    st = {"next": 5.0, "traj": 0.0, "pred_next": 10.0}
    m = Counter()
    motp: list[float] = []
    last_match: dict[str, str] = {}  # truth -> track
    track_truth: dict[str, list] = defaultdict(list)  # track id -> [(t, truth)]
    cls_hits = Counter()
    intent_hits = Counter()
    intent_conf: Counter = Counter()
    preds: list = []

    def on_step(r: SimRunner, _snap) -> None:
        t = r.engine.t
        if t >= st["traj"]:
            st["traj"] = t + 0.5
            for a in world.agents.values():
                truth_traj[a.id].append((t, *a.pos))
        if t < st["next"]:
            return
        st["next"] = t + 1.0
        tracks = [tr for tr in r.pipeline.tracker.tracks.values() if tr.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]
        agents = [a for a in world.agents.values() if t - a.born > 5.0]
        if not agents and not tracks:
            return
        C = np.full((len(agents), len(tracks)), 1e6)
        for i, a in enumerate(agents):
            gate = 60.0 + 1.5 * float(np.linalg.norm(a.vel))
            for j, tr in enumerate(tracks):
                d = float(np.linalg.norm(tr.position_at(t) - a.pos))
                if d < gate:
                    C[i, j] = d
        matched_a, matched_t = set(), set()
        if len(agents) and len(tracks):
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] >= 1e6:
                    continue
                a, tr = agents[i], tracks[j]
                matched_a.add(i)
                matched_t.add(j)
                motp.append(C[i, j])
                m[f"motp_n_{a.cls.value}"] += 1
                m[f"motp_sum_{a.cls.value}"] += C[i, j]
                if a.id in last_match and last_match[a.id] != tr.id:
                    m["idsw"] += 1
                last_match[a.id] = tr.id
                track_truth[tr.id].append(a.id)
                if t - tr.created > 10 and "class" in tr.analytics:
                    cls_hits["n"] += 1
                    cls_hits["ok"] += int(tr.analytics["class"]["label"] == a.cls.value)
                    cls_hits[f"n_{a.cls.value}"] += 1
                    cls_hits[f"ok_{a.cls.value}"] += int(tr.analytics["class"]["label"] == a.cls.value)
                # Intent is only defined for operated aircraft (birds / unknown objects have no operator intent).
                if t - tr.created > 30 and "intent" in tr.analytics and a.cls.value not in ("bird", "unknown"):
                    ti = a.intent.value
                    pi = tr.analytics["intent"]["label"]
                    intent_hits["n"] += 1
                    intent_hits["ok"] += int(pi == ti)
                    intent_conf[(ti, pi)] += 1
                hyps = tr.analytics.get("_hyps")
                if hyps and t >= st["pred_next"] and t - tr.created > 15 and tr.analytics.get("_t", -1e9) >= t - 4.0:
                    entry = {"t0": tr.t, "truth": a.id, "cls": a.cls.value, "pred": {}}
                    vel = tr.velocity.copy()
                    pos = tr.position.copy()
                    for hz in HORIZONS:
                        mix, mcov = mixture_at(hyps, hz)
                        top = max(hyps, key=lambda h: h.weight).at(hz)[0]
                        allm = [h.at(hz)[0] for h in hyps]
                        learned = next((h.at(hz)[0] for h in hyps if h.kind == "learned"), None)
                        entry["pred"][hz] = (mix, mcov, top, allm, learned, pos + vel * hz)
                    preds.append(entry)
        if t >= st["pred_next"]:
            st["pred_next"] = t + 5.0
        m["gt"] += len(agents)
        m["fn"] += len(agents) - len(matched_a)
        m["fp"] += len(tracks) - len(matched_t)
        m["frames"] += 1

    t0 = time.time()
    asyncio.run(runner.run_headless(duration, on_step))
    wall = time.time() - t0

    # ---------------------------------------------------------- prediction
    traj = {k: np.array(v) for k, v in truth_traj.items()}
    perr: dict = defaultdict(lambda: defaultdict(list))
    for e in preds:
        tj = traj.get(e["truth"])
        if tj is None:
            continue
        for hz, (mix, mcov, top, allm, learned, cv) in e["pred"].items():
            tt = e["t0"] + hz
            if tt > tj[-1, 0] or tt < tj[0, 0]:
                continue
            truth = np.array([np.interp(tt, tj[:, 0], tj[:, k]) for k in (1, 2, 3)])
            pe = perr[hz]
            pe["mixture"].append(float(np.linalg.norm(mix - truth)))
            pe["top"].append(float(np.linalg.norm(top - truth)))
            pe["best_of_n"].append(float(min(np.linalg.norm(x - truth) for x in allm)))
            pe["cv"].append(float(np.linalg.norm(cv - truth)))
            if learned is not None:
                pe["learned"].append(float(np.linalg.norm(learned - truth)))
            d = (truth - mix)[:2]
            pe["inside95"].append(float(d @ np.linalg.solve(mcov[:2, :2] + np.eye(2) * 1e-6, d) <= 5.991))

    # ------------------------------------------------------------- anomaly
    truth_of_track = {tid: Counter(v).most_common(1)[0][0] for tid, v in track_truth.items()}
    events = []
    for a in list(world.agents.values()) + world.retired:
        for kind, ta, tb in a.anomalies:
            events.append((a.id, kind, ta, tb if tb is not None else world.t))
    detected, delays = 0, []
    for aid, kind, ta, tb in events:
        hits = [i for i in insights if truth_of_track.get(i.track_id) == aid and ta - 2 <= i.timestamp <= tb + 30
                and i.kind in ANOMALY_KINDS.get(kind, set())]
        if hits:
            detected += 1
            delays.append(min(i.timestamp for i in hits) - ta)
    ins_tp = 0
    ins_kind = Counter()
    ins_kind_tp = Counter()
    fp_detail = Counter()
    cls_of = {a.id: a.cls.value for a in list(world.agents.values()) + world.retired}
    for i in insights:
        aid = truth_of_track.get(i.track_id)
        ok = any(e[0] == aid and e[2] - 2 <= i.timestamp <= e[3] + 30 for e in events) if aid else False
        if not ok:
            fp_detail[f"{i.kind}|{cls_of.get(aid, 'none')}"] += 1
        ins_kind[i.kind] += 1
        if ok:
            ins_kind_tp[i.kind] += 1
            ins_tp += 1

    return {
        "seed": seed, "wall_s": wall, "sim_s": duration, "mot": dict(m), "motp": motp, "cls": dict(cls_hits),
        "intent": dict(intent_hits), "intent_conf": {f"{k[0]}|{k[1]}": v for k, v in intent_conf.items()},
        "perr": {hz: {k: v for k, v in d.items()} for hz, d in perr.items()},
        "anomaly": {"events": len(events), "detected": detected, "delays": delays, "insights": len(insights),
                    "insights_tp": ins_tp, "by_kind": dict(ins_kind), "by_kind_tp": dict(ins_kind_tp), "fp_detail": dict(fp_detail),
                    "event_kinds": dict(Counter(e[1] for e in events))},
        "tracker": vars(runner.pipeline.tracker.stats),
    }


def summarize(results: list[dict]) -> dict[str, Any]:
    mot = Counter()
    for r in results:
        mot.update(r["mot"])
    motp_all = [x for r in results for x in r["motp"]]
    gt = max(mot["gt"], 1)
    classes = sorted({k[len("motp_n_"):] for r in results for k in r["mot"] if k.startswith("motp_n_")})
    tracking = {
        "MOTA": 1 - (mot["fn"] + mot["fp"] + mot["idsw"]) / gt,
        "MOTP_m": float(np.mean(motp_all)) if motp_all else None,
        "coverage": 1 - mot["fn"] / gt,
        "false_track_rate": mot["fp"] / gt,
        "id_switches": mot["idsw"],
        "id_switches_per_object_hour": mot["idsw"] / max(gt / 3600, 1e-9),
        "position_error_by_class_m": {c: mot[f"motp_sum_{c}"] / max(mot[f"motp_n_{c}"], 1) for c in classes},
    }
    cls = Counter()
    for r in results:
        cls.update(r["cls"])
    per_class = {k[2:]: cls[f"ok_{k[2:]}"] / v for k, v in cls.items() if k.startswith("n_") and v}
    classification = {"accuracy": cls["ok"] / max(cls["n"], 1), "samples": cls["n"], "per_class_accuracy": per_class}
    it = Counter()
    ic = Counter()
    for r in results:
        it.update(r["intent"])
        ic.update(r["intent_conf"])
    by_true = defaultdict(lambda: [0, 0])
    for k, v in ic.items():
        tru, pre = k.split("|")
        by_true[tru][1] += v
        by_true[tru][0] += v if tru == pre else 0
    confusions = sorted(((v, k) for k, v in ic.items() if k.split("|")[0] != k.split("|")[1]), reverse=True)[:8]
    intent = {"accuracy": it["ok"] / max(it["n"], 1), "samples": it["n"],
              "per_intent_accuracy": {k: v[0] / v[1] for k, v in by_true.items() if v[1]},
              "top_confusions": [{"true": k.split("|")[0], "predicted": k.split("|")[1], "count": v} for v, k in confusions]}
    pred = {}
    for hz in HORIZONS:
        agg = defaultdict(list)
        for r in results:
            for k, v in r["perr"].get(hz, {}).items():
                agg[k].extend(v)
        if agg:
            pred[f"{int(hz)}s"] = {k: float(np.mean(v)) for k, v in agg.items() if k != "inside95"} | {
                "median_mixture": float(np.median(agg["mixture"])) if agg["mixture"] else None,
                "coverage_95": float(np.mean(agg["inside95"])) if agg["inside95"] else None, "n": len(agg["mixture"])}
    an = Counter()
    delays, by_kind, by_kind_tp, ev_kinds, fpd = [], Counter(), Counter(), Counter(), Counter()
    for r in results:
        a = r["anomaly"]
        an.update({k: a[k] for k in ("events", "detected", "insights", "insights_tp")})
        delays += a["delays"]
        by_kind.update(a["by_kind"])
        by_kind_tp.update(a["by_kind_tp"])
        ev_kinds.update(a["event_kinds"])
        fpd.update(a.get("fp_detail", {}))
    anomaly = {"events": an["events"], "event_recall": an["detected"] / max(an["events"], 1),
               "insight_precision": an["insights_tp"] / max(an["insights"], 1), "insights": an["insights"],
               "median_delay_s": float(np.median(delays)) if delays else None,
               "insights_by_kind": {k: {"count": v, "precision": by_kind_tp[k] / v} for k, v in by_kind.items()},
               "injected_by_kind": dict(ev_kinds), "unmatched_insights_by_kind_class": dict(fpd.most_common(15))}
    wall = sum(r["wall_s"] for r in results)
    sim = sum(r["sim_s"] for r in results)
    return {"episodes": len(results), "tracking": tracking, "classification": classification, "intent": intent,
            "prediction": pred, "anomaly": anomaly, "performance": {"wall_s_per_sim_s": wall / max(sim, 1), "realtime_factor": sim / max(wall, 1e-9)}}


def to_markdown(summary: dict, scenario: str, duration: float) -> str:
    t, c, i, p, a = summary["tracking"], summary["classification"], summary["intent"], summary["prediction"], summary["anomaly"]
    lines = [f"# SkyWatch benchmark - `{scenario}`", "",
             f"{summary['episodes']} episodes x {duration:.0f} s simulated, independent seeds (not used for training). "
             f"Each episode ran the full pipeline at {summary['performance']['realtime_factor']:.1f}x real time "
             f"(episodes run in parallel, one process each).", "",
             "## Tracking", "", "| MOTA | MOTP | Coverage | False-track rate | ID switches | ID sw. / object-hour |", "|---|---|---|---|---|---|",
             f"| {t['MOTA']:.3f} | {t['MOTP_m']:.1f} m | {t['coverage']:.3f} | {t['false_track_rate']:.3f} | {t['id_switches']} | {t['id_switches_per_object_hour']:.2f} |", "",
             "Mean position error by class: " + ", ".join(f"{k.replace('_', ' ')} {v:.1f} m" for k, v in sorted(t["position_error_by_class_m"].items())), "",
             "## Classification", "", f"Accuracy **{c['accuracy']:.3f}** over {c['samples']} track-seconds (tracks older than 10 s).", "",
             "| Class | Accuracy |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v:.3f} |" for k, v in sorted(c["per_class_accuracy"].items())] + [
             "", "## Intent inference", "", f"Accuracy **{i['accuracy']:.3f}** over {i['samples']} track-seconds (operated aircraft, tracks older than 30 s).", "",
             "| True intent | Accuracy |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v:.3f} |" for k, v in sorted(i["per_intent_accuracy"].items())] + [
             "", "## Trajectory prediction (mean 3-D error vs. ground truth, metres)", "",
             "| Horizon | Mixture | Top hypothesis | Learned only | Const. velocity | Best-of-N (oracle) | 95% region coverage | n |", "|---|---|---|---|---|---|---|---|"]
    for hz, v in p.items():
        lines.append(f"| {hz} | {v.get('mixture', float('nan')):.1f} | {v.get('top', float('nan')):.1f} | {v.get('learned', float('nan')):.1f} | {v.get('cv', float('nan')):.1f} | {v.get('best_of_n', float('nan')):.1f} | {v.get('coverage_95') or 0:.2f} | {v['n']} |")
    lines += ["", "## Anomaly insights", "",
              f"{a['events']} injected anomalies; event recall **{a['event_recall']:.2f}**, insight precision **{a['insight_precision']:.2f}** "
              f"({a['insights']} insights), median detection delay {a['median_delay_s'] if a['median_delay_s'] is not None else float('nan'):.1f} s.", "",
              "| Insight kind | Count | Precision |", "|---|---|---|"] + [f"| {k.replace('_', ' ')} | {v['count']} | {v['precision']:.2f} |" for k, v in sorted(a["insights_by_kind"].items())]
    lines += ["", "Precision counts an insight as correct only if it coincides with an *injected* anomaly; restricted-area entries, "
              "genuine coverage gaps and naturally unusual (but uninjected) behaviour therefore count against it.", ""]
    return "\n".join(lines)


def run_benchmark(cfg: dict, scenario: str = "urban_mixed", episodes: int = 4, duration: float = 600.0, workers: int = 4,
                  seed0: int = 5000, out_dir: str = "reports") -> dict:
    jobs = [(cfg, scenario, seed0 + i, duration) for i in range(episodes)]
    print(f"Benchmark: {episodes} x {duration:.0f} s of '{scenario}' on {workers} workers ...")
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(run_episode, jobs))
    else:
        results = [run_episode(j) for j in jobs]
    summary = summarize(results)
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark_{scenario}.json").write_text(json.dumps(summary, indent=1))
    md = to_markdown(summary, scenario, duration)
    (out / f"benchmark_{scenario}.md").write_text(md)
    print(md)
    return summary
