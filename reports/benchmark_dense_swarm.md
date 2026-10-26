# SkyWatch benchmark - `dense_swarm`

4 episodes x 600 s simulated, independent seeds (not used for training). Each episode ran the full pipeline at 1.9x real time (episodes run in parallel, one process each).

## Tracking

| MOTA | MOTP | Coverage | False-track rate | ID switches | ID sw. / object-hour |
|---|---|---|---|---|---|
| 0.915 | 8.8 m | 0.948 | 0.031 | 276 | 8.25 |

Mean position error by class: bird 20.5 m, commercial aircraft 5.6 m, delivery drone 4.0 m, industrial drone 2.1 m, recreational drone 6.1 m

## Classification

Accuracy **0.999** over 112266 track-seconds (tracks older than 10 s).

| Class | Accuracy |
|---|---|
| bird | 1.000 |
| commercial aircraft | 1.000 |
| delivery drone | 0.997 |
| industrial drone | 0.999 |
| recreational drone | 0.998 |

## Intent inference

Accuracy **0.884** over 73864 track-seconds (operated aircraft, tracks older than 30 s).

| True intent | Accuracy |
|---|---|
| anomalous | 0.063 |
| delivery route | 0.974 |
| holding pattern | 0.378 |
| inspection | 0.679 |
| mapping | 0.834 |
| recreational | 0.951 |
| transit | 0.765 |

## Trajectory prediction (mean 3-D error vs. ground truth, metres)

| Horizon | Mixture | Top hypothesis | Learned only | Const. velocity | Best-of-N (oracle) | 95% region coverage | n |
|---|---|---|---|---|---|---|---|
| 10s | 36.2 | 39.0 | 39.1 | 47.9 | 25.1 | 0.94 | 21594 |
| 20s | 75.4 | 84.2 | 79.6 | 105.7 | 48.4 | 0.94 | 20957 |
| 30s | 121.1 | 135.5 | 134.2 | 174.1 | 75.2 | 0.93 | 20319 |
| 60s | 285.5 | 326.8 | 367.0 | 425.1 | 157.0 | 0.95 | 18487 |

## Anomaly insights

20 injected anomalies; event recall **0.90**, insight precision **0.34** (94 insights), median detection delay 7.6 s.

| Insight kind | Count | Precision |
|---|---|---|
| abrupt maneuver | 22 | 0.41 |
| altitude change | 15 | 0.40 |
| comms loss | 7 | 0.43 |
| excessive acceleration | 17 | 0.41 |
| prolonged hover | 3 | 1.00 |
| route deviation | 13 | 0.00 |
| signal loss | 6 | 0.00 |
| unusual path | 11 | 0.36 |

Precision counts an insight as correct only if it coincides with an *injected* anomaly; restricted-area entries, genuine coverage gaps and naturally unusual (but uninjected) behaviour therefore count against it.
