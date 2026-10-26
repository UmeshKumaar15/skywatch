# SkyWatch benchmark - `urban_mixed`

4 episodes x 600 s simulated, independent seeds (not used for training). Each episode ran the full pipeline at 4.6x real time (episodes run in parallel, one process each).

## Tracking

| MOTA | MOTP | Coverage | False-track rate | ID switches | ID sw. / object-hour |
|---|---|---|---|---|---|
| 0.884 | 10.4 m | 0.914 | 0.029 | 49 | 2.99 |

Mean position error by class: bird 24.3 m, commercial aircraft 6.1 m, delivery drone 4.1 m, emergency vehicle 7.3 m, helicopter 6.6 m, industrial drone 4.5 m, recreational drone 6.9 m, unknown 10.4 m, weather balloon 18.2 m

## Classification

Accuracy **0.990** over 53036 track-seconds (tracks older than 10 s).

| Class | Accuracy |
|---|---|
| bird | 1.000 |
| commercial aircraft | 1.000 |
| delivery drone | 0.996 |
| emergency vehicle | 0.991 |
| helicopter | 0.981 |
| industrial drone | 0.981 |
| recreational drone | 0.977 |
| unknown | 0.973 |
| weather balloon | 1.000 |

## Intent inference

Accuracy **0.878** over 36577 track-seconds (operated aircraft, tracks older than 30 s).

| True intent | Accuracy |
|---|---|
| anomalous | 0.085 |
| delivery route | 0.961 |
| drifting | 0.963 |
| emergency response | 0.973 |
| holding pattern | 0.463 |
| inspection | 0.721 |
| mapping | 0.863 |
| recreational | 0.921 |
| transit | 0.734 |

## Trajectory prediction (mean 3-D error vs. ground truth, metres)

| Horizon | Mixture | Top hypothesis | Learned only | Const. velocity | Best-of-N (oracle) | 95% region coverage | n |
|---|---|---|---|---|---|---|---|
| 10s | 38.7 | 41.3 | 44.7 | 49.6 | 26.7 | 0.93 | 10163 |
| 20s | 87.1 | 95.7 | 103.7 | 118.2 | 53.6 | 0.93 | 9844 |
| 30s | 147.8 | 163.5 | 171.6 | 203.2 | 86.2 | 0.92 | 9530 |
| 60s | 359.7 | 413.1 | 364.7 | 507.1 | 183.6 | 0.95 | 8600 |

## Anomaly insights

18 injected anomalies; event recall **0.67**, insight precision **0.51** (39 insights), median detection delay 12.5 s.

| Insight kind | Count | Precision |
|---|---|---|
| abrupt maneuver | 2 | 1.00 |
| altitude change | 5 | 0.60 |
| comms loss | 8 | 0.88 |
| excessive acceleration | 12 | 0.42 |
| prolonged hover | 1 | 1.00 |
| route deviation | 5 | 0.40 |
| signal loss | 6 | 0.00 |

Precision counts an insight as correct only if it coincides with an *injected* anomaly; restricted-area entries, genuine coverage gaps and naturally unusual (but uninjected) behaviour therefore count against it.
