# SkyWatch benchmark - `emergency_response`

4 episodes x 600 s simulated, independent seeds (not used for training). Each episode ran the full pipeline at 7.1x real time (episodes run in parallel, one process each).

## Tracking

| MOTA | MOTP | Coverage | False-track rate | ID switches | ID sw. / object-hour |
|---|---|---|---|---|---|
| 0.839 | 11.8 m | 0.888 | 0.047 | 57 | 5.08 |

Mean position error by class: bird 24.1 m, commercial aircraft 7.1 m, delivery drone 5.3 m, emergency vehicle 7.7 m, helicopter 7.7 m, industrial drone 5.5 m, weather balloon 25.8 m

## Classification

Accuracy **0.999** over 35044 track-seconds (tracks older than 10 s).

| Class | Accuracy |
|---|---|
| bird | 1.000 |
| commercial aircraft | 1.000 |
| delivery drone | 0.999 |
| emergency vehicle | 0.998 |
| helicopter | 1.000 |
| industrial drone | 1.000 |
| weather balloon | 1.000 |

## Intent inference

Accuracy **0.838** over 23895 track-seconds (operated aircraft, tracks older than 30 s).

| True intent | Accuracy |
|---|---|
| anomalous | 0.028 |
| delivery route | 0.971 |
| drifting | 0.641 |
| emergency response | 0.994 |
| holding pattern | 0.566 |
| inspection | 0.823 |
| mapping | 0.880 |
| transit | 0.622 |

## Trajectory prediction (mean 3-D error vs. ground truth, metres)

| Horizon | Mixture | Top hypothesis | Learned only | Const. velocity | Best-of-N (oracle) | 95% region coverage | n |
|---|---|---|---|---|---|---|---|
| 10s | 49.4 | 52.9 | 58.2 | 66.6 | 33.7 | 0.94 | 6679 |
| 20s | 123.5 | 134.9 | 152.2 | 175.6 | 74.3 | 0.94 | 6446 |
| 30s | 223.1 | 242.7 | 270.6 | 314.0 | 126.4 | 0.93 | 6220 |
| 60s | 559.3 | 626.2 | 574.8 | 767.3 | 301.9 | 0.94 | 5553 |

## Anomaly insights

17 injected anomalies; event recall **0.88**, insight precision **0.75** (36 insights), median detection delay 11.8 s.

| Insight kind | Count | Precision |
|---|---|---|
| abrupt maneuver | 6 | 0.83 |
| altitude change | 4 | 0.50 |
| comms loss | 7 | 0.86 |
| excessive acceleration | 11 | 0.91 |
| route deviation | 8 | 0.50 |

Precision counts an insight as correct only if it coincides with an *injected* anomaly; restricted-area entries, genuine coverage gaps and naturally unusual (but uninjected) behaviour therefore count against it.
