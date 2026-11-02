# vitals_net vs the hand-made pipeline: held-out test split

`python compare_net.py --data mcd_features --split test --out docs/validation/net_test`, run once, after
the network, its tracker thresholds and its 6 s start were fixed on the training and validation splits.
Tracker: enter 0.4, keep 0.3, steady at 0.8 (network); the existing SNR thresholds (hand-made).
"What the app shows" replays every recording second by second through the app's own buffer and tracker.

200 recordings, 100 people. 95% intervals resample people.

| Window | Pipeline | MAE (BPM) [95% CI] | Within 5 BPM [95% CI] |
| --- | --- | --- | --- |
| 15 s | network | 5.48 [4.43-6.67] | 72% [66%-77%] |
| 15 s | hand-made | 10.34 [8.46-12.24] | 57% [50%-65%] |
| 8 s | network | 6.99 [5.98-8.03] | 64% [59%-69%] |
| 8 s | hand-made | 12.48 [10.74-14.33] | 48% [41%-54%] |

| What the app shows | network | hand-made |
| --- | --- | --- |
| steady share | 65% | 50% |
| steady within5 | 85% | 67% |
| steady MAE | 3.05 | 6.56 |
| settling share | 28% | 17% |
| settling within5 | 52% | 56% |
| first median | 7.0 s | 10.0 s |
| first by15 | 96% | 66% |
| never | 0% | 2% |

| Face brightness | Recordings | network: within 5 BPM (15 s) | hand-made |
| --- | --- | --- | --- |
| 0-79 | 48 | 61% | 34% |
| 80-94 | 76 | 73% | 66% |
| 95-255 | 76 | 77% | 64% |

Network confidence vs how often it is right (15 s windows):

| Confidence | Windows | Within 5 BPM |
| --- | --- | --- |
| 0.0-0.3 | 649 | 25% |
| 0.3-0.5 | 2090 | 40% |
| 0.5-0.7 | 1841 | 58% |
| 0.7-0.9 | 1648 | 74% |
| 0.9-1.0 | 5096 | 95% |

Extra statistics (same run):

| | network | hand-made |
| --- | --- | --- |
| 15 s windows: RMSE / Pearson r | 11.41 BPM / 0.69 | 18.26 BPM / 0.44 |
| steady numbers: RMSE / Pearson r | 7.75 BPM / 0.83 | 12.87 BPM / 0.61 |
| recordings with 15 s MAE under 5 BPM | 121 / 200 | 94 / 200 |
| at rest: 15 s MAE / within 5 BPM | 5.39 / 71% | 9.92 / 57% |
| after exercise: 15 s MAE / within 5 BPM | 5.58 / 72% | 10.76 / 58% |

Per recording: [`per_recording.csv`](per_recording.csv).
