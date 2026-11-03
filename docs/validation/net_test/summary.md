# vitals_net vs the hand-made pipeline: held-out test split

`python compare_net.py --data mcd_features --split test --out docs/validation/net_test`, run after the
network, its tracker thresholds and its 6 s start were fixed on the training and validation splits.
Tracker: enter 0.4, keep 0.3, steady at 0.8 (network); the existing SNR thresholds (hand-made).
"What the app shows" replays every recording second by second through the app's own buffer and tracker.

The test split has now been used twice. The first run (72% of 15 s windows within 5 BPM, MAE 5.5 BPM)
used frame times from the video headers, which are wrong for 29% of the dataset's videos. The frame
timing was fixed, the network retrained with the same recipe, and its tracker re-checked on validation
(same thresholds kept); only then was the test split run again. Nothing was chosen using test results.

200 recordings, 100 people. 95% intervals resample people.

| Window | Pipeline | MAE (BPM) [95% CI] | Within 5 BPM [95% CI] |
| --- | --- | --- | --- |
| 15 s | network | 2.32 [1.75-3.00] | 90% [88%-92%] |
| 15 s | hand-made | 7.08 [5.43-8.57] | 77% [73%-82%] |
| 8 s | network | 4.37 [3.73-5.09] | 80% [77%-83%] |
| 8 s | hand-made | 9.86 [8.27-11.36] | 63% [59%-68%] |

| What the app shows | network | hand-made |
| --- | --- | --- |
| steady share | 90% | 46% |
| steady within5 | 92% | 95% |
| steady MAE | 1.97 | 1.89 |
| settling share | 8% | 17% |
| settling within5 | 67% | 78% |
| first median | 7.0 s | 10.3 s |
| first by15 | 100% | 63% |
| never | 0% | 1% |

| Face brightness | Recordings | network: within 5 BPM (15 s) | hand-made |
| --- | --- | --- | --- |
| 0-79 | 48 | 92% | 79% |
| 80-94 | 76 | 89% | 79% |
| 95-255 | 76 | 90% | 74% |

Network confidence vs how often it is right (15 s windows):

| Confidence | Windows | Within 5 BPM |
| --- | --- | --- |
| 0.0-0.3 | 60 | 30% |
| 0.3-0.5 | 399 | 40% |
| 0.5-0.7 | 707 | 54% |
| 0.7-0.9 | 1196 | 79% |
| 0.9-1.0 | 8639 | 98% |

Extra statistics (same run):

| | network | hand-made |
| --- | --- | --- |
| 15 s windows: RMSE / Pearson r | 8.29 BPM / 0.83 | 16.63 BPM / 0.48 |
| steady numbers: RMSE / Pearson r | 7.20 BPM / 0.87 | 7.88 BPM / 0.80 |
| recordings with 15 s MAE under 5 BPM | 176 / 200 | 120 / 200 |
| at rest: 15 s MAE / within 5 BPM | 2.36 / 90% | 6.60 / 77% |
| after exercise: 15 s MAE / within 5 BPM | 2.29 / 91% | 7.56 / 77% |

Per recording: [`per_recording.csv`](per_recording.csv).
