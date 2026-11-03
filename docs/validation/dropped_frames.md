# Dropped frames (validation split)

`python dropped_frames.py --data mcd_features`. Flicker removal: `vitals_dsp.deflicker`.

80 recordings (validation), 15 s windows

| Frames | Pipeline | without flicker removal: MAE / within 5 BPM | with: MAE / within 5 BPM |
| --- | --- | --- | --- |
| all frames | network | 2.39 / 90% | 2.39 / 90% |
| all frames | hand-made | 6.49 / 78% | 6.49 / 78% |
| 30 -> 24 fps (every 5th dropped) | network | 9.70 / 63% | 3.05 / 88% |
| 30 -> 24 fps (every 5th dropped) | hand-made | 8.61 / 75% | 8.01 / 76% |
| 30% dropped at random | network | 5.65 / 78% | 3.29 / 86% |
| 30% dropped at random | hand-made | 7.92 / 75% | 7.30 / 76% |
