# Dropped frames (validation split)

`python dropped_frames.py --data mcd_features`. Flicker removal: `vitals_dsp.deflicker`.

80 recordings (validation), 15 s windows

| Frames | Pipeline | without flicker removal: MAE / within 5 BPM | with: MAE / within 5 BPM |
| --- | --- | --- | --- |
| all frames | network | 2.13 / 91% | 2.13 / 91% |
| all frames | hand-made | 6.49 / 78% | 6.49 / 78% |
| 30 -> 24 fps (every 5th dropped) | network | 10.65 / 65% | 2.92 / 88% |
| 30 -> 24 fps (every 5th dropped) | hand-made | 8.61 / 75% | 8.01 / 76% |
| 30% dropped at random | network | 7.13 / 75% | 2.95 / 88% |
| 30% dropped at random | hand-made | 7.92 / 75% | 7.30 / 76% |
