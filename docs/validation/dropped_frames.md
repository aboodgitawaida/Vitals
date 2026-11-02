# Dropped frames (validation split)

`python dropped_frames.py --data mcd_features`. Flicker removal: `vitals_dsp.deflicker`.

80 recordings (validation), 15 s windows

| Frames | Pipeline | without flicker removal: MAE / within 5 BPM | with: MAE / within 5 BPM |
| --- | --- | --- | --- |
| all frames | network | 4.39 / 75% | 4.39 / 75% |
| all frames | hand-made | 9.79 / 58% | 9.79 / 58% |
| 30 -> 24 fps (every 5th dropped) | network | 9.45 / 61% | 5.68 / 70% |
| 30 -> 24 fps (every 5th dropped) | hand-made | 11.64 / 54% | 11.00 / 55% |
| 30% dropped at random | network | 10.87 / 56% | 5.80 / 70% |
| 30% dropped at random | hand-made | 11.00 / 54% | 10.69 / 55% |
