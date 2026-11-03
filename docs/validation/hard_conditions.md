# Harder conditions (validation split)

MCD-rPPG filmed every person with three cameras at once, each with its own
frame-synced finger PPG. The network was trained on the front camera only. The
left and right cameras show the same people with the face turned and smaller,
in other light. "Chin cut off" is the front video moved down until the face
sits at the bottom edge, like a laptop camera tilted down at someone. Same 40
validation people in every row; nothing was tuned on these numbers except the
face-loss bridge below, which changes no accuracy number.

```
python build_mcd_dataset.py --out mcd_left  --camera USBVideo    --splits mcd_features/splits.json --only validation
python build_mcd_dataset.py --out mcd_right --camera IriunWebcam --splits mcd_features/splits.json --only validation
python build_mcd_dataset.py --out mcd_low   --transform low --front mcd_features --splits mcd_features/splits.json --only validation
python hard_conditions.py --split validation --cond front=mcd_features --cond left=mcd_left \
    --cond right=mcd_right --cond "chin cut off"=mcd_low --gap 1 --gap 3
```

**Start over after the face is gone longer than 1 s** (the app used 1 s; it now uses 3 s). 95% intervals resample people.

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| front | 80 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 91% [86%-95%] | 79% |
| left | 79 | 98% | 2% | 14% | 0.14 | 47% | 75% | 5.5 | 7.0 s (never 3%) | 56% [48%-63%] | 37% |
| right | 80 | 100% | 0% | 0% | 0.00 | 58% | 71% | 6.0 | 7.6 s (never 0%) | 55% [47%-63%] | 33% |
| chin cut off | 78 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 92% [87%-95%] | 79% |

**Start over after the face is gone longer than 3 s** (the app used 1 s; it now uses 3 s). 95% intervals resample people.

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| front | 80 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 91% [86%-95%] | 79% |
| left | 79 | 98% | 2% | 14% | 0.03 | 47% | 75% | 5.5 | 7.0 s (never 3%) | 56% [48%-63%] | 37% |
| right | 80 | 100% | 0% | 0% | 0.00 | 58% | 71% | 6.0 | 7.6 s (never 0%) | 55% [47%-63%] | 33% |
| chin cut off | 78 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 92% [87%-95%] | 79% |

First number: within the first 75 s of each recording.

What this shows:

- **Chin cut off costs nothing.** The forehead and cheeks are what the app
  measures; with them in view the result is the same as a centred face.
- **Side views are much harder.** About 55% of 15 s windows within 5 BPM,
  against 91% from the front (the hand-made pipeline: about 35%). The network
  never saw a turned face in training. Its confidence still drops, so it shows
  a steady number about half the time instead of 92%, and those numbers are
  less accurate (71-75% within 5 BPM, against 92%). Training on the side
  cameras too is the obvious next step.
- **Start-overs.** Dataset participants sit still, so the face is rarely lost
  for long. On the left camera, bridging up to 3 s instead of 1 s cut
  start-overs from 0.14 to 0.03 per minute and changed no accuracy number.
- **Not covered:** looking down at a phone (no pitch-down data), glasses and
  hair over the forehead were not measured separately, and these numbers are
  validation only: the held-out test people were not filmed through the side
  cameras in this benchmark.

Timing: about 29% of the dataset's videos say 24 fps in their header but were
filmed at 30 (some right-camera videos the other way round). Frame times come
from the dataset's per-frame timestamps (`fetch_mcd_rppg.true_fps`). Before
that fix the right camera looked far worse (14% within 5 BPM) because the
reference itself was off by 25%.
