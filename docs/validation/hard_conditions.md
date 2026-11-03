# Harder conditions

MCD-rPPG filmed every person with three cameras at once, each with its own frame-synced finger
PPG: front, left and right (smaller, turned face, other light; the right one at 24 fps). "Chin cut
off" is the front video moved down until the face sits at the bottom edge, like a laptop camera
tilted down at someone. Same people in every row of a table.

The network is now trained on all three cameras of the 400 training people (`train_net.py --data`
repeated). Before that it saw the front camera only, and side views were its biggest weakness.

```
python build_mcd_dataset.py --out mcd_left  --camera USBVideo    --splits mcd_features/splits.json
python build_mcd_dataset.py --out mcd_right --camera IriunWebcam --splits mcd_features/splits.json
python build_mcd_dataset.py --out mcd_low   --transform low --front mcd_features --splits mcd_features/splits.json --only validation
python hard_conditions.py --split validation --cond front=mcd_features --cond left=mcd_left --cond right=mcd_right --cond "chin cut off"=mcd_low
python hard_conditions.py --split test --max-people 50 --cond front=mcd_features --cond left=mcd_left --cond right=mcd_right
```

Start over after the face is gone longer than 3 s. 95% intervals resample people.

## Held-out test: 50 people never used for training or any choice

Network trained on three camera angles:

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| front | 100 | 100% | 0% | 0% | 0.00 | 89% | 93% | 1.6 | 7.0 s (never 0%) | 91% [87%-94%] | 80% |
| left | 99 | 99% | 2% | 7% | 0.04 | 47% | 87% | 2.6 | 7.5 s (never 4%) | 68% [62%-74%] | 45% |
| right | 99 | 100% | 0% | 1% | 0.00 | 55% | 87% | 2.4 | 7.6 s (never 0%) | 68% [61%-74%] | 44% |

The previous network (front camera only), same recordings:

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| left | 99 | 99% | 2% | 7% | 0.04 | 49% | 77% | 4.8 | 7.5 s (never 3%) | 59% [53%-65%] | 45% |
| right | 99 | 100% | 0% | 1% | 0.00 | 61% | 79% | 4.5 | 7.6 s (never 0%) | 63% [56%-69%] | 44% |

## Validation: 40 people

Network trained on three camera angles:

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| front | 80 | 100% | 0% | 0% | 0.00 | 90% | 93% | 1.8 | 7.0 s (never 0%) | 91% [86%-94%] | 79% |
| left | 79 | 98% | 2% | 14% | 0.03 | 46% | 85% | 3.4 | 7.5 s (never 3%) | 62% [54%-69%] | 37% |
| right | 80 | 100% | 0% | 0% | 0.00 | 50% | 84% | 3.5 | 7.6 s (never 0%) | 62% [54%-69%] | 33% |
| chin cut off | 78 | 100% | 0% | 0% | 0.00 | 90% | 93% | 1.7 | 7.0 s (never 0%) | 90% [86%-94%] | 79% |

The previous network (front camera only):

| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | 15 s windows within 5 BPM: network [95% CI] | hand-made |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| front | 80 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 91% [86%-95%] | 79% |
| left | 79 | 98% | 2% | 14% | 0.03 | 47% | 75% | 5.5 | 7.0 s (never 3%) | 56% [48%-63%] | 37% |
| right | 80 | 100% | 0% | 0% | 0.00 | 58% | 71% | 6.0 | 7.6 s (never 0%) | 55% [47%-63%] | 33% |
| chin cut off | 78 | 100% | 0% | 0% | 0.00 | 92% | 92% | 1.9 | 7.0 s (never 0%) | 92% [87%-95%] | 79% |

## What this shows

- **Side views: 59-63% -> 68% of windows within 5 BPM on held-out people**, and the steady numbers
  the app shows there went from 77-79% to 87% within 5 BPM (average error 4.5-4.8 -> 2.4-2.6 BPM).
  Facing the camera is unchanged (90-91%). A turned face is still harder than a straight one, so the
  network is less sure there and shows a steady number about half the time instead of ~90%.
- **Chin cut off costs nothing.** The forehead and cheeks are what the app measures.
- **Start-overs.** Bridging face losses of up to 3 s (was 1 s) cut start-overs on the left camera
  from 0.14 to 0.03 per minute on validation and changed no accuracy number.
- **Not covered:** looking down at a phone (no pitch-down data); glasses and hair over the forehead
  were not measured separately; the chin-cut-off view was measured on validation only.

Timing: about 29% of the dataset's videos say 24 fps in their header but were filmed at 30 (some
right-camera videos the other way round). Frame times come from the dataset's per-frame timestamps
(`fetch_mcd_rppg.true_fps`). Before that fix the right camera looked far worse (14% within 5 BPM)
because the reference itself was off by 25%.
