# Vitals from Video

Camera-based heart-rate **estimation** from an ordinary laptop webcam
(remote photoplethysmography, rPPG). Everything runs locally on your machine;
no video is saved or uploaded.

> Not a medical device. Not for health, stress or diagnostic use. Accuracy has
> so far been tested only on simulated data; validation against a reference
> device is Phase 4 of this project.

## How it works

```
webcam 1280x720 -> FaceMesh on a 640-wide copy (fast)
       -> 16 skin patches averaged at full resolution (hair / shadow / glare removed,
          thresholds relative to the person's own skin, not a fixed skin-colour range)
       -> rolling 15 s buffer (bridges face-detection gaps < 1 s, blanks fast-motion frames)
       -> nuisance regression per patch: remove what the background wall (room light)
          and, where the head moved enough to matter, head pose and mouth
          movement (fidgeting, talking) explain
       -> inverse-noise-variance pooling of patches
       -> POS or green channel (auto: whichever has the cleaner spectrum)
       -> 0.7-4 Hz band -> FFT peak with harmonic check -> SNR
       -> cross-check: POS, CHROM and green each find their own peak; if no
          second method agrees (within 5 BPM) the estimate is not trusted
       -> tracker: acquiring / locked / holding / lost -> dashboard
          (first estimate after 5 s, with a stricter bar for short windows)
```

| File | What it does |
| --- | --- |
| `vitals_app.py` | Live app: camera, FaceMesh, dashboard, recording |
| `live_session.py` | The live app's heart-rate logic without camera or UI (shared with the web version) |
| `signal_lite.py` | numpy-only butter / filtfilt / detrend, so the browser needn't download scipy |
| `face_layout.py` | Face regions, patch grid, thresholds, lighting coach (no OpenCV, so the browser can use it) |
| `web/` | Browser version (Vercel): camera + MediaPipe in JS, the Python analysis via Pyodide |
| `vitals_dsp.py` | All signal processing (no camera code, fully testable) |
| `frame_features.py` | Per-frame skin colours, wall, head pose and motion, shared by the app and the video tools |
| `evaluate_video.py` | Accuracy on videos with a reference (UBFC-rPPG dataset, test videos, your own clips) |
| `make_test_video.py` | Turns a selfie into a test video with a known heart rate |
| `vitals_stream.py` | Rolling buffer and display tracker, shared by app, replay and benchmarks |
| `face_regions.py` | Skin regions, patch grid, skin filter, background boxes, lighting coach |
| `dashboard.py` | OpenCV UI, dark and light themes |
| `perf.py` | Stage timer and background analysis process (keeps the camera loop smooth) |
| `liveness.py` | Liveness check prototype: pulse check + random screen-flash challenge; re-scores saved attempts |
| `liveness_benchmark.py` | Simulated real people vs attacks (photo, mask, phone replay, virtual camera, live deepfake) |
| `replay.py` | Re-runs every configuration on a recording against a reference BPM |
| `synthetic_benchmark.py` | Per-window ablation on simulated lighting scenarios |
| `session_benchmark.py` | End-to-end test of what the screen shows on simulated 2-minute sessions |
| `tests/` | Unit and end-to-end tests (run in CI on every push) |

## Run it (macOS)

```zsh
conda create -n vitals python=3.11 -y      # MediaPipe needs Python 3.9-3.12
conda activate vitals
pip install -r requirements.txt
python vitals_app.py
```

Options: `--res 640x480` (old resolution, to A/B against the default 1280x720),
`--no-motion-correction`, `--no-background` (run the analysis inline instead of in
a background process, to compare speed).

Keys: `v` liveness check, `q` quit, `m` method (auto / pos / green), `b` background cancellation,
`l` light mode (white full-screen so the screen lights your face), `r` patch SNR
map, `p` performance readout, `s` start/stop recording (average colours per frame only, never images).

```zsh
python replay.py recordings/session_YYYYMMDD_HHMMSS.csv --ref 72   # 72 = your watch's BPM
python session_benchmark.py              # simulated sessions: v3 vs v4 vs v5, still and fidgeting
python session_benchmark.py --start      # how fast and how right is the first number?
python session_benchmark.py --vanish     # does the number disappear when the pulse does?
python session_benchmark.py --calibrate  # how often pure noise passes the thresholds
pip install pytest && python -m pytest -q tests
```

## Run it in the browser (web version)

`web/` is a browser version that runs on any static host (it is set up for
Vercel). The camera, face tracking (MediaPipe FaceLandmarker) and the
skin-patch colours run in JavaScript; the analysis is **this repo's Python**
(`live_session.py`, `vitals_dsp.py`, `vitals_stream.py`, `face_layout.py`),
run unchanged in a Web Worker with [Pyodide](https://pyodide.org). Video never
leaves the browser; the page only downloads its code and the face model
(~16 MB the first time, cached after; it starts while you read the intro).
Face tracking loads first and measuring starts right away; Python loads
during the first 5 s of collecting and then analyses everything measured so
far. `web/face_layout.json` (generated: `python face_layout.py >
web/face_layout.json`) gives the page the face regions without waiting for
Python.
scipy is not loaded in the browser: `vitals_dsp` falls back to
`signal_lite.py`, numpy copies of the three scipy functions it uses
(checked against scipy by `tests/test_signal_lite.py`).

```zsh
node web/build.mjs                         # assembles dist/ (page + the .py files it needs)
cd dist && python -m http.server 8000      # open http://localhost:8000
```

**Deploy to Vercel:** import the GitHub repo at vercel.com/new and click
Deploy. `vercel.json` already sets the build (`node web/build.mjs`) and the
output folder (`dist`), so there is nothing to configure. Or from a terminal:
`npx vercel` (preview) and `npx vercel --prod`.

Same keys as the desktop app (M, B, L, R, P). Not in the web version yet:
the liveness check and recording. `tests/test_web_features.py` checks the
JavaScript colour measurement against the Python one (needs Node), and
`tests/test_live_session.py` the app logic the browser runs.

## Validation

Accuracy is measured the way rPPG research does it: run the pipeline on
videos recorded with a contact sensor and compare, window by window.

```zsh
python make_test_video.py selfie.jpg --bpm 72          # known-BPM test video from a photo
python evaluate_video.py test_videos/selfie_72
python evaluate_video.py path/to/UBFC-rPPG/DATASET_2   # public research dataset (request access)
python evaluate_video.py my_clip.mp4 --ref-bpm 68      # your own clip + a watch reading
```

It reports, per method and for what the app would display: MAE, RMSE,
Pearson r, share within 5 BPM, and a Bland-Altman plot. The reference heart
rate is computed from the pulse-oximeter waveform over the same 15 s windows.

**Status:** the full video pipeline (decoding, compression, frame features,
analysis, tracker) recovers a known heart rate from synthetic test videos
with under 0.2 BPM error, including head sway and in-band lamp flicker. In
the flicker video, single methods failed in different ways (POS locked onto
the flicker without wall cancellation; CHROM with it) while the full
pipeline stayed correct: that is why the three methods are cross-checked.
Synthetic videos only test the plumbing. **UBFC-rPPG results: not run yet.**

## What this project deliberately does not measure

* **Blood pressure.** Claims exist, but without credible evidence that it
  works across the full range; a dataset of mostly normal readings makes
  any method look accurate.
* **Blood oxygen (SpO2).** Needs calibrated measurements at specific
  wavelengths; a consumer webcam's RGB filters are not that.
* **Heart-rate variability (HRV).** Standard HRV uses 5-minute recordings,
  and at 30 fps beat timing is only accurate to ~33 ms, the same order as
  the beat-to-beat differences HRV measures.
* **Stress, emotion, lying.** Heart rate changes for many reasons;
  inferring mental states from it is not scientifically established.

## Privacy

Everything runs on your machine. Recordings, liveness attempts and the
evaluation cache store per-frame average colours only, never images. If
you record other people, tell them that the video is analysed for heart
rate, and get their consent.

## Liveness check (prototype)

Press `v`. **Step 1** (10-30 s): the screen turns white and the app looks for
a pulse that shows up across the forehead and both cheeks. **Step 2** (~7 s):
the screen flashes a random colour sequence, decided only when the check
starts, and the app checks that the skin's tint follows it within a fraction
of a second.

| Attack | Pulse check | Flash challenge | Result |
| --- | --- | --- | --- |
| Printed photo, mask | no pulse | reflects | never LIVE |
| Phone replaying a video | has a pulse | only glints in 1-3 spots | NOT LIVE |
| Video injected as a virtual camera | has a pulse | never sees the screen | NOT LIVE |
| Real-time face swap of a real person | carried through | carried through | **not caught (blind spot)** |

Simulated results (`python liveness_benchmark.py`, 60 attempts per scenario):
0% of photo / mask / replay / virtual-camera attacks accepted, 0% of real
people called NOT LIVE, typical time to confirm 10 s. Real people confirmed
LIVE, by skin tone and room light:

| | bright | normal | dim |
| --- | --- | --- | --- |
| lighter | 100% | 100% | 100% |
| medium | 100% | 100% | 67% |
| darker | 100% | 15% | 0% |

**Design choice:** a weak pulse can stop a LIVE result but never produces
NOT LIVE. The first version turned "weak" into "fake" and rejected every
simulated darker-skinned person in normal light; now they get NOT CONFIRMED
with advice to add light. NOT LIVE is reserved for strong physical evidence
(the face did not react to the flashes).

Attempts are saved as colours only (`recordings/liveness_*`) and can be
re-scored: `python liveness.py recordings/liveness_YYYYMMDD_HHMMSS`.
This is a research prototype, not a security product; real attacks (a good
OLED phone, a matte photo under a lamp) still need testing in person.

## Known limitations

* **Skin tone.** Melanin absorbs light, so darker skin reflects a weaker pulse
  signal. In simulation, darker skin gets a number on screen far less often
  (it shows "--" rather than guessing). This is a known weakness of camera-based
  rPPG generally, not just this project, and the real-world size of the gap is
  measured in Phase 4.
* **Lighting and motion.** Dim light raises camera noise; movement corrupts the
  signal. The app tells you which one it sees, and it refuses to show a number
  when the signal is too weak.
* **Motion correction assumes small movements.** It removes the part of each
  patch's brightness that follows head pose linearly. Large turns, talking with
  big jaw movement and walking are out of scope. The head also moves very
  slightly with every heartbeat (ballistocardiography); if FaceMesh picked that
  up, the correction could remove a little real pulse. Only real recordings
  (motion correction on vs off in `replay.py`) can show how much this matters.
* **Talking.** Mouth movement is used for motion correction, but in
  simulation talking did not hurt POS in the first place (jaw shading is
  mostly a brightness change). Whether the correction helps on real faces
  is measured with `replay.py` (motion correction on vs off).
* **Lag.** The number describes roughly the last 15 seconds, so it reacts to
  changes with a delay of several seconds.
* **Simulated results only so far.** The benchmarks generate data from the
  same physics the methods assume, which flatters them. They test that the
  code does what it is designed to do; they are not accuracy validation.
