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
          and, where the head moved enough to matter, head pose (fidgeting) explain
       -> inverse-noise-variance pooling of patches
       -> POS or green channel (auto: whichever has the cleaner spectrum)
       -> 0.7-4 Hz band -> FFT peak with harmonic check -> SNR
       -> tracker: acquiring / locked / holding / lost -> dashboard
          (first estimate after 5 s, with a stricter bar for short windows)
```

| File | What it does |
| --- | --- |
| `vitals_app.py` | Live app: camera, FaceMesh, dashboard, recording |
| `vitals_dsp.py` | All signal processing (no camera code, fully testable) |
| `vitals_stream.py` | Rolling buffer and display tracker, shared by app, replay and benchmarks |
| `face_regions.py` | Skin regions, patch grid, skin filter, background boxes, lighting coach |
| `dashboard.py` | OpenCV UI, dark and light themes |
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
`--no-motion-correction`.

Keys: `q` quit, `m` method (auto / pos / green), `b` background cancellation,
`l` light mode (white full-screen so the screen lights your face), `r` patch SNR
map, `s` start/stop recording (average colours per frame only, never images).

```zsh
python replay.py recordings/session_YYYYMMDD_HHMMSS.csv --ref 72   # 72 = your watch's BPM
python session_benchmark.py              # simulated sessions: v3 vs v4 vs v5, still and fidgeting
python session_benchmark.py --start      # how fast and how right is the first number?
python session_benchmark.py --vanish     # does the number disappear when the pulse does?
python session_benchmark.py --calibrate  # how often pure noise passes the thresholds
pip install pytest && python -m pytest -q tests
```

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
* **Lag.** The number describes roughly the last 15 seconds, so it reacts to
  changes with a delay of several seconds.
* **Simulated results only so far.** The benchmarks generate data from the
  same physics the methods assume, which flatters them. They test that the
  code does what it is designed to do; they are not accuracy validation.
