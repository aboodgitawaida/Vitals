"""
make_test_video.py -- turn a still photo of a face into a test video with a KNOWN heart rate.

Why: a pipeline check end to end -- real FaceMesh, real video decoding, real
compression -- where the right answer is known exactly. Same idea as the
check described by Neutropic for its rPPG tool. It is NOT a validation
study: a synthetic clip has no real blood flow, no talking, no skin-tone
range. For accuracy claims use real data (evaluate_video.py on UBFC-rPPG, or
your own recordings next to a pulse oximeter).

What the clip contains:
  * the face's skin changes colour at the chosen heart rate, in the
    direction real blood volume changes colour (Wang et al. 2017),
    with a little natural beat-to-beat variation
  * slow lighting drift on the whole picture (so the method can't just
    find a clean sine), camera noise, 8-bit rounding, lossy compression
  * optional: small head sway (--motion) and lamp-like colour flicker (--flicker)

    python make_test_video.py selfie.jpg --bpm 72 --out test_videos/selfie_72
    python evaluate_video.py test_videos/selfie_72

Output folder (UBFC-rPPG layout, so evaluate_video.py reads it directly):
    vid.avi            the clip (MJPG)
    ground_truth.txt   line 1: pulse waveform, line 2: heart rate, line 3: time (s)
"""

import argparse
import os

import cv2
import numpy as np

PULSE_DIRECTION_RGB = np.array([0.33, 0.77, 0.53]) / 0.77      # relative R, G, B change, green = 1
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def face_mask(image, box=None):
    """Soft mask (0..1) over the face. Uses FaceMesh's face outline, or a box (x, y, w, h as fractions)."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.float32)
    if box is None:
        import mediapipe as mp
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            res = fm.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise SystemExit("No face found in the photo. Use a clear, frontal, well-lit selfie.")
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_OVAL], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
    else:
        x, y, bw, bh = box
        cv2.ellipse(mask, (int((x + bw / 2) * w), int((y + bh / 2) * h)),
                    (int(bw / 2 * w), int(bh / 2 * h)), 0, 0, 360, 1.0, -1)
    k = max(3, (w // 60) | 1)                              # feather the edge
    return cv2.GaussianBlur(mask, (k, k), 0)


def make_video(image, out_dir, bpm=72.0, seconds=30.0, fps=30.0, amplitude=0.003, motion=False,
               flicker=False, noise=1.5, seed=0, box=None, size=(1280, 720)):
    rng = np.random.default_rng(seed)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = face_mask(image.astype(np.uint8), box)[:, :, None]

    n = int(seconds * fps)
    t = np.arange(n) / fps
    # Natural variation: heart rate wanders a few BPM (breathing-linked + slow drift).
    hr = bpm + 1.5 * np.sin(2 * np.pi * 0.25 * t) + 2.0 * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 6))
    phase = 2 * np.pi * np.cumsum(hr / 60.0) / fps
    wave = np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)              # sharper upstroke, 2nd harmonic
    drift = 1 + 0.02 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
    bgr_dir = PULSE_DIRECTION_RGB[::-1]

    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(os.path.join(out_dir, "vid.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        light = np.full(3, drift[i], np.float32)
        if flicker:                                                     # warm/cool wobble, in band
            light *= 1 + 0.004 * np.sin(2 * np.pi * 1.7 * t[i]) * np.array([-0.3, 0.45, 1.0])
        pulse = 1 + amplitude * wave[i] * bgr_dir
        frame = image * light * (1 + mask * (pulse - 1))
        if motion:                                                      # small sway, like sitting naturally
            dx, dy = 3 * np.sin(2 * np.pi * 0.3 * t[i]), 2 * np.sin(2 * np.pi * 0.17 * t[i] + 1)
            frame = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), size,
                                   borderMode=cv2.BORDER_REFLECT)
        frame += rng.normal(0, noise, frame.shape).astype(np.float32)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    with open(os.path.join(out_dir, "ground_truth.txt"), "w") as f:     # UBFC-rPPG DATASET_2 layout
        f.write(" ".join(f"{v:.6f}" for v in wave) + "\n")
        f.write(" ".join(f"{v:.3f}" for v in hr) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in t) + "\n")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", help="a clear, frontal photo of a face")
    ap.add_argument("--bpm", type=float, default=72.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--amplitude", type=float, default=0.003,
                    help="pulse colour change as a fraction (real skin: roughly 0.001-0.005)")
    ap.add_argument("--motion", action="store_true", help="add a small head sway")
    ap.add_argument("--flicker", action="store_true", help="add a lamp-like colour flicker in the heart-rate band")
    ap.add_argument("--out", default=None, help="output folder (default test_videos/<photo>_<bpm>)")
    args = ap.parse_args()
    image = cv2.imread(args.photo)
    if image is None:
        raise SystemExit(f"Could not read {args.photo}")
    out = args.out or os.path.join("test_videos", f"{os.path.splitext(os.path.basename(args.photo))[0]}_{args.bpm:.0f}")
    make_video(image, out, args.bpm, args.seconds, amplitude=args.amplitude, motion=args.motion, flicker=args.flicker)
    print(f"wrote {out}/vid.avi and ground_truth.txt -> now run: python evaluate_video.py {out}")


if __name__ == "__main__":
    main()
