"""
vitals_app.py -- Vitals from Video, v5 (Phase 1: tracker, motion correction, HD sampling).

webcam (1280x720) -> FaceMesh on a 640-wide copy (fast)
       -> 16 skin patches averaged on the FULL-resolution frame (more pixels = less noise)
       + background wall reference + head pose (x, y, scale, roll)
       -> rolling 15 s buffer (rides out brief face loss; fast-motion frames blanked)
       -> per patch: remove what the wall (room light) and head pose (fidgeting) explain
       -> inverse-noise-weighted pooling -> POS or green (auto-selected)
       -> harmonic-aware FFT peak -> tracker (lock / hold / lost) -> dashboard

Run:   python vitals_app.py               (HD sampling, 1280x720)
       python vitals_app.py --res 640x480 (old resolution, for A/B comparison)
Keys:  q quit
       m method: AUTO -> POS -> GREEN
       b background cancellation on/off
       l light mode (white full-screen dashboard = your screen becomes a front light)
       r show/hide the patch SNR map
       s start/stop recording (average colours only, never images) -> replay.py

Camera-based vitals ESTIMATION for learning/portfolio purposes. Not a medical
device. All processing happens locally; no video is saved or uploaded.
"""

import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from dashboard import compose
from face_regions import (PATCHES, REGIONS, assess_lighting, background_color, draw_overlay,
                          extract_patch_colors, landmarks_to_array, patch_region)
from vitals_dsp import analyze_heart_rate
from vitals_stream import HeartRateTracker, SampleBuffer

# ---------------- Settings you can tune ----------------
WINDOW_SEC = 15          # seconds of history per estimate (15 s -> ~4 BPM frequency resolution)
MIN_SEC = 5              # first estimate after 5 s (the tracker asks more of short windows)
UPDATE_EVERY = 0.5       # seconds between estimates
SNR_GOOD_DB = 3.0        # at or above: "good signal" (show/hide thresholds live in HeartRateTracker)
FACE_GAP_SEC = 1.0       # face missing longer than this -> start over (shorter gaps are bridged)
MOTION_LIMIT = 0.025     # nose movement per frame (fraction of face width) that pauses readings
SMALL_MOTION = 0.008     # smaller movement: no pause, but auto mode then prefers POS
MOTION_HOLD = 1.5        # seconds to pause readings after big movement
SMOOTHING = 0.5          # landmark smoothing 0..1 (higher = steadier patches, more lag)
FRAME_WIDTH = 640        # FaceMesh and the dashboard use a copy this wide (keeps them fast)
WINDOW = "Vitals from Video"
METHODS = ["auto", "pos", "green"]


def open_camera(width, height):
    """
    Why ask for HD: each patch's colour is an AVERAGE over its pixels, and
    averaging N pixels shrinks independent sensor noise by sqrt(N). 1280x720
    has 3x the pixels of 640x480, so in theory up to ~1.7x less noise per
    patch (~5 dB). The camera's own processing already smooths some noise
    away, so the real gain is smaller -- compare with --res 640x480 + replay.py.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Check camera permissions and that no other app is using it.")
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"camera: asked for {width}x{height}, got {got[0]}x{got[1]}")
    return cap


def head_pose(lm, w, h):
    """
    Four numbers describing where the head is, for motion correction:
    nose x, nose y, face width (moving closer/further) and roll (tilt), all
    relative to the frame so they don't depend on the resolution.
    """
    left_eye, right_eye = lm[33], lm[263]
    roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    return np.array([lm[1][0] / w, lm[1][1] / h, np.linalg.norm(lm[234] - lm[454]) / w, roll])


def set_window_mode(light):
    """Light mode = full-screen white dashboard; normal = regular window."""
    cv2.destroyWindow(WINDOW)
    if light:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except cv2.error:
            print("Full screen not supported here - maximise the window manually.")
    else:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)


def start_recording():
    """
    Save raw per-patch and background colours every frame. replay.py can then
    re-run ANY version of the pipeline on exactly the same session.
    """
    os.makedirs("recordings", exist_ok=True)
    path = time.strftime("recordings/session_%Y%m%d_%H%M%S.csv")
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    header = ["time_s"]
    for name in PATCHES + ["background"]:
        header += [f"{name}_r", f"{name}_g", f"{name}_b"]
    # face/moving/state let replay.py reproduce exactly what the screen did.
    writer.writerow(header + ["bpm_shown", "snr_db", "method_used", "light_mode", "face", "moving", "state",
                              "pose_x", "pose_y", "pose_scale", "pose_roll", "frame_w", "frame_h"])
    print("recording to", path)
    return f, writer


def region_summary(analysis):
    """Best patch SNR and total weight share per region, for the dashboard."""
    summary = {r: (None, 0.0) for r in REGIONS}
    if not analysis:
        return summary
    for name, snr in analysis["patch_snr"].items():
        if snr is None:
            continue
        region = patch_region(name)
        best, share = summary[region]
        summary[region] = (snr if best is None else max(best, snr),
                           share + analysis["patch_weight"].get(name, 0.0))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Camera-based heart-rate estimation (not a medical device).")
    ap.add_argument("--res", default="1280x720", help="camera resolution, e.g. 1280x720 (default) or 640x480")
    ap.add_argument("--no-motion-correction", action="store_true", help="turn off head-pose regression")
    args = ap.parse_args()
    cam_w, cam_h = (int(v) for v in args.res.lower().split("x"))
    use_motion = not args.no_motion_correction

    cap = open_camera(cam_w, cam_h)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    buffer = SampleBuffer(PATCHES, WINDOW_SEC, FACE_GAP_SEC)   # rolling per-patch colours
    tracker = HeartRateTracker()                               # decides what number to show
    reading = tracker.reading(0.0)
    trend = deque()                                  # (time, bpm or None) for the 60 s trend
    analysis, smooth_lm, prev_nose = None, None, None
    method_idx, use_background, light_mode, show_map = 0, True, False, True
    lighting = ("none", "Light: waiting for face")
    last_update, last_motion, last_small_motion = 0.0, -99.0, -99.0
    last_frame_time, fps = None, 0.0
    rec_file = rec_writer = None
    t0 = time.monotonic()

    def reset_all():
        nonlocal analysis, reading
        buffer.reset()
        tracker.reset()
        analysis, reading = None, tracker.reading(0.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic() - t0
        if last_frame_time is not None:
            fps = 0.9 * fps + 0.1 / max(now - last_frame_time, 1e-3)
        last_frame_time = now

        full = cv2.flip(frame, 1)                          # mirror view, full resolution
        h, w = full.shape[:2]
        scale = FRAME_WIDTH / w
        small = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # FaceMesh runs on the small copy (fast). Its landmarks are fractions of the image,
        # so they map straight onto the full-resolution frame, where colours are measured.
        result = face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        frame = full
        status, colors, bg_rgb, moving_now, pose = "", None, None, False, None
        method = METHODS[method_idx]

        if result.multi_face_landmarks:
            lm = landmarks_to_array(result.multi_face_landmarks[0], w, h)

            # Motion check: head movement corrupts the signal far more than a pulse changes it.
            face_width = np.linalg.norm(lm[234] - lm[454])
            movement = 0.0 if prev_nose is None else np.linalg.norm(lm[1] - prev_nose) / face_width
            if movement > MOTION_LIMIT:
                last_motion = now
            if movement > SMALL_MOTION:
                last_small_motion = now
            prev_nose = lm[1].copy()

            # Landmark smoothing: FaceMesh points jitter a pixel or two per frame, making patch
            # edges flicker on/off skin. Average with the previous frame unless the head really moved.
            big_move = movement > MOTION_LIMIT
            smooth_lm = lm if smooth_lm is None or big_move else SMOOTHING * smooth_lm + (1 - SMOOTHING) * lm

            colors, patches, regions, stats = extract_patch_colors(frame, smooth_lm)
            bg_rgb, bg_boxes = background_color(frame, smooth_lm)
            pose = head_pose(lm, w, h)                     # raw (unsmoothed) landmarks: real head motion

            # Fast-motion frames are stored as gaps (filled by interpolation), so a
            # head turn doesn't poison the next 15 s of windows.
            moving_now = big_move
            if big_move:
                buffer.add_unusable(now)
            else:
                buffer.add(now, colors, bg_rgb, pose)

            have = buffer.span()
            moving = now - last_motion < MOTION_HOLD
            if have < MIN_SEC:
                lighting = assess_lighting(stats, fps)
                reading = tracker.update(now)
            elif now - last_update >= UPDATE_EVERY:
                last_update = now
                lighting = assess_lighting(stats, fps)
                # If the head moved at all during this window, make green prove itself (see vitals_dsp).
                margin = 3.0 if now - last_small_motion < WINDOW_SEC else 0.0
                times, traces, background = buffer.arrays()
                analysis = analyze_heart_rate(
                    times, traces, background, method=method, prev_bpm=reading.prior_bpm,
                    use_background=use_background, green_margin_db=margin,
                    pose=buffer.pose_array(), use_motion=use_motion)
                reading = tracker.update(now, analysis and analysis["bpm"],
                                         analysis and analysis["snr_db"], usable=not moving,
                                         window_sec=have)
                trend.append((now, reading.bpm))

            if show_map:
                patch_snr = analysis["patch_snr"] if analysis else {}
                frame = draw_overlay(frame, regions, patches, patch_snr, bg_boxes)

            if have < MIN_SEC:
                status = f"Hold still... collecting {have:.0f}/{MIN_SEC} s"
            elif moving:
                status = "Movement detected - hold still"
            elif reading.state == "holding":
                status = "Weak signal - holding last reading"
            elif reading.bpm is None:
                status = lighting[1] if lighting[0] != "good" else "Weak signal - keep still"
        else:
            # Brief loss (blink, hand passing): keep the data and ride it out.
            prev_nose = None
            if buffer.add_missing(now):
                status = "Face lost - hold still"
                reading = tracker.reading(now)
            else:
                reset_all()
                smooth_lm = None
                status = "No face detected"

        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)   # display size

        while trend and now - trend[0][0] > 60:
            trend.popleft()

        bpm = reading.bpm
        snr = analysis["snr_db"] if analysis else None
        used = analysis["method_used"] if analysis else (method if method != "auto" else "pos")
        if bpm is None:
            level = "none"
        elif reading.state == "holding":
            level = "fair"
        else:
            level = "good" if snr is not None and snr >= SNR_GOOD_DB else "fair"
        info = {
            "bpm": bpm, "snr_db": snr, "fps": fps, "status": status, "now": now,
            "level": level, "state": reading.state,
            "method_label": f"AUTO>{used.upper()}" if method == "auto" else method.upper(),
            "method_used": used, "lighting": lighting,
            "features": (f"bg {'ON' if use_background else 'OFF'}  "
                         + (f"motion {analysis['motion_patches']}/{len(PATCHES)}"
                            if use_motion and analysis else ("motion OFF" if not use_motion else "motion -"))),
            "recording": rec_writer is not None, "trend": trend,
            "region_stats": region_summary(analysis),
            "freqs": analysis["freqs"] if analysis else None,
            "power": analysis["power"] if analysis else None,
            "pulse": analysis["pulse"] if analysis else None,
            "peak_bpm": analysis["bpm"] if analysis else None,
        }
        cv2.imshow(WINDOW, compose(frame, info, theme="light" if light_mode else "dark"))

        if rec_writer is not None:
            # Every frame is written, including frames with no face (empty colours),
            # so replay.py sees exactly the gaps the app saw.
            row = [f"{now:.4f}"]
            face_colors = [colors[n] for n in PATCHES] if colors is not None else [None] * len(PATCHES)
            for c in face_colors + [bg_rgb]:
                row += [f"{v:.4f}" for v in c] if c is not None else ["", "", ""]
            row += [f"{bpm:.1f}" if bpm is not None else "", f"{snr:.2f}" if snr is not None else "",
                    used, int(light_mode), int(colors is not None), int(moving_now), reading.state]
            row += [f"{v:.6f}" for v in pose] if pose is not None else ["", "", "", ""]
            row += [w, h]
            rec_writer.writerow(row)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            method_idx = (method_idx + 1) % len(METHODS)
            tracker.reset()
            print("method:", METHODS[method_idx])
        elif key == ord("b"):
            use_background = not use_background
            tracker.reset()
            print("background cancellation:", use_background)
        elif key == ord("l"):
            light_mode = not light_mode
            set_window_mode(light_mode)
            # The camera re-adjusts exposure after a big lighting change; old samples would
            # contain that jump, so start fresh.
            reset_all()
            print("light mode:", light_mode)
        elif key == ord("r"):
            show_map = not show_map
        elif key == ord("s"):
            if rec_writer is None:
                rec_file, rec_writer = start_recording()
            else:
                rec_file.close()
                rec_file = rec_writer = None
                print("recording stopped")

    if rec_file:
        rec_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
