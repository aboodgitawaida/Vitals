"""
vitals_app.py -- Vitals from Video, v4 (Phase 1, light-adaptive + tracker).

webcam -> FaceMesh landmarks -> 16 skin patches (hair/shadow/glare removed)
       + background wall reference
       -> rolling 15 s buffer (rides out brief face loss; fast-motion frames blanked)
       -> per patch: cancel room-light changes seen on the wall, measure noise
       -> inverse-noise-weighted pooling -> POS or green (auto-selected)
       -> harmonic-aware FFT peak -> tracker (lock / hold / lost) -> dashboard

Run:   python vitals_app.py
Keys:  q quit
       m method: AUTO -> POS -> GREEN
       b background cancellation on/off
       l light mode (white full-screen dashboard = your screen becomes a front light)
       r show/hide the patch SNR map
       s start/stop recording (average colours only, never images) -> replay.py

Camera-based vitals ESTIMATION for learning/portfolio purposes. Not a medical
device. All processing happens locally; no video is saved or uploaded.
"""

import csv
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
MIN_SEC = 8              # wait for this much data before the first estimate
UPDATE_EVERY = 0.5       # seconds between estimates
SNR_GOOD_DB = 3.0        # at or above: "good signal" (show/hide thresholds live in HeartRateTracker)
FACE_GAP_SEC = 1.0       # face missing longer than this -> start over (shorter gaps are bridged)
MOTION_LIMIT = 0.025     # nose movement per frame (fraction of face width) that pauses readings
SMALL_MOTION = 0.008     # smaller movement: no pause, but auto mode then prefers POS
MOTION_HOLD = 1.5        # seconds to pause readings after big movement
SMOOTHING = 0.5          # landmark smoothing 0..1 (higher = steadier patches, more lag)
FRAME_WIDTH = 640        # frames are resized to this width (keeps FaceMesh fast)
WINDOW = "Vitals from Video"
METHODS = ["auto", "pos", "green"]


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Check camera permissions and that no other app is using it.")
    return cap


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
    writer.writerow(header + ["bpm_shown", "snr_db", "method_used", "light_mode", "face", "moving", "state"])
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
    cap = open_camera()
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

        frame = cv2.flip(frame, 1)                         # mirror view
        scale = FRAME_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status, colors, bg_rgb, moving_now = "", None, None, False
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

            # Fast-motion frames are stored as gaps (filled by interpolation), so a
            # head turn doesn't poison the next 15 s of windows.
            moving_now = big_move
            if big_move:
                buffer.add_unusable(now)
            else:
                buffer.add(now, colors, bg_rgb)

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
                    use_background=use_background, green_margin_db=margin)
                reading = tracker.update(now, analysis and analysis["bpm"],
                                         analysis and analysis["snr_db"], usable=not moving)
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
            "features": f"bg cancel {'ON' if use_background else 'OFF'}",
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
