"""
vitals_phase1.py -- webcam -> face landmarks -> green-channel signal -> live BPM.

Run:  python vitals_phase1.py
Keys: q = quit,  s = save the raw signal to a CSV (useful for Phase 4 validation)

Camera-based vitals ESTIMATION for learning/portfolio purposes.
It is not a medical device. All processing happens locally; nothing is uploaded.
"""

import csv
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from vitals_dsp import estimate_rate

# ---------------- Settings you can tune ----------------
WINDOW_SEC = 10        # how much history the FFT sees. Longer = finer resolution but slower response
MIN_SEC = 6            # don't show anything until we have at least this much data
UPDATE_EVERY = 0.5     # seconds between BPM recalculations
HR_LOW, HR_HIGH = 0.7, 4.0   # Hz -> 42..240 BPM
QUALITY_MIN = 0.35     # below this we show "--" instead of a BPM. Starting guess; Phase 4 will tune it.
MOTION_LIMIT = 0.02    # per-frame head movement (as a fraction of face width) that counts as "moving"
PANEL_H = 160          # height of the waveform panel under the video

# ---------------- Skin regions (MediaPipe FaceMesh landmark indices) ----------------
# FaceMesh gives 468 landmark points on the face. We pick groups of them that
# outline flat, well-lit skin: forehead and both cheeks. We take the convex hull
# (smallest convex outline) of each group so the order of points doesn't matter.
# Use the green overlay on screen to check these look right on YOUR face.
FOREHEAD = [10, 67, 109, 338, 297, 69, 299, 108, 337, 151]
LEFT_CHEEK = [50, 101, 118, 117, 116, 123, 187, 205, 36]
RIGHT_CHEEK = [280, 330, 347, 346, 345, 352, 411, 425, 266]
REGIONS = [FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK]


def landmarks_to_pixels(face, indices, w, h):
    """FaceMesh returns coordinates in 0..1; convert to pixel positions."""
    return np.array([[face.landmark[i].x * w, face.landmark[i].y * h] for i in indices], dtype=np.int32)


def build_skin_mask(face, w, h):
    """White (255) where skin ROIs are, black elsewhere."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in REGIONS:
        pts = landmarks_to_pixels(face, region, w, h)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask


def draw_waveform(panel, signal, label_color=(0, 255, 0)):
    """Draw the filtered pulse waveform with plain OpenCV (no matplotlib needed)."""
    h, w = panel.shape[:2]
    if signal is None or len(signal) < 2:
        return
    s = signal[-int(WINDOW_SEC * 30):]
    s = (s - s.mean()) / (s.std() + 1e-9)          # normalise so the wave always fills the panel
    s = np.clip(s, -3, 3)
    xs = np.linspace(0, w - 1, len(s)).astype(np.int32)
    ys = (h / 2 - s * (h / 7)).astype(np.int32)
    cv2.polylines(panel, [np.column_stack([xs, ys])], False, label_color, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Check camera permissions and that no other app is using it.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    times = deque()    # timestamps (seconds) of each sample
    values = deque()   # mean green value of the skin at that moment
    bpm_history = deque(maxlen=5)   # median of recent estimates -> steadier display
    filtered_signal, bpm, quality = None, None, 0.0
    last_update, last_motion, prev_nose = 0.0, 0.0, None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                      # mirror view feels natural
        h, w = frame.shape[:2]
        now = time.time() - t0

        # MediaPipe expects RGB; OpenCV gives BGR.
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status = ""

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]

            # --- 1. Motion check: head movement corrupts the signal far more than a pulse does.
            nose = np.array([face.landmark[1].x * w, face.landmark[1].y * h])
            face_width = np.linalg.norm(
                np.array([face.landmark[234].x * w, face.landmark[234].y * h]) -
                np.array([face.landmark[454].x * w, face.landmark[454].y * h]))
            if prev_nose is not None and np.linalg.norm(nose - prev_nose) / face_width > MOTION_LIMIT:
                last_motion = now
            prev_nose = nose

            # --- 2. THE SIGNAL: average green value over the skin pixels.
            # OpenCV stores colour as B,G,R so index 1 is green.
            mask = build_skin_mask(face, w, h)
            green = cv2.mean(frame, mask=mask)[1]
            times.append(now)
            values.append(green)

            # Keep only the last WINDOW_SEC seconds.
            while times and now - times[0] > WINDOW_SEC:
                times.popleft()
                values.popleft()

            # Visual check that the regions sit on skin.
            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

            # --- 3. Every 0.5 s, estimate heart rate from the window.
            enough = len(times) > 1 and (times[-1] - times[0]) >= MIN_SEC
            if enough and now - last_update >= UPDATE_EVERY:
                last_update = now
                est, quality, filtered_signal = estimate_rate(times, values, HR_LOW, HR_HIGH)
                moved_recently = (now - last_motion) < 2.0
                if quality >= QUALITY_MIN and not moved_recently:
                    bpm_history.append(est)
                    bpm = float(np.median(bpm_history))
                else:
                    bpm = None
                    bpm_history.clear()
            if not enough:
                status = f"Collecting data... {len(times) / 30:.0f}/{MIN_SEC}s (hold still)"
            elif (now - last_motion) < 2.0:
                status = "Hold still"
            elif bpm is None:
                status = "Signal too weak (try more even light)"
        else:
            # Face lost: old samples are no longer continuous, so start over.
            times.clear(); values.clear(); bpm_history.clear()
            filtered_signal, bpm, prev_nose = None, None, None
            status = "No face detected"

        # ---------------- Draw the UI ----------------
        panel = np.zeros((PANEL_H, w, 3), dtype=np.uint8)
        draw_waveform(panel, filtered_signal)
        bpm_text = f"{bpm:.0f} BPM" if bpm is not None else "-- BPM"
        cv2.putText(frame, bpm_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"signal quality: {quality:.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if status:
            cv2.putText(frame, status, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "filtered pulse signal (0.7-4 Hz)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Vitals from Video - Phase 1  (q = quit, s = save CSV)", np.vstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and len(times) > 1:
            name = time.strftime("vitals_raw_%Y%m%d_%H%M%S.csv")
            with open(name, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["time_s", "mean_green"])
                wr.writerows(zip(times, values))
            print("saved", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
