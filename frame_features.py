"""
frame_features.py -- everything measured from ONE video frame, given FaceMesh landmarks.

Shared by the live app (vitals_app.py) and the offline video tools
(evaluate_video.py), so a recorded video is processed exactly like the webcam.

Per frame:
  * motion check (how far the nose moved, relative to face width)
  * landmark smoothing (FaceMesh jitters a pixel or two per frame)
  * mean colour of the whole face (the heart-rate signal), of the 16 skin
    patches (the SNR map) and of the background wall
  * head pose for motion correction (x, y, scale, roll, mouth opening)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from face_layout import (FACE_LOCK_JUMP, FACE_LOCK_SEC, FACE_OVAL, FACE_TRACE, MOTION_LIMIT,  # noqa: F401
                         SMALL_MOTION, SMOOTHING)
from face_regions import background_color, extract_patch_colors, face_color, landmarks_to_array


def head_pose(lm, w, h):
    """
    Five numbers (vitals_stream.POSE_FIELDS) describing the head, for motion
    correction: nose x, nose y, face width (moving closer/further), roll
    (tilt), and how open the mouth is (inner-lip gap / face width: talking).
    All relative, so they don't depend on the resolution.
    """
    left_eye, right_eye = lm[33], lm[263]
    roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    face_w = np.linalg.norm(lm[234] - lm[454]) + 1e-9
    mouth = np.linalg.norm(lm[13] - lm[14]) / face_w          # 13/14 = inner upper/lower lip
    return np.array([lm[1][0] / w, lm[1][1] / h, face_w / w, roll, mouth])


def face_framing(lm, w, h):
    """
    How the face sits in the picture, for the framing coach (face_layout.assess_framing):
    share of the face outline outside the frame (MediaPipe places landmarks beyond the
    edge when part of the face is cut off), face width / frame width, and roll.
    """
    oval = lm[FACE_OVAL]
    outside = np.mean((oval[:, 0] < 0) | (oval[:, 0] >= w) | (oval[:, 1] < 0) | (oval[:, 1] >= h))
    left_eye, right_eye = lm[33], lm[263]
    return {"outside": float(outside), "size": float(np.linalg.norm(lm[234] - lm[454]) / w),
            "roll": float(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))}


def as_pixels(face, w, h):
    """A FaceMesh face (0..1 coordinates) or an (N, 2) pixel array -> (N, 2) pixels."""
    return face if isinstance(face, np.ndarray) else landmarks_to_array(face, w, h)


class FaceLock:
    """
    More than one person in view: measure one and keep following them.

    pick() gets every face this frame and returns the index of the one to measure, or
    None. The first time it takes the biggest, most central face. After that it takes
    the face nearest to where that person was; if none is within FACE_LOCK_JUMP face
    widths, that person is not in view (None) until FACE_LOCK_SEC passes and it picks
    again. request(x, y) (a tap or click) switches to the face at that point on the
    next frame. `switched` is True on the frame it moved to someone else; the caller
    then starts the measurement over. Mirrored in web/features.js (FaceLock).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.center = None          # where the measured face was (pixels)
        self.size = None            # its width (pixels)
        self.last_seen = None
        self.wanted = None          # a tap waiting for the next frame
        self.count = 0              # faces in view this frame
        self.switched = False

    def request(self, x, y):
        self.wanted = (float(x), float(y))

    @staticmethod
    def _boxes(faces):
        out = []
        for f in faces:
            x0, y0 = f.min(axis=0)
            x1, y1 = f.max(axis=0)
            out.append((float(x0), float(y0), float(x1), float(y1)))
        return out

    def pick(self, faces, now, w, h):
        """faces: list of (N, 2) pixel arrays; now: seconds; (w, h): frame size."""
        self.count, self.switched = len(faces), False
        if not faces:
            return None
        boxes = self._boxes(faces)
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        sizes = [max(b[2] - b[0], 1.0) for b in boxes]

        def nearest(p):
            return min(range(len(faces)), key=lambda i: math.hypot(centers[i][0] - p[0], centers[i][1] - p[1]))

        if self.wanted is not None:
            x, y = self.wanted
            self.wanted = None
            inside = [i for i, b in enumerate(boxes) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
            i = inside[0] if inside else nearest((x, y))
            self.switched = self.center is not None and math.hypot(
                centers[i][0] - self.center[0], centers[i][1] - self.center[1]) > FACE_LOCK_JUMP * self.size
        elif self.center is not None and now - self.last_seen <= FACE_LOCK_SEC:
            i = nearest(self.center)
            if math.hypot(centers[i][0] - self.center[0], centers[i][1] - self.center[1]) > FACE_LOCK_JUMP * self.size:
                return None                         # the measured person is not in view
        else:
            half_diag = math.hypot(w, h) / 2
            i = max(range(len(faces)), key=lambda k: sizes[k] * (1 - 0.5 * math.hypot(
                centers[k][0] - w / 2, centers[k][1] - h / 2) / half_diag))
            self.switched = self.center is not None
        self.center, self.size, self.last_seen = centers[i], sizes[i], now
        return i


@dataclass
class FrameFeatures:
    face: bool                        # did FaceMesh find a face?
    colors: dict = None               # {patch or FACE_TRACE: RGB or None}
    bg_rgb: np.ndarray = None         # wall colour, or None
    pose: np.ndarray = None           # head pose (POSE_FIELDS)
    movement: float = 0.0             # nose movement this frame / face width
    fast_motion: bool = False         # movement > MOTION_LIMIT: store the frame as a gap
    small_motion: bool = False        # movement > SMALL_MOTION
    overlay: dict = field(default_factory=dict)   # patches / regions / boxes / stats for drawing


class FeatureExtractor:
    """Keeps the little bit of state needed between frames (smoothed landmarks, last nose)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.smooth_lm = None
        self.prev_nose = None

    def process(self, frame_bgr, face_landmarks, others=()):
        """
        frame_bgr: full-resolution frame; face_landmarks: the face to measure (a FaceMesh
        face or (N, 2) pixels), or None; others: anyone else in view (kept out of the wall boxes).
        """
        if face_landmarks is None:
            self.prev_nose = None
            return FrameFeatures(face=False)
        h, w = frame_bgr.shape[:2]
        lm = as_pixels(face_landmarks, w, h)

        # Motion check: head movement corrupts the signal far more than a pulse changes it.
        face_width = np.linalg.norm(lm[234] - lm[454])
        movement = 0.0 if self.prev_nose is None else float(np.linalg.norm(lm[1] - self.prev_nose) / face_width)
        self.prev_nose = lm[1].copy()
        fast = movement > MOTION_LIMIT

        # Landmark smoothing: average with the previous frame unless the head really moved.
        if self.smooth_lm is None or fast:
            self.smooth_lm = lm
        else:
            self.smooth_lm = SMOOTHING * self.smooth_lm + (1 - SMOOTHING) * lm

        colors, patches, regions, stats = extract_patch_colors(frame_bgr, self.smooth_lm)
        stats["framing"] = face_framing(lm, w, h)
        colors[FACE_TRACE] = face_color(frame_bgr, self.smooth_lm)
        bg_rgb, bg_boxes = background_color(frame_bgr, self.smooth_lm, [as_pixels(o, w, h) for o in others])
        return FrameFeatures(
            face=True, colors=colors, bg_rgb=bg_rgb,
            pose=head_pose(lm, w, h),              # raw (unsmoothed) landmarks: the real head motion
            movement=movement, fast_motion=fast, small_motion=movement > SMALL_MOTION,
            overlay={"patches": patches, "regions": regions, "bg_boxes": bg_boxes, "stats": stats})
