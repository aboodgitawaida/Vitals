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

from face_layout import FACE_TRACE, MOTION_LIMIT, SMALL_MOTION, SMOOTHING  # noqa: F401 (re-exported)
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

    def process(self, frame_bgr, face_landmarks):
        """frame_bgr: full-resolution frame; face_landmarks: one FaceMesh face, or None."""
        if face_landmarks is None:
            self.prev_nose = None
            return FrameFeatures(face=False)
        h, w = frame_bgr.shape[:2]
        lm = landmarks_to_array(face_landmarks, w, h)

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
        colors[FACE_TRACE] = face_color(frame_bgr, self.smooth_lm)
        bg_rgb, bg_boxes = background_color(frame_bgr, self.smooth_lm)
        return FrameFeatures(
            face=True, colors=colors, bg_rgb=bg_rgb,
            pose=head_pose(lm, w, h),              # raw (unsmoothed) landmarks: the real head motion
            movement=movement, fast_motion=fast, small_motion=movement > SMALL_MOTION,
            overlay={"patches": patches, "regions": regions, "bg_boxes": bg_boxes, "stats": stats})
