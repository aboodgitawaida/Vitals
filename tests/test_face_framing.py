"""The framing coach (face_layout.assess_framing) and its measurement (frame_features.face_framing)."""

import numpy as np

from face_layout import FACE_OVAL, FACE_SIZE_MIN, assess_framing
from frame_features import face_framing


def face(cx, cy, width, w=640, h=480, tilt=0.0):
    """478 landmarks of a face-shaped ellipse (only the outline, cheeks and eyes matter here)."""
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 478)
    lm = np.column_stack([cx + 0.5 * width * np.cos(ang), cy + 0.7 * width * np.sin(ang)])
    oval = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    lm[FACE_OVAL] = np.column_stack([cx + 0.5 * width * np.cos(oval), cy + 0.7 * width * np.sin(oval)])
    lm[234], lm[454] = [cx - width / 2, cy], [cx + width / 2, cy]
    dy = np.tan(tilt) * width * 0.3
    lm[33], lm[263] = [cx - width * 0.15, cy - width * 0.1 - dy], [cx + width * 0.15, cy - width * 0.1 + dy]
    return lm


def test_well_framed_face_needs_no_advice():
    f = face(320, 240, 120)
    assert face_framing(f, 640, 480)["outside"] == 0.0
    assert assess_framing(face_framing(f, 640, 480)) is None


def test_face_cut_off_at_the_bottom():
    msg = assess_framing(face_framing(face(320, 440, 120), 640, 480))
    assert msg.startswith("Face partly out of view")


def test_far_away_and_tilted():
    small = face(320, 240, 640 * FACE_SIZE_MIN * 0.8)
    assert assess_framing(face_framing(small, 640, 480)).startswith("Face too small")
    tilted = face(320, 240, 120, tilt=0.6)
    assert assess_framing(face_framing(tilted, 640, 480)).startswith("Head tilted")


def test_no_framing_data_means_no_advice():
    assert assess_framing(None) is None
