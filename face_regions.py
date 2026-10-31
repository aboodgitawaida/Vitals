"""
face_regions.py -- where on the face we measure, which pixels we trust,
what the background is doing, and whether the lighting is good enough.

MediaPipe FaceMesh gives 468 numbered points on the face. We outline three
skin regions with them, split each region into a grid of small patches, and
drop pixels that clearly aren't usable skin (hair, eyebrows, shadow, glare).
"""

import math

import cv2
import numpy as np

from face_layout import (CLIP_LEVEL, DARK_FRACTION, FACE_EXCLUDE, FACE_OVAL, GRID,  # noqa: F401 (re-exported)
                         MIN_FACE_PIXELS, MIN_PATCH_PIXELS, PATCHES, REGIONS, TRACES, assess_lighting,
                         patch_region, snr_level)


def landmarks_to_array(face, w, h):
    """FaceMesh coordinates are 0..1; convert all 468 points to pixels, shape (468, 2)."""
    return np.array([[p.x * w, p.y * h] for p in face.landmark], dtype=np.float32)


def region_crop(landmarks, indices, shape):
    """Bounding box (x, y) of a region plus its mask inside that box (small crops = fast)."""
    pts = landmarks[indices].astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x, y = max(x, 0), max(y, 0)
    w, h = max(min(w, shape[1] - x), 1), max(min(h, shape[0] - y), 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts - [x, y]), 255)
    return x, y, mask


def skin_filter(crop, mask):
    """
    Keep only pixels inside `mask` that look like usable skin.

    Why not a classic fixed "skin colour" range (e.g. YCrCb thresholds)? Fixed
    ranges are tuned on particular skin tones and are a known source of bias
    against darker skin. Every rule here is RELATIVE to the person's own
    region, except "the sensor is saturated", which is unusable for everyone.
    Returns (kept mask, number of pixels rejected for glare).
    """
    ys, xs = np.nonzero(mask)
    kept = np.zeros_like(mask)
    if len(ys) == 0:
        return kept, 0
    px = crop[ys, xs].astype(np.float32)                   # BGR
    luminance = 0.114 * px[:, 0] + 0.587 * px[:, 1] + 0.299 * px[:, 2]
    clipped = px.max(axis=1) >= CLIP_LEVEL
    keep = (luminance > DARK_FRACTION * np.median(luminance)) & ~clipped
    kept[ys[keep], xs[keep]] = 255
    return kept, int(clipped.sum())


def extract_patch_colors(frame, landmarks):
    """
    Mean RGB of the usable skin in every patch.

    Returns:
      colors:  {patch name: RGB array or None}
      patches: {patch name: (x0, y0, x1, y1)} rectangles in frame coordinates
      regions: {region name: (x, y, region_mask, kept_mask)} for the overlay
      stats:   numbers for the lighting coach
    """
    colors, patches, regions = {}, {}, {}
    lum_by_region, total, clipped_total = {}, 0, 0
    for region_name, indices in REGIONS.items():
        x, y, region = region_crop(landmarks, indices, frame.shape)
        crop = frame[y:y + region.shape[0], x:x + region.shape[1]]
        kept, clipped = skin_filter(crop, region)
        regions[region_name] = (x, y, region, kept)
        total += cv2.countNonZero(region)
        clipped_total += clipped
        if cv2.countNonZero(kept) > 0:
            b, g, r, _ = cv2.mean(crop, mask=kept)
            lum_by_region[region_name] = 0.299 * r + 0.587 * g + 0.114 * b

        cols, rows = GRID[region_name]
        h, w = region.shape
        key = region_name.replace(" ", "_")
        for i in range(cols * rows):
            cx, cy = i % cols, i // cols
            x0, x1 = cx * w // cols, (cx + 1) * w // cols
            y0, y1 = cy * h // rows, (cy + 1) * h // rows
            cell_mask = kept[y0:y1, x0:x1]
            name = f"{key}_p{i}"
            patches[name] = (x + x0, y + y0, x + x1, y + y1)
            if cv2.countNonZero(cell_mask) >= MIN_PATCH_PIXELS:
                b, g, r, _ = cv2.mean(crop[y0:y1, x0:x1], mask=cell_mask)
                colors[name] = np.array([r, g, b])
            else:
                colors[name] = None

    stats = {"luminance": lum_by_region, "clipped_fraction": clipped_total / max(total, 1)}
    return colors, patches, regions, stats


def fill_convex(mask, hull, value):
    """
    Scanline fill of a convex polygon (integer vertices in order), boundary included.
    The same arithmetic as fillConvex in web/features.js, so the browser and this
    code measure exactly the same face pixels. (cv2.fillConvexPoly treats edge
    pixels slightly differently, ~1% of a face.)
    """
    h, w = mask.shape
    pts = [tuple(map(int, p)) for p in hull]
    ys = [p[1] for p in pts]
    for y in range(max(min(ys), 0), min(max(ys), h - 1) + 1):
        xl, xr = math.inf, -math.inf
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            if ay <= y <= by or by <= y <= ay:
                if ay == by:
                    xl, xr = min(xl, ax, bx), max(xr, ax, bx)
                else:
                    x = ax + (y - ay) * (bx - ax) / (by - ay)
                    xl, xr = min(xl, x), max(xr, x)
        if xl > xr:
            continue
        x0, x1 = max(math.floor(xl + 0.5), 0), min(math.floor(xr + 0.5), w - 1)
        if x0 <= x1:                                   # (a negative x1 must not wrap around)
            mask[y, x0:x1 + 1] = value


def face_color(frame, landmarks):
    """
    Mean RGB of the whole face (outline minus eyes with brows, and lips): the heart-rate signal.

    No brightness filter here, unlike the patches: pixels that cross a threshold
    from frame to frame add noise, and on a large region the few dark pixels
    (nostrils, a strand of hair) barely move the mean. Saturated pixels (glare)
    are still dropped. Returns RGB or None.
    """
    pts = landmarks[FACE_OVAL].astype(np.int32)
    x, y = pts.min(axis=0)
    w, h = pts.max(axis=0) - (x, y) + 1
    x, y = max(x, 0), max(y, 0)
    w, h = max(min(w, frame.shape[1] - x), 1), max(min(h, frame.shape[0] - y), 1)
    mask = np.zeros((h, w), np.uint8)
    fill_convex(mask, cv2.convexHull(pts - [x, y])[:, 0], 1)
    for indices in FACE_EXCLUDE:
        fill_convex(mask, cv2.convexHull(landmarks[indices].astype(np.int32) - [x, y])[:, 0], 0)
    crop = frame[y:y + h, x:x + w]
    mask[crop.max(axis=2) >= CLIP_LEVEL] = 0
    if cv2.countNonZero(mask) < MIN_FACE_PIXELS:
        return None
    b, g, r, _ = cv2.mean(crop, mask=mask)
    return np.array([r, g, b])


def background_color(frame, landmarks):
    """
    Mean RGB of the background (two boxes in the top corners of the frame,
    skipping any that overlap the head). This is the "no pulse here"
    reference used to cancel room-light changes. Returns (rgb or None, boxes).
    """
    h, w = frame.shape[:2]
    fx0, fy0 = landmarks.min(axis=0)
    fx1, fy1 = landmarks.max(axis=0)
    pad = 0.35 * (fx1 - fx0)                                  # keep clear of hair / shoulders
    head = (fx0 - pad, fy0 - 2 * pad, fx1 + pad, fy1 + pad)
    bw, bh = int(0.18 * w), int(0.22 * h)
    candidates = [(10, 40, 10 + bw, 40 + bh), (w - 10 - bw, 40, w - 10, 40 + bh)]
    boxes = [b for b in candidates
             if b[2] < head[0] or b[0] > head[2] or b[3] < head[1] or b[1] > head[3]]
    if not boxes:
        return None, []
    pixels = np.concatenate([frame[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in boxes])
    pixels = pixels[pixels.max(axis=1) < CLIP_LEVEL]
    if len(pixels) < 100:
        return None, boxes
    b, g, r = pixels.mean(axis=0)
    return np.array([r, g, b]), boxes


def draw_overlay(frame, regions, patches, patch_snr, background_boxes):
    """
    Patch SNR map: each patch's skin pixels tinted green / amber / red by how
    clean its pulse is. Rejected pixels (hair, shadow, glare) tinted purple.
    Background reference boxes outlined in white.
    """
    tint_colors = {"good": (90, 230, 90), "fair": (40, 190, 250), "none": (70, 70, 230)}
    out = frame.copy()
    for region_name, (x, y, region, kept) in regions.items():
        crop = out[y:y + region.shape[0], x:x + region.shape[1]]
        tint = crop.copy()
        tint[(region > 0) & (kept == 0)] = (200, 70, 170)
        key = region_name.replace(" ", "_")
        for name, (x0, y0, x1, y1) in patches.items():
            if not name.startswith(key + "_p"):
                continue
            sub = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
            color = tint_colors[snr_level(patch_snr.get(name))]
            tint[sub][kept[sub] > 0] = color
        crop[:] = cv2.addWeighted(tint, 0.35, crop, 0.65, 0)
    k = max(1.0, frame.shape[1] / 640)      # the frame is shrunk to 640 wide for display
    for x0, y0, x1, y1 in background_boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (230, 230, 230), int(round(k)))
        cv2.putText(out, "bg ref", (x0 + int(4 * k), y0 + int(14 * k)), cv2.FONT_HERSHEY_DUPLEX,
                    0.4 * k, (230, 230, 230), int(round(k)), cv2.LINE_AA)
    return out
