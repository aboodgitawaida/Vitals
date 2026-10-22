"""
dashboard.py -- draws the live UI with plain OpenCV (no extra GUI library).

Layout:
  +----------------------+----------------+
  |                      |  HEART RATE    |
  |  camera + SNR map    |  lighting      |
  |                      |  regions       |
  |                      |  spectrum      |
  +----------------------+----------------+
  |  pulse waveform      |  60 s trend    |
  +----------------------+----------------+

Two themes: "dark" (default) and "light". Light mode turns the dashboard
white and goes full screen so the laptop screen acts as a soft front light.
"""

import cv2
import numpy as np

from vitals_dsp import HR_LOW, HR_HIGH

FONT = cv2.FONT_HERSHEY_DUPLEX
SIDE_W, BOTTOM_H, MIN_VIDEO_H = 380, 190, 480
SCREEN_ASPECT = 1.54      # MacBook screens are ~1.54:1; light mode pads to this so there are no black bars

# Colours are BGR (OpenCV's order)
THEMES = {
    "dark": {
        "bg": (22, 20, 18), "panel": (38, 34, 30), "grid": (64, 58, 54),
        "text": (240, 240, 240), "muted": (160, 150, 145), "wave": (255, 200, 110),
        "good": (120, 220, 110), "fair": (60, 195, 250), "none": (110, 110, 235),
    },
    "light": {
        "bg": (255, 255, 255), "panel": (243, 243, 243), "grid": (215, 215, 215),
        "text": (30, 30, 30), "muted": (110, 110, 110), "wave": (190, 110, 20),
        "good": (40, 150, 40), "fair": (0, 130, 225), "none": (40, 40, 200),
    },
}
LEVEL_TEXT = {"good": "GOOD SIGNAL", "fair": "FAIR SIGNAL", "none": "NO READING"}
C = THEMES["dark"]        # current theme, swapped by compose()


def put(img, text, org, scale=0.5, color=None, thick=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color or C["text"], thick, cv2.LINE_AA)


def panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), C["panel"], -1)
    if title:
        put(img, title, (x + 12, y + 22), 0.45, C["muted"])


def line_plot(img, x, y, w, h, values, color, thickness=2):
    """Normalised line plot of `values` inside the box (x, y, w, h)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return
    v = np.clip((v - v.mean()) / (v.std() + 1e-9), -3, 3)
    xs = np.linspace(x, x + w - 1, v.size)
    ys = y + h / 2 - v * (h / 6.5)
    cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_heart_rate(img, x, y, info):
    color = C[info["level"]]
    put(img, "HEART RATE", (x + 20, y + 32), 0.5, C["muted"])
    value = f"{info['bpm']:.0f}" if info["bpm"] is not None else "--"
    put(img, value, (x + 18, y + 105), 2.6, color, 4)
    (tw, _), _ = cv2.getTextSize(value, FONT, 2.6, 4)
    put(img, "BPM", (x + 30 + tw, y + 105), 0.8, color, 1)
    cv2.circle(img, (x + 26, y + 130), 6, color, -1, cv2.LINE_AA)
    label = "HOLDING (weak signal)" if info.get("state") == "holding" else LEVEL_TEXT[info["level"]]
    put(img, label, (x + 40, y + 136), 0.5, color)
    snr = f"SNR {info['snr_db']:+.1f} dB" if info["snr_db"] is not None else "SNR --"
    put(img, f"{snr}   {info['method_label']}   {info['fps']:.0f} fps", (x + 20, y + 160), 0.42, C["muted"])
    level, message = info["lighting"]
    put(img, message, (x + 20, y + 182), 0.42, C[level])


def draw_region_bars(img, x, y, w, region_stats, features):
    """
    Per region: best patch SNR (bar) and the share of the final signal that
    came from this region (inverse-noise weights).
    """
    panel(img, x, y, w, 112, "REGIONS  SNR | weight")
    (fw, _), _ = cv2.getTextSize(features, FONT, 0.38, 1)
    put(img, features, (x + w - fw - 10, y + 22), 0.38, C["muted"])
    for i, (name, (snr, share)) in enumerate(region_stats.items()):
        row_y = y + 46 + i * 24
        put(img, name, (x + 12, row_y + 4), 0.42)
        bx, bw = x + 112, w - 212
        cv2.rectangle(img, (bx, row_y - 7), (bx + bw, row_y + 5), C["grid"], -1)
        if snr is None:
            put(img, "no skin", (bx + bw + 10, row_y + 4), 0.4, C["none"])
            continue
        frac = float(np.clip((snr + 6) / 15, 0, 1))
        level = "good" if snr >= 3 else "fair" if snr >= 0 else "none"
        cv2.rectangle(img, (bx, row_y - 7), (bx + int(bw * frac), row_y + 5), C[level], -1)
        put(img, f"{snr:+.1f}  {share:3.0%}", (bx + bw + 8, row_y + 4), 0.4)


def draw_spectrum(img, x, y, w, h, info):
    """Power at each frequency in the heart-rate band. The tallest bump is the heart rate."""
    panel(img, x, y, w, h, "PULSE SPECTRUM")
    px, py, pw, ph = x + 14, y + 34, w - 28, h - 62
    for bpm in (60, 90, 120, 150, 180, 210):
        gx = px + int((bpm / 60 - HR_LOW) / (HR_HIGH - HR_LOW) * pw)
        cv2.line(img, (gx, py), (gx, py + ph), C["grid"], 1)
        put(img, str(bpm), (gx - 11, py + ph + 17), 0.38, C["muted"])
    freqs, power = info.get("freqs"), info.get("power")
    if freqs is None:
        put(img, "waiting for data...", (px + 10, py + ph // 2), 0.45, C["muted"])
        return
    band = (freqs >= HR_LOW) & (freqs <= HR_HIGH)
    f, p = freqs[band], power[band]
    p = p / (p.max() + 1e-12)
    xs = px + (f - HR_LOW) / (HR_HIGH - HR_LOW) * pw
    ys = py + ph - p * ph
    pts = np.column_stack([xs, ys]).astype(np.int32)
    filled = np.vstack([[[int(xs[0]), py + ph]], pts, [[int(xs[-1]), py + ph]]])
    roi = img[py:py + ph + 1, px:px + pw + 1]          # blend only inside the plot area (fast)
    layer = roi.copy()
    cv2.fillPoly(layer, [filled - [px, py]], C["wave"])
    cv2.addWeighted(layer, 0.25, roi, 0.75, 0, dst=roi)
    cv2.polylines(img, [pts], False, C["wave"], 2, cv2.LINE_AA)
    if info.get("peak_bpm") is not None:
        color = C[info["level"]]
        gx = px + int((info["peak_bpm"] / 60 - HR_LOW) / (HR_HIGH - HR_LOW) * pw)
        cv2.line(img, (gx, py), (gx, py + ph), color, 1, cv2.LINE_AA)
        put(img, f"peak {info['peak_bpm']:.0f}", (min(gx + 6, px + pw - 80), py + 14), 0.42, color)


def draw_trend(img, x, y, w, h, trend, now):
    """Confident readings from the last 60 s; gaps where there was no reading."""
    panel(img, x, y, w, h, "HEART RATE TREND (60 s)")
    px, py, pw, ph = x + 44, y + 36, w - 58, h - 52
    points = [(t, b) for t, b in trend if now - t <= 60]
    valid = [b for _, b in points if b is not None]
    if not valid:
        put(img, "no readings yet", (px + 10, py + ph // 2), 0.45, C["muted"])
        return
    lo, hi = min(valid) - 8, max(valid) + 8
    for frac in (0, 0.5, 1):
        gy = int(py + ph - frac * ph)
        cv2.line(img, (px, gy), (px + pw, gy), C["grid"], 1)
        put(img, f"{lo + frac * (hi - lo):.0f}", (x + 8, gy + 4), 0.36, C["muted"])
    prev = None
    for t, b in points:
        if b is None:
            prev = None
            continue
        pt = (int(px + (1 - (now - t) / 60) * pw), int(py + ph - (b - lo) / (hi - lo) * ph))
        if prev is not None:
            cv2.line(img, prev, pt, C["good"], 2, cv2.LINE_AA)
        prev = pt


def compose(frame, info, theme="dark"):
    """Build the full dashboard image around the camera frame."""
    global C
    C = THEMES[theme]
    vh, vw = frame.shape[:2]
    top_h = max(vh, MIN_VIDEO_H)
    canvas = np.full((top_h + BOTTOM_H, vw + SIDE_W, 3), C["bg"], dtype=np.uint8)
    canvas[:vh, :vw] = frame

    # Overlays on the video
    if info["status"]:
        (tw, th), _ = cv2.getTextSize(info["status"], FONT, 0.55, 1)
        cv2.rectangle(canvas, (10, vh - th - 26), (30 + tw, vh - 10), (0, 0, 0), -1)
        put(canvas, info["status"], (20, vh - 18), 0.55, THEMES["dark"]["fair"])
    if info["recording"]:
        cv2.circle(canvas, (vw - 70, 28), 8, (60, 60, 235), -1, cv2.LINE_AA)
        put(canvas, "REC", (vw - 56, 34), 0.55, (60, 60, 235))
    hint = "q quit  v liveness  m method  b bg  l light  r map  p perf  s rec"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.42, 1)
    cv2.rectangle(canvas, (6, 6), (18 + hw, 32), (0, 0, 0), -1)
    put(canvas, hint, (12, 24), 0.42, (240, 240, 240))
    if info.get("perf"):
        (pw, _), _ = cv2.getTextSize(info["perf"], FONT, 0.36, 1)
        cv2.rectangle(canvas, (6, 36), (18 + pw, 56), (0, 0, 0), -1)
        put(canvas, info["perf"], (12, 51), 0.36, (170, 230, 170))

    if info.get("liveness"):
        draw_liveness_result(canvas, *info["liveness"], vw)

    # Side panel
    sx = vw
    draw_heart_rate(canvas, sx, 0, info)
    draw_region_bars(canvas, sx + 12, 196, SIDE_W - 24, info["region_stats"], info["features"])
    draw_spectrum(canvas, sx + 12, 316, SIDE_W - 24, top_h - 316 - 10, info)

    # Bottom row
    by = top_h + 10
    panel(canvas, 10, by, vw - 20, BOTTOM_H - 20,
          f"PULSE WAVEFORM ({info['method_used'].upper()}, 0.7-4 Hz)")
    if info.get("pulse") is not None:
        line_plot(canvas, 22, by + 30, vw - 44, BOTTOM_H - 60, info["pulse"], C["wave"])
    draw_trend(canvas, sx + 12, by, SIDE_W - 24, BOTTOM_H - 20, info["trend"], info["now"])

    if theme == "light":
        # Pad with white to the screen's shape: in full screen the whole display glows white.
        h, w = canvas.shape[:2]
        target_w = max(w + 160, int((h + 120) * SCREEN_ASPECT))
        target_h = int(target_w / SCREEN_ASPECT)
        top, left = (target_h - h) // 2, (target_w - w) // 2
        canvas = cv2.copyMakeBorder(canvas, top, target_h - h - top, left, target_w - w - left,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return canvas


# --------------------------------------------------------------------------
# Liveness check screens
# --------------------------------------------------------------------------

LIVENESS_SIZE = (1440, 900)       # drawn this size; full-screen mode scales it to the display
VERDICT_COLOURS = {"LIVE": (90, 200, 60), "NOT LIVE": (60, 60, 230), "NOT CONFIRMED": (40, 170, 240)}


def liveness_screen(frame, screen_rgb, text, progress):
    """
    The whole screen in the challenge colour (so it lights the face), a small
    camera preview to stay centred, one instruction line and a progress bar.
    screen_rgb is RGB; OpenCV draws BGR.
    """
    w, h = LIVENESS_SIZE
    bgr = tuple(int(c) for c in screen_rgb[::-1])
    canvas = np.full((h, w, 3), bgr, dtype=np.uint8)
    # Text colour that stays readable on any challenge colour
    luminance = 0.299 * screen_rgb[0] + 0.587 * screen_rgb[1] + 0.114 * screen_rgb[2]
    ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)

    pw = 360
    preview = cv2.resize(frame, (pw, int(frame.shape[0] * pw / frame.shape[1])), interpolation=cv2.INTER_AREA)
    px, py = (w - pw) // 2, 120
    canvas[py:py + preview.shape[0], px:px + pw] = preview
    cv2.rectangle(canvas, (px - 2, py - 2), (px + pw + 1, py + preview.shape[0] + 1), ink, 2)

    title = "LIVENESS CHECK"
    (tw, _), _ = cv2.getTextSize(title, FONT, 1.1, 2)
    cv2.putText(canvas, title, ((w - tw) // 2, 80), FONT, 1.1, ink, 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 1)
    ty = py + preview.shape[0] + 60
    cv2.putText(canvas, text, ((w - tw) // 2, ty), FONT, 0.7, ink, 1, cv2.LINE_AA)
    bx, bw = (w - 600) // 2, 600
    cv2.rectangle(canvas, (bx, ty + 30), (bx + bw, ty + 44), ink, 1)
    cv2.rectangle(canvas, (bx, ty + 30), (bx + int(bw * float(np.clip(progress, 0, 1))), ty + 44), ink, -1)
    note = "Colours only are analysed on this device; no images are saved.   v = cancel"
    (tw, _), _ = cv2.getTextSize(note, FONT, 0.5, 1)
    cv2.putText(canvas, note, ((w - tw) // 2, h - 40), FONT, 0.5, ink, 1, cv2.LINE_AA)
    return canvas


def draw_liveness_result(canvas, result, lines, vw):
    """Verdict panel over the bottom of the video area."""
    colour = VERDICT_COLOURS.get(result, (200, 200, 200))
    x0, y0, x1 = 10, 300, vw - 10
    y1 = y0 + 46 + 22 * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, dst=canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
    put(canvas, result, (x0 + 14, y0 + 34), 1.0, colour, 2)
    put(canvas, "liveness prototype - not a security product", (x0 + 260, y0 + 30), 0.4, (180, 180, 180))
    for i, line in enumerate(lines):
        put(canvas, line[:110], (x0 + 14, y0 + 62 + 22 * i), 0.42, (235, 235, 235))
