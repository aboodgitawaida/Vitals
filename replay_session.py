"""
replay_session.py -- replay a session saved with "Save session data" in the web app, exactly.

The file holds what the page measured every frame (average skin and wall colours,
head pose, motion flags; no images) and what the screen showed. This re-runs the
frames through live_session.LiveSession in the same batches as the browser's
worker, so you can see why it did what it did:

    python replay_session.py vitals_session_20261102_080500.json.gz
    python replay_session.py session.json.gz --ref 72        # your real heart rate (watch)

Prints:
  1. A summary: face found, motion, framing, every start-over with its reason,
     time to the first number, share of time a number was shown, and whether the
     replay reproduces the browser's screen (it should: same code, same batches).
  2. A timeline, one line per second.
  3. Every 15 s window: the network and the hand-made pipeline on the same data,
     and their error against --ref (or the real heart rate typed in before saving).
It also saves a plot next to the file (needs matplotlib).
"""

import argparse
import gzip
import json
import os
from collections import Counter

import numpy as np

from face_layout import TRACES
from live_session import LiveSession
from vitals_dsp import analyze_heart_rate

WINDOW_SEC, WINDOW_STEP = 15.0, 5.0


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":                     # gzip
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    if doc.get("format") != "vitals-session":
        raise ValueError(f"{path} is not a Vitals session file")
    return doc


def replay(doc):
    """
    Frames -> LiveSession, batch by batch, with the controls (method, background,
    reset) applied where the browser applied them. Returns (per-frame records,
    per-batch screens, start-overs).
    """
    session = LiveSession()
    events = {}
    for ev in doc.get("events", []):
        events.setdefault(ev["before_batch"], []).append(ev)
    frames, records, screens, resets = doc["frames"], [], [], []
    pos = 0
    for b, n in enumerate(doc["batches"]):
        for ev in events.get(b, []):
            if ev["action"] == "method":
                session.set_method(ev["value"])
            elif ev["action"] == "background":
                session.set_background(ev["value"])
            elif ev["action"] == "reset":
                session.reset()
                resets.append((ev.get("t") or 0.0, ev.get("why") or "page reset"))
        batch = frames[pos:pos + n]
        pos += n
        for k, f in enumerate(batch):
            before = len(session.resets)
            session.frame(f["t"], f["face"], f.get("colors"), f.get("bg"), f.get("pose"), f.get("fast", False),
                          f.get("small", False), f.get("stats"), f.get("fps", 0.0), analyze=k == n - 1)
            if len(session.resets) > before:
                resets.append(session.resets[-1])
            a = session.analysis
            records.append({"t": f["t"], "face": bool(f["face"]), "fast": bool(f.get("fast")),
                            "framing": session.framing_problem(), "collected": session.buffer.span(),
                            "status": session.status, "bpm": session.reading.bpm, "settling":
                            bool(session.reading.bpm is not None and session.tracker.settling(f["t"])),
                            "confidence": None if a is None else a.get("confidence")})
        if batch:
            info = session.info()
            screens.append({"batch": b, "t": batch[-1]["t"], "bpm": info["bpm"], "state": info["state"]})
    return records, screens, sorted(resets)


def screen_match(screens, logged):
    """Share of updates where the replay shows the same number and state as the browser did."""
    by_batch = {s["batch"]: s for s in logged if "batch" in s}
    same = total = 0
    for s in screens:
        g = by_batch.get(s["batch"])
        if g is None:
            continue
        total += 1
        same_bpm = (s["bpm"] is None) == (g["bpm"] is None) and (s["bpm"] is None or abs(s["bpm"] - g["bpm"]) < 0.05)
        same += same_bpm and s["state"] == g["state"]
    return (same / total if total else None), total


def arrays(doc):
    """Frames -> times, {trace: (N, 3)}, wall (N, 3), usable mask (face found, not moving fast)."""
    frames = doc["frames"]
    t = np.array([f["t"] for f in frames])
    usable = np.array([bool(f["face"]) and not f.get("fast") for f in frames])

    def rgb(get):
        out = np.full((len(frames), 3), np.nan)
        for i, f in enumerate(frames):
            v = get(f)
            if v is not None:
                out[i] = v
        return out
    traces = {n: rgb(lambda f, n=n: (f.get("colors") or {}).get(n)) for n in TRACES}
    return t, traces, rgb(lambda f: f.get("bg")), usable


def windows(doc, ref):
    """Every 15 s window: network and hand-made heart rate on the usable frames."""
    t, traces, bg, usable = arrays(doc)
    rows = []
    for end in np.arange(t[0] + WINDOW_SEC, t[-1] + 1e-9, WINDOW_STEP):
        sel = (t > end - WINDOW_SEC) & (t <= end) & usable
        if sel.sum() < 0.5 * WINDOW_SEC * 15:        # under half the window at 15+ fps
            rows.append({"end": end, "frames": int(sel.sum())})
            continue
        row = {"end": end, "frames": int(sel.sum())}
        for label, method in (("network", "net"), ("hand-made", "classic")):
            r = analyze_heart_rate(t[sel], {n: v[sel] for n, v in traces.items()}, bg[sel], method=method,
                                   patch_map=False)
            row[label] = None if r is None else r["bpm"]
            if label == "network" and r is not None:
                row["confidence"] = r.get("confidence")
        rows.append(row)
    return rows


def fmt(x, spec=".0f", none="--"):
    return none if x is None else format(x, spec)


def report(doc, records, screens, resets, ref):
    t = np.array([r["t"] for r in records])
    dur = t[-1] - t[0] if len(t) > 1 else 0.0
    fps = 1 / np.median(np.diff(t)) if len(t) > 2 else 0.0
    face = np.mean([r["face"] for r in records])
    fast = np.mean([r["fast"] for r in records if r["face"]] or [0])
    framing = Counter(r["framing"] for r in records if r["framing"])
    shown = [r for r in records if r["bpm"] is not None]
    first = shown[0]["t"] - t[0] if shown else None
    app = doc.get("app") or {}
    lines = [f"Session: {dur:.0f} s, {len(records)} frames ({fps:.1f} fps), saved {doc.get('saved', '?')}, "
             f"code {str(app.get('commit') or '?')[:7]}",
             f"Browser: {doc.get('user_agent', '?')}",
             f"Camera: {doc.get('camera')}, face tracking on {doc.get('delegate')}", ""]
    lines.append(f"Face found in {face:.0%} of frames; fast head motion in {fast:.0%} of those.")
    if framing:
        lines.append("Framing advice shown: " + "; ".join(f"{m} ({c / len(records):.0%} of frames)"
                                                         for m, c in framing.most_common()))
    lines.append(f"Started over {len(resets)} time(s)" + (":" if resets else "."))
    lines += [f"  at {rt - t[0]:5.1f} s: {why}" for rt, why in resets]
    lines.append(f"First number after {fmt(first, '.1f')} s; a number on screen {len(shown) / len(records):.0%} "
                 f"of the time ({np.mean([r['settling'] for r in shown] or [0]):.0%} of that still settling).")
    if shown:
        b = np.array([r["bpm"] for r in shown])
        lines.append(f"Numbers shown: median {np.median(b):.0f}, range {b.min():.0f}-{b.max():.0f} BPM"
                     + (f"; your real heart rate: {ref:.0f} BPM, median error {np.median(np.abs(b - ref)):.1f} BPM, "
                        f"{np.mean(np.abs(b - ref) <= 5):.0%} of the time within 5 BPM" if ref else ""))
    match, n = screen_match(screens, doc.get("screen", []))
    if match is not None:
        lines.append(f"Replay reproduces the browser's screen in {match:.0%} of {n} updates.")

    lines += ["", "Timeline (one line per second):", "",
              "    s | face | moving | collected | shown | sure | status"]
    by_sec = {}
    for r in records:
        by_sec.setdefault(int(r["t"] - t[0]), []).append(r)
    for sec, rs in sorted(by_sec.items()):
        last = rs[-1]
        lines.append(f"{sec:5d} | {np.mean([r['face'] for r in rs]):4.0%} | {np.mean([r['fast'] for r in rs]):6.0%} | "
                     f"{last['collected']:7.1f} s | {fmt(last['bpm']):>5}{'*' if last['settling'] else ' '}| "
                     f"{fmt(None if last['confidence'] is None else 100 * last['confidence'], '.0f'):>3}% | {last['status']}")
    lines.append("(* = settling)")
    return lines


def window_report(rows, ref, t0):
    lines = ["", f"Every {WINDOW_SEC:.0f} s window (usable frames only):", "",
             "  end s | frames | network (sure) | hand-made" + (" | error: network / hand-made" if ref else "")]
    for r in rows:
        if "network" not in r:
            lines.append(f"{r['end'] - t0:7.0f} | {r['frames']:6d} | too few usable frames")
            continue
        conf = r.get("confidence")
        sure = fmt(None if conf is None else 100 * conf, '.0f')
        line = f"{r['end'] - t0:7.0f} | {r['frames']:6d} | {fmt(r['network'], '.1f'):>6} ({sure}%) | {fmt(r['hand-made'], '.1f'):>6}"
        if ref:
            line += f" | {fmt(None if r['network'] is None else abs(r['network'] - ref), '.1f')} / " \
                    f"{fmt(None if r['hand-made'] is None else abs(r['hand-made'] - ref), '.1f')}"
        lines.append(line)
    return lines


def plot(path, records, rows, ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    t = np.array([r["t"] for r in records])
    t = t - t[0]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for mask, color, label in ((~np.array([r["face"] for r in records]), "#e5484d", "face lost"),
                               (np.array([r["fast"] for r in records]), "#eda100", "fast motion"),
                               (np.array([bool(r["framing"]) for r in records]), "#8e8e93", "framing advice")):
        if mask.any():
            ax.fill_between(t, 0, 1, where=mask, color=color, alpha=0.18, transform=ax.get_xaxis_transform(),
                            label=label, step="mid")
    bpm = np.array([np.nan if r["bpm"] is None else r["bpm"] for r in records])
    ax.plot(t, bpm, color="#2a78d6", lw=2, label="number on screen")
    t0 = records[0]["t"]
    ax.plot([r["end"] - t0 for r in rows if r.get("network")], [r["network"] for r in rows if r.get("network")],
            "o", color="#2a78d6", ms=4, label="network, 15 s windows")
    if ref:
        ax.axhline(ref, color="#0b0b0b", ls="--", lw=1, label=f"your real heart rate ({ref:.0f})")
    ax.set_ylabel("BPM")
    ax.legend(fontsize=8, loc="upper right")
    conf = np.array([np.nan if r["confidence"] is None else r["confidence"] for r in records])
    ax2.plot(t, 100 * conf, color="#2a78d6")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("sure (%)")
    ax2.set_xlabel("seconds")
    fig.tight_layout()
    out = os.path.splitext(path.removesuffix(".gz"))[0] + ".png"
    fig.savefig(out, dpi=120)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="vitals_session_*.json(.gz) saved by the web app")
    ap.add_argument("--ref", type=float, default=None, help="your real heart rate (BPM), e.g. from a watch")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    doc = load(args.session)
    ref = args.ref if args.ref is not None else doc.get("reference_bpm")
    records, screens, resets = replay(doc)
    if not records:
        print("The session has no frames.")
        return
    rows = windows(doc, ref)
    print("\n".join(report(doc, records, screens, resets, ref) + window_report(rows, ref, records[0]["t"])))
    if not args.no_plot:
        out = plot(args.session, records, rows, ref)
        if out:
            print(f"\nplot: {out}")


if __name__ == "__main__":
    main()
