"""
Draws docs/model/training_curve.png from train_net.py's history file:

    python docs/model/plot_training_curve.py docs/model/training_history.json docs/model/training_curve.png ["Previous model"]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
h = json.load(open(sys.argv[1]))
ep = [r["epoch"] for r in h]
INK, SEC, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C = {"15": "#2a78d6", "8": "#eb6834", "hard_left": "#1baa7a", "hard_right": "#8a5cd1"}
# The flat comparison line: analyze_heart_rate's default method. When train_net.py runs with weights
# already installed that is the installed network ("previous model"), otherwise the hand-made pipeline.
BASE = sys.argv[3] if len(sys.argv) > 3 else "Hand-made (POS)"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
labels = []                    # (value, colour) at the right edge; spread apart below so none overlap
for w in ("15", "8"):
    net = [100 * r[f"v{w}_net_within5"] for r in h]
    pos = 100 * h[0][f"v{w}_pos_within5"]
    ax.plot(ep, net, color=C[w], lw=2, marker="o", ms=3.5, label=f"Network, front, {w} s")
    ax.hlines(pos, ep[0], ep[-1], color=C[w], lw=1.5, ls=(0, (4, 3)), label=f"{BASE}, front, {w} s")
    labels += [(net[-1], INK), (pos, SEC)]
for view, name in (("hard_left", "left camera"), ("hard_right", "right camera")):
    key = f"{view}_v15_net_within5"
    if key in h[0]:
        net = [100 * r[key] for r in h]
        pos = 100 * h[0][f"{view}_v15_pos_within5"]
        ax.plot(ep, net, color=C[view], lw=2, marker="o", ms=3.5, label=f"Network, {name}, 15 s")
        ax.hlines(pos, ep[0], ep[-1], color=C[view], lw=1.5, ls=(0, (4, 3)), label=f"{BASE}, {name}, 15 s")
        labels += [(net[-1], INK), (pos, SEC)]
placed = []
for v, c in sorted(labels):
    y = max(v, placed[-1] + 2.6) if placed else v
    placed.append(y)
    ax.annotate(f"{v:.0f}%", (ep[-1], v), xytext=(ep[-1] + 0.8, y), textcoords="data", va="center", color=c, fontsize=9)
ax.set_xlim(0.5, ep[-1] + 3)
lo = min(min(100 * r[k] for r in h) for k in h[0] if k.endswith("_net_within5") or k.endswith("_pos_within5"))
ax.set_ylim(10 * (lo // 10) - 2, 100)
ax.set_xlabel("Training epoch", color=SEC)
ax.set_ylabel("Within 5 BPM of finger sensor (%)", color=SEC)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED)
views = ", three camera angles" if "hard_left_v15_net_within5" in h[0] else ""
ax.set_title("Validation: 40 people the network never trained on" + views, loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=SEC)
fig.tight_layout()
fig.savefig(sys.argv[2], facecolor=SURF)
