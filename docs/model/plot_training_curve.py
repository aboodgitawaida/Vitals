"""
Draws docs/model/training_curve.png from train_net.py's history file:

    python docs/model/plot_training_curve.py docs/model/training_history.json docs/model/training_curve.png
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
h = json.load(open(sys.argv[1]))
ep = [r["epoch"] for r in h]
INK, SEC, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C = {"15": "#2a78d6", "8": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
for w in ("15", "8"):
    net = [100 * r[f"v{w}_net_within5"] for r in h]
    pos = 100 * h[0][f"v{w}_pos_within5"]
    ax.plot(ep, net, color=C[w], lw=2, marker="o", ms=3.5, label=f"Network, {w} s windows")
    ax.hlines(pos, ep[0], ep[-1], color=C[w], lw=1.5, ls=(0, (4, 3)), label=f"Hand-made (POS), {w} s windows")
    ax.annotate(f"{net[-1]:.0f}%", (ep[-1], net[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.annotate(f"{pos:.0f}%", (ep[-1], pos), xytext=(6, 0), textcoords="offset points", va="center", color=SEC, fontsize=9)
ax.set_xlim(0.5, ep[-1] + 2.5)
ax.set_ylim(40, 82)
ax.set_xlabel("Training epoch", color=SEC)
ax.set_ylabel("Within 5 BPM of finger sensor (%)", color=SEC)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(AXIS)
ax.tick_params(colors=MUTED)
ax.set_title("Validation: 40 people the network never trained on", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=SEC)
fig.tight_layout()
fig.savefig(sys.argv[2], facecolor=SURF)
