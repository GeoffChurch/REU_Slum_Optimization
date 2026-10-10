"""Figure: per-home escape metric, two-scale vs fine (5810), and relative error vs depth."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})
files = sys.argv[1:]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for path, col in zip(files, (BLUE, ORANGE), strict=False):
    z = np.load(path)
    u_f, u_t, hd, live = z["u_f"], z["u_t"], z["hd"], z["live"]
    ok = live & (u_f > 0) & (u_t > 0)
    lab = path.replace("homes_", "").replace(".npz", "").replace("_", " windows, ")
    axs[0].scatter(u_f[ok], u_t[ok], s=4, color=col, alpha=0.35, edgecolors="none", label=lab)
    rel = (u_t - u_f) / u_f
    bins = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 450])
    mids, med, lo, hi = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        m = live & (u_f > 0) & (hd >= a) & (hd < b)
        if m.sum() < 10:
            continue
        mids.append(0.5 * (a + b))
        med.append(np.median(rel[m]))
        lo.append(np.percentile(rel[m], 10))
        hi.append(np.percentile(rel[m], 90))
    axs[1].plot(mids, med, color=col, lw=2, marker="o", ms=5, label=lab + " (median)")
    axs[1].fill_between(mids, lo, hi, color=col, alpha=0.15, lw=0)
lim = [0.3, 1200]
axs[0].plot(lim, lim, color=MUTED, lw=1)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlim(lim)
axs[0].set_ylim(lim)
axs[0].set_xlabel("fine solve: home escape metric u_i")
axs[0].set_ylabel("two-scale u_i")
axs[0].legend(fontsize=8, markerscale=3, loc="upper left")
axs[0].set_title("Per home, block 5810 (scalar uni model)", loc="left", fontsize=9)
axs[1].axhline(0, color=MUTED, lw=1)
axs[1].set_xscale("symlog", linthresh=10)
axs[1].set_ylim(-0.6, 1.0)
axs[1].set_xlabel("home's distance to the exit (m)")
axs[1].set_ylabel("relative error of u_i (two-scale / fine - 1)")
axs[1].legend(fontsize=8, loc="upper right")
axs[1].set_title("Error by depth (band: 10th to 90th percentile)", loc="left", fontsize=9)
for ax in axs:
    ax.grid(True, color="#e6e5e1", lw=0.6)
fig.tight_layout()
fig.savefig("fig_homes.png", dpi=110)
