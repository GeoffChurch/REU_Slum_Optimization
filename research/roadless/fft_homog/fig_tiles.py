"""Figure: the macro adjoint map of where raising the effective conductivity lowers J2, and the
per-tile test of the two-scale prediction against the fine solve."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})
S = np.load("sens_map.npy")              # (NI-1, NJ-1, 3): lam^T M_xx U, lam^T M_yy U, lam^T M_xy U
J0 = float(np.load("tiles_base.npz")["J0f"])
d = np.load("fabric_5810.npz")
ins = d["inside"]
K, H = 4, 0.5
gain = (S[..., 0] + S[..., 1]) / J0  # -dJ/d(sigma_iso) per element (dJ = -S . dk), share of J0
NI, NJ = gain.shape
ri = np.clip((np.arange(NI) + 0.5) * K, 0, ins.shape[0] - 1).astype(int)
cj = np.clip((np.arange(NJ) + 0.5) * K, 0, ins.shape[1] - 1).astype(int)
gain = np.where(ins[np.ix_(ri, cj)], gain, np.nan)
t = pd.read_csv("tiles_clear.csv")
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
gpos =np.where(gain > 0, gain, np.nan) * 1e4
im = ax.imshow(gpos, origin="lower", cmap="Blues", interpolation="nearest",
               extent=(0, NJ * K * H, 0, NI * K * H),
               norm=LogNorm(vmin=np.nanpercentile(gpos, 5), vmax=np.nanpercentile(gpos, 99.5)))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
cb.set_label("-dJ2/d(sigma_iso) per 2 m element, x 1e-4 of J2 (log scale)")
g = t[t.rule == "R20"]
q = np.nanpercentile(-g.dJ_fine, 90)
for _, r in g.iterrows():
    if -r.dJ_fine >= q:
        ax.add_patch(Rectangle((r.c0 * H, r.r0 * H), 50, 50, fill=False, ec=ORANGE, lw=1.6))
ax.set_xlim(0, 800)
ax.set_ylim(0, 850)
ax.set_xlabel("m (east)")
ax.set_ylabel("m (north)")
ax.set_title("Macro adjoint map of 5810 (one macro solve + one adjoint)\n"
             "orange: the top 10% of 50 m tiles by fine-solve gain (rule R20)",
             loc="left", fontsize=9)
ax = axs[1]
for rule, col in (("R20", BLUE), ("STRIP", ORANGE)):
    g = t[(t.rule == rule) & (t.n_bldg > 0)]
    ax.scatter(-g.dJ_fine / J0 * 100, -g.dJ_twoscale / J0 * 100, s=14, color=col, alpha=0.7,
               edgecolors="white", linewidths=0.4, label=rule)
lim = [1e-3, 3]
ax.plot(lim, lim, color=MUTED, lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("fine solve: drop in J2 from the tile's clearing (% of J2)")
ax.set_ylabel("two-scale prediction (% of J2)")
ax.legend(title="clearing rule", fontsize=8)
ax.grid(True, color="#e6e5e1", lw=0.6)
ax.set_title("Per 50 m tile: two-scale vs fine", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig("fig_tiles.png", dpi=110)
