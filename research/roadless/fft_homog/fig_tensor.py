"""Figure: the tensor field over 5810 (25 m tiles) and a zoom with principal axes vs lanes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
d = np.load("fabric_5810.npz")
ff, ins = d["ff0"].astype(float), d["inside"]
t = pd.read_csv("tiles.csv")
H = 0.5

fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
bg = np.where(ins, ff, np.nan)
ax.imshow(bg, origin="lower", cmap="Greys_r", vmin=-0.6, vmax=1.0, interpolation="nearest",
          extent=(0, ff.shape[1] * H, 0, ff.shape[0] * H))
g = t[t.L_m == 25].copy()
g["iso"] = (g.sxx + g.syy) / 2
cm = plt.get_cmap("Blues")
patches = [Rectangle((c * H, r * H), 25, 25) for r, c in zip(g.r0, g.c0, strict=True)]
pc = PatchCollection(patches, cmap=cm, alpha=0.75, edgecolor="none")
pc.set_array(g.iso.values)
pc.set_clim(0, 1)
ax.add_collection(pc)
cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.01)
cb.set_label("isotropic part (sxx + syy)/2 of the effective tensor, 25 m tiles")
segs = []
for _, r in g.iterrows():
    a = np.radians(r.angle_deg)
    ln = 10 * min((r.s_major / max(r.s_minor, 1e-3) - 1) / 0.5, 1.0)
    cx, cy = (r.c0 + 25) * H, (r.r0 + 25) * H
    segs.append([(cx - ln * np.cos(a), cy - ln * np.sin(a)),
                 (cx + ln * np.cos(a), cy + ln * np.sin(a))])
ax.add_collection(LineCollection(segs, colors=INK, linewidths=0.8))
ax.set_xlim(0, 800)
ax.set_ylim(0, 850)
ax.set_xlabel("m (east)")
ax.set_ylabel("m (north)")
ax.set_title("5810 main body: effective conductivity per 25 m tile\n"
             "(ticks: major axis, length grows with anisotropy, full at 1.5)",
             fontsize=9, color=INK, loc="left")
# zoom
ax = axs[1]
r0, c0, n = 640, 360, 200        # 100 m window, south-west dense fabric
sub = ff[r0:r0 + n, c0:c0 + n]
ax.imshow(sub, origin="lower", cmap="Greys_r", vmin=-0.6, vmax=1.0, interpolation="nearest",
          extent=(c0 * H, (c0 + n) * H, r0 * H, (r0 + n) * H))
z = t[(t.L_m == 12.5) & (t.r0 >= r0) & (t.r0 < r0 + n) & (t.c0 >= c0) & (t.c0 < c0 + n)]
s_sig, s_lane = [], []
for _, r in z.iterrows():
    cx, cy = (r.c0 + 12.5) * H, (r.r0 + 12.5) * H
    an = r.s_major / max(r.s_minor, 1e-3)
    if an < 1.25:
        continue
    for deg, out in ((r.angle_deg, s_sig), (r.lane_deg, s_lane)):
        a = np.radians(deg)
        out.append([(cx - 2.8 * np.cos(a), cy - 2.8 * np.sin(a)),
                    (cx + 2.8 * np.cos(a), cy + 2.8 * np.sin(a))])
ax.add_collection(LineCollection(s_lane, colors=ORANGE, linewidths=3.0,
                                 label="lane axis (structure tensor)"))
ax.add_collection(LineCollection(s_sig, colors=BLUE, linewidths=1.6, label="tensor major axis"))
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_xlabel("m (east)")
ax.set_title("100 m zoom: 12.5 m tiles with anisotropy >= 1.25\n"
             "(open space white, buildings black)",
             fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig("fig_tensor_field.png", dpi=110)
