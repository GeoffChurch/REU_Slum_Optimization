"""Style the raw fields into the page's RGBA layers (row 0 = north), for a dark map ground."""
import sys

import numpy as np
from PIL import Image

r = np.load("raw.npz")
ny, nx = r["shape"]
s0 = r["s0"]
on0 = s0 > 0
LO, HI = np.quantile(s0[on0], [0.5, 0.995])          # the uniform metric's before, every arm


def save(rgb, a, name):
    img = np.zeros((ny * nx, 4))
    img[:, :3] = rgb
    img[:, 3] = np.clip(a, 0, 1)
    Image.fromarray((img.reshape(ny, nx, 4)[::-1] * 255).astype(np.uint8), "RGBA") \
        .save(f"layers/{name}.png", optimize=True)


def ramp(t, stops):
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], axis=1) / 255


FLOW = [(0, (20, 70, 110)), (0.5, (40, 170, 220)), (1, (210, 248, 255))]
GAIN = [(0, (255, 190, 80)), (1, (255, 64, 48))]
LOSS = [(0, (90, 140, 220)), (1, (160, 200, 255))]
FAST = [(0, (40, 120, 90)), (1, (100, 235, 165))]
SLOW = [(0, (200, 110, 40)), (1, (255, 175, 95))]


def flux(s, name):
    t = np.clip((s - LO) / (HI - LO), 0, 1)
    save(ramp(t, FLOW), np.where(s > LO, 0.2 + 0.8 * t ** 0.8, 0), name)


def change(d, name, G=2.5):
    up, dn = np.clip(d / G, 0, 1), np.clip(-d / G, 0, 1)
    rgb = np.where((d > 0)[:, None], ramp(up, GAIN), ramp(dn, LOSS))
    a = np.where(d > 0, np.clip((d - 0.15) / (G - 0.15), 0, 1) ** 0.6,
                 np.clip((-d - 0.3) / (G - 0.3), 0, 1) ** 0.8 * 0.55)
    save(rgb, a, name)


def escape(ru, name):
    m = ~np.isnan(ru)
    t = np.clip(np.nan_to_num(-ru) / 0.5, -1, 1)
    rgb = np.where((t > 0)[:, None], ramp(t.clip(0), FAST), ramp((-t).clip(0), SLOW))
    save(rgb, np.where(m, np.clip(np.abs(t) / 0.6, 0, 1) ** 0.7 * 0.85, 0), name)


if "ss" not in sys.argv:
    flux(s0, "flux_before")
    for k in ("cheap", "default"):
        flux(r[f"s1_{k}"], f"flux_{k}")
        change(r[f"s1_{k}"] - s0, f"dflux_{k}")
        escape(r[f"ru_{k}"], f"escape_{k}")
else:
    q = np.load("raw_ss.npz")
    flux(q["s0"], "flux_before_ss")
    flux(q["s1_ss"], "flux_ss")
    change(q["s1_ss"] - q["s0"], "dflux_ss")
    escape(q["ru_ss"], "escape_ss")
print("styled", LO, HI)
