"""PNG layers (1 m) for the 5810@major page from the painted rasters: raw flow, flow against the
no-buildings prior (log share ratio, and its Monroe z), change in flow, change in escape time.
No floor anywhere; display ranges are the before map's 99.5th percentiles."""
import sys

import numpy as np
from PIL import Image

arms = sys.argv[1:]
S0, S_open, op = np.load("S_before.npy"), np.load("S_open.npy"), np.load("open0.npy")
U0 = np.load("u_before.npy")
ins = ~np.isnan(S0)
FLOW = [(0, (20, 70, 110)), (0.5, (40, 170, 220)), (1, (210, 248, 255))]
GAIN = [(0, (255, 190, 80)), (1, (255, 64, 48))]
LOSS = [(0, (90, 140, 220)), (1, (160, 200, 255))]
FAST = [(0, (40, 120, 90)), (1, (100, 235, 165))]
SLOW = [(0, (200, 110, 40)), (1, (255, 175, 95))]


def ramp(t, stops):
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], axis=-1) / 255


def save(rgb, a, name):
    img = np.concatenate([rgb, np.clip(a, 0, 1)[..., None]], axis=-1)
    Image.fromarray((img * 255).astype(np.uint8), "RGBA").save(f"layers/{name}.png", optimize=True)


def dz(S):
    ok = ins & (S > 0) & (S_open > 0)
    d = np.full(S.shape, np.nan)
    z = np.full(S.shape, np.nan)
    d[ok] = np.log(S[ok] / S[ok].sum()) - np.log(S_open[ok] / S_open[ok].sum())
    z[ok] = d[ok] / np.sqrt(1 / S[ok] + 1 / S_open[ok])
    return d, z


def pos_layer(v, hi, name):
    t = np.clip(np.nan_to_num(v) / hi, 0, 1)
    save(ramp(t, FLOW), np.where(np.nan_to_num(v) > 0, 0.2 + 0.8 * t ** 0.8, 0), name)


LO, HI = np.nanquantile(S0[ins & (S0 > 0)], [0.5, 0.995])
d0, z0 = dz(S0)
HD, HZ = np.nanquantile(d0, 0.995), np.nanquantile(z0, 0.995)
G = float(np.nanquantile(S0[ins & (S0 > 0)], 0.99))       # change scale: the before map's 99th


def raw(S, name):
    t = np.clip((np.nan_to_num(S) - LO) / (HI - LO), 0, 1)
    save(ramp(t, FLOW), np.where(np.nan_to_num(S) > LO, 0.2 + 0.8 * t ** 0.8, 0), name)


raw(S0, "flux_before")
pos_layer(d0, HD, "rel_before")
pos_layer(z0, HZ, "z_before")
for k in arms:
    S1, U1 = np.load(f"S_{k}.npy"), np.load(f"u_{k}.npy")
    raw(S1, f"flux_{k}")
    d1, z1 = dz(S1)
    pos_layer(d1, HD, f"rel_{k}")
    pos_layer(z1, HZ, f"z_{k}")
    dd = np.nan_to_num(S1) - np.nan_to_num(S0)
    up, dn = np.clip(dd / G, 0, 1), np.clip(-dd / G, 0, 1)
    rgb = np.where((dd > 0)[..., None], ramp(up, GAIN), ramp(dn, LOSS))
    a = np.where(dd > 0, np.clip((dd / G - 0.06) / 0.94, 0, 1) ** 0.6,
                 np.clip((-dd / G - 0.12) / 0.88, 0, 1) ** 0.8 * 0.55)
    save(rgb, np.where(ins, a, 0), f"dflux_{k}")
    with np.errstate(divide="ignore", invalid="ignore"):
        ru = np.where((U0 > 0) & ~np.isnan(U1), (U1 - U0) / U0, np.nan)
    t = np.clip(np.nan_to_num(-ru) / 0.5, -1, 1)
    rgb = np.where((t > 0)[..., None], ramp(t.clip(0), FAST), ramp((-t).clip(0), SLOW))
    alpha = np.where(~np.isnan(ru), np.clip(np.abs(t) / 0.6, 0, 1) ** 0.7 * 0.85, 0)
    save(rgb, alpha, f"escape_{k}")
    print(k, "median escape-time change", float(np.nanmedian(ru)))
print("ranges: ratio", float(np.exp(HD)), "z", float(HZ))
