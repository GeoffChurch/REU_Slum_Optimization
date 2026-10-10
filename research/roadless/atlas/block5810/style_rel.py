"""Flow against the no-buildings prior (the same homes' walk to the same edge, every building
gone), with no floor and no clamp, two ways (Monroe, Colaresi & Quinn 2008):
- delta: the log ratio of a cell's share of all walking to its share in the open field (Eq 16
  with the counts large against any prior, so the prior is dropped);
- z: delta over its standard error, sqrt(1/y + 1/y') (Eq 18/21), reading a cell's flow as the
  expected number of walkers crossing it (Doyle & Snell's walker reading of current). The count
  scale multiplies every z by one constant, so the picture does not depend on it.
Display range: 0 to each quantity's 99.5th percentile over the metric's before map, shared by
every map under that metric."""
import numpy as np
from PIL import Image

r = np.load("raw.npz")
q = np.load("raw_ss.npz")
pr = np.load("raw_prior.npz")
ny, nx = r["shape"]
FLOW = [(0, (20, 70, 110)), (0.5, (40, 170, 220)), (1, (210, 248, 255))]


def ramp(t, stops):
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=float)
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], axis=-1) / 255


def dz(y, yp):
    ok = (y > 0) & (yp > 0)
    d = np.full(y.shape, np.nan)
    z = np.full(y.shape, np.nan)
    d[ok] = np.log(y[ok] / y[ok].sum()) - np.log(yp[ok] / yp[ok].sum())
    z[ok] = d[ok] / np.sqrt(1 / y[ok] + 1 / yp[ok])
    return d, z


def save(v, hi, name):
    t = np.clip(np.nan_to_num(v) / hi, 0, 1)
    a = np.where(np.nan_to_num(v) > 0, 0.2 + 0.8 * t ** 0.8, 0)
    img = np.concatenate([ramp(t, FLOW), a[:, None]], axis=1).reshape(ny, nx, 4)[::-1]
    Image.fromarray((img * 255).astype(np.uint8), "RGBA").save(f"layers/{name}.png", optimize=True)


scales = {}
for metric, before, prior, afters in (
        ("uni", r["s0"], pr["open_uni"], {"cheap": r["s1_cheap"], "default": r["s1_default"]}),
        ("ss", q["s0"], pr["open_ss"], {"ss": q["s1_ss"]})):
    d0, z0 = dz(before, prior)
    hd, hz = np.nanquantile(d0, 0.995), np.nanquantile(z0, 0.995)
    scales[metric] = (hd, hz)
    sfx = "" if metric == "uni" else "_ss"
    save(d0, hd, f"rel_before{sfx}")
    save(z0, hz, f"z_before{sfx}")
    for k, s1 in afters.items():
        d1, z1 = dz(s1, prior)
        save(d1, hd, f"rel_{k}")
        save(z1, hz, f"z_{k}")
print({k: (round(float(np.exp(a)), 2), round(float(b), 3)) for k, (a, b) in scales.items()})
