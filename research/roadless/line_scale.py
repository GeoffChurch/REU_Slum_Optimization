"""What length scale makes a block's lines a network? From geometry alone (before any clearing),
per open cell, its best line: L = max over 48 headings of the soft line length F + B (kappa 2).
Candidate r0 for the S-curve g = L^n / (L^n + r0^n):
  maxent   the log-logistic fit (histogram equalization): r0 = median L, n = pi / (sqrt 3 sd log L)
  otsu     the split of log L maximizing between-class variance (meaningful only if bimodal)
  perc     the percolation threshold: the length t where the street-connected share of the cells
           with L >= t drops fastest (the scale at which a long-line network stops existing)
Writes line_scale_<id>.png (histogram of log L with the three, and the connected share vs t).

    PYTHONPATH=. pixi run python research/roadless/line_scale.py <ids,>
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

FIELD = lifted.SoftSightline(beta=1.0, kappa=2.0)


def best_line(g: lifted.Grid) -> np.ndarray:
    L = np.zeros(g.ff0.shape)
    for _i, R in FIELD.runs(g.ff0, g.h):
        L = np.maximum(L, R)
    return L


def otsu(x: np.ndarray, bins: int = 200) -> tuple[float, float]:
    """(threshold, between-class share of variance)."""
    hist, edges = np.histogram(x, bins=bins)
    mid = 0.5 * (edges[1:] + edges[:-1])
    w = hist / hist.sum()
    w0 = np.cumsum(w)
    m = np.cumsum(w * mid)
    mt = m[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    i = int(np.nanargmax(between))
    return float(mid[i]), float(between[i] / x.var())


def percolation(g: lifted.Grid, L: np.ndarray, open_: np.ndarray,
                ts: np.ndarray) -> np.ndarray:
    """Per threshold t: the share of the cells with L >= t that are 8-connected (within that
    set) to a street cell."""
    out = []
    for t in ts:
        C = open_ & (L >= t)
        lab, _n = ndimage.label(C, structure=np.ones((3, 3)))
        hit = np.unique(lab[C & g.ground])
        hit = hit[hit > 0]
        out.append(np.isin(lab, hit).sum() / max(C.sum(), 1))
    return np.array(out)


def main(ids: list[str]) -> None:
    for b in common.build_blocks(ids):
        polys = np.asarray(b.buildings.outlines)
        g = lifted.Grid.of(b.boundary, polys, list(b.streets.geometry), 0.5)
        open_ = g.ff0 >= 0.5
        L = best_line(g)
        x = np.log(L[open_ & (L > 0.5)])
        r_me = float(np.exp(np.median(x)))
        n_me = float(np.pi / (np.sqrt(3) * x.std()))
        t_ot, sep = otsu(x)
        ts = np.exp(np.linspace(np.log(2), np.log(np.exp(x).max()), 60))
        share = percolation(g, L, open_, ts)
        drop = -np.diff(share) / np.diff(np.log(ts))
        r_pc = float(np.sqrt(ts[1:] * ts[:-1])[int(np.argmax(drop))])
        print(f"{b.block_id} n={len(polys)}: maxent r0 {r_me:5.1f} m n {n_me:.2f} | otsu "
              f"{np.exp(t_ot):5.1f} m (between-class {sep:.2f} of var) | percolation "
              f"{r_pc:5.1f} m (connected share {share[0]:.2f} -> {share[-1]:.2f})", flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].hist(np.exp(x), bins=np.exp(np.linspace(x.min(), x.max(), 80)), color="0.6")
        ax[0].set_xscale("log")
        for r, c, lab in ((r_me, "C0", f"max-entropy r0 {r_me:.0f} m (n {n_me:.1f})"),
                          (np.exp(t_ot), "C1", f"Otsu {np.exp(t_ot):.0f} m"),
                          (r_pc, "C3", f"percolation {r_pc:.0f} m")):
            ax[0].axvline(r, color=c, label=lab)
        ax2 = ax[0].twinx()
        rr = np.exp(np.linspace(x.min(), x.max(), 200))
        ax2.plot(rr, rr ** n_me / (rr ** n_me + r_me ** n_me), "C0--")
        ax2.set_ylabel("max-entropy g")
        ax[0].set_xlabel("best line through a cell (m)")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[1].plot(ts, share, "C3")
        ax[1].set_xscale("log")
        ax[1].axvline(r_pc, color="C3", ls=":")
        ax[1].set_xlabel("threshold t (m)")
        ax[1].set_ylabel("share of cells with L >= t connected to a street")
        fig.suptitle(b.block_id)
        fig.tight_layout()
        fig.savefig(HERE / f"line_scale_{b.block_id}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
