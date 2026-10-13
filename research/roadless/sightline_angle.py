"""The sightline metric's angle error (NOTES, "The sightline metric's angle error"): P of a
W x 40 m corridor rotated 0 -- 45 deg (checks.channel), max / min over the angles, under
variants of the sightline conductance that separate its two causes.

    PYTHONPATH=. uv run python research/roadless/sightline_angle.py [variant ...]

Variants: uni; ss (ss100k2n2r30, the 48 lattice directions); nmax12, nmax24 (more lattice
directions); k16 (16 walker headings); h025 (h 0.25); rays (96 uniformly spaced directions, each
scanned along the rows of its own rotated copy of the raster); turn (the turning edges boosted
too: lifted.TurnBoosted); both (rays + turn). Prototype: UniformRays has no gradient (vjp zero)
and runs on the CPU only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lifted  # noqa: E402
from checks import channel  # noqa: E402

ANGLES = (0, 2, 4, 6, 8, 10, 15, 20, 26.57, 30, 35, 40, 45)
WIDTHS = (1.0, 1.5, 2.0, 4.0)
NO_OFFS = np.zeros((0, 2), dtype=np.int64)


@dataclass(frozen=True)
class UniformRays:
    """SoftSightline's factor from M uniformly spaced directions: each direction's free paths
    are scanned along the rows of a copy of the open raster rotated to it (bilinear), mapped back
    to the cells (bilinear), and spread onto the K headings by the same hat weights."""
    beta: float
    kappa: float
    r0_m: float
    hill: float
    M: int
    name: str = "rays"

    def g(self, R):
        x = (R / self.r0_m) ** self.hill
        return x / (1.0 + x)

    def layers(self, grid, open_, K):
        o = grid.paint(open_, np)
        ny, nx = o.shape
        cy, cx = (ny - 1) / 2, (nx - 1) / 2
        n = int(np.ceil(np.hypot(ny, nx))) + 2            # covers the raster at every angle
        u = np.arange(n) - (n - 1) / 2
        across, along = np.meshgrid(u, u, indexing="ij")
        ang = np.arange(self.M) * np.pi / self.M
        W = lifted._angle_weights(K, ang)
        dr, dc = np.mgrid[0:ny, 0:nx].astype(float)
        dr, dc = dr - cy, dc - cx
        out = np.ones((K, *o.shape))
        for j, th in enumerate(ang):
            e = np.array([np.cos(th), np.sin(th)])
            nv = np.array([-e[1], e[0]])
            rot = ndimage.map_coordinates(
                o, [cy + along * e[1] + across * nv[1], cx + along * e[0] + across * nv[0]],
                order=1, cval=0.0)
            F, B = lifted._soft_fb(rot, 1, 0, NO_OFFS, grid.h, self.kappa)
            back = ndimage.map_coordinates(
                self.g(F + B), [dc * nv[0] + dr * nv[1] + (n - 1) / 2,
                                dc * e[0] + dr * e[1] + (n - 1) / 2], order=1, cval=0.0)
            gj = grid.mean(back, np)
            for k in np.flatnonzero(W[:, j]):
                out[k] += (self.beta * W[k, j]) * gj
        return out

    def turning(self, grid, open_, K):
        return [1.0] * K

    def vjp(self, grid, open_, K, A):
        return np.zeros(open_.shape)


def ss(nmax: int = 6) -> lifted.SoftSightline:
    return lifted.SoftSightline(100.0, 2.0, r0_m=30.0, hill=2.0, nmax=nmax)


RAYS = UniformRays(100.0, 2.0, 30.0, 2.0, 96)
# variant -> (along conductance, K, h)
VARIANTS = {
    "uni": (lifted.Uniform(), 8, 0.5),
    "ss": (ss(), 8, 0.5),
    "nmax12": (ss(12), 8, 0.5),
    "nmax24": (ss(24), 8, 0.5),
    "k16": (ss(), 16, 0.5),
    "h025": (ss(), 8, 0.25),
    "rays": (RAYS, 8, 0.5),
    "turn": (lifted.TurnBoosted(ss()), 8, 0.5),
    "both": (lifted.TurnBoosted(RAYS), 8, 0.5),
}


def main(names: list[str]) -> None:
    for name in names:
        along, K, h = VARIANTS[name]
        p = lifted.Params(ell_m=3.0, K=K, along=along, cells=lifted.OpenFraction(),
                          steps=lifted.SmallestCell())
        for W in WIDTHS:
            ps = np.array([channel(a, W, 40.0, h, p) for a in ANGLES])
            print(f"{name:7s} W {W:3.1f}  max/min {ps.max() / ps.min():5.2f}  P/P(0): "
                  + " ".join(f"{a:g}:{v:.2f}" for a, v in zip(ANGLES, ps / ps[0], strict=True)),
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
