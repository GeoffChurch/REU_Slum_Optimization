"""A cheap, GW-consistent shape signature: shortlist donors before paying for a real GW fit.

The sorted eigenvalues of the max-normalized pairwise-distance matrix of a fixed-size subsample of
parcel centroids -- invariant to translation, rotation, reflection, permutation and scale, and
built from the same normalized distances the GW cost itself matches. Bootstrap-averaged over
several subsamples, because one random draw from a block well above `n_sub` points is noisy.

It is a PROXY. On the committed 500-pair matrix it correlates only moderately with the real GW
distance (r = 0.52), so it stratifies or shortlists and is never reported in place of `gw_dist`.
A properly n-invariant heat-trace signature was tried instead and was worse at predicting
transplant fidelity (docs/superpowers/notes/2026-07-23-ot-road-transplant.md §4): what matters
here is agreement with the GW cost, not n-invariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reblock.transplant.gw import Arr


@dataclass(frozen=True)
class SignatureParams:
    n_sub: int      # points per subsample: the signature's length, and the fewest a cloud may have
    n_boot: int     # subsamples averaged, drawn with seeds seed .. seed + n_boot - 1
    seed: int


def _one_signature(xy: Arr, n_sub: int, seed: int) -> Arr:
    rng = np.random.default_rng(seed)
    if len(xy) > n_sub:
        xy = xy[rng.choice(len(xy), size=n_sub, replace=False)]
    elif len(xy) < n_sub:
        raise ValueError(f"only {len(xy)} points, need >= {n_sub}")
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    dmax = d.max()
    if dmax <= 0:
        raise ValueError("degenerate (coincident) point cloud")
    return np.linalg.eigvalsh(d / dmax)[::-1]      # symmetric -> real; descending


def signature(xy: Arr, params: SignatureParams) -> Arr:
    """The (n_sub,) signature of a point cloud. Raises if it has fewer than `n_sub` points."""
    return np.mean([_one_signature(xy, params.n_sub, params.seed + k)
                    for k in range(params.n_boot)], axis=0)


def signature_distance(a: Arr, b: Arr) -> float:
    return float(np.linalg.norm(a - b))
