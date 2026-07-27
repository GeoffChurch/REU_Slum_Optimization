"""Geometric agreement between a predicted road network and a reference one.

Plain functions, deliberately NOT an Eval: `Eval.score(block, proposal)` has no slot for a
reference network, and forcing it into that Protocol would mean smuggling the reference in through
construction and lying about the signature.

Geometric only. The functional reading of "same network" (per-parcel egress agreement) is out of
scope -- permeability already measures function and is a primary scorer. These answer the question
permeability cannot: did the prediction put the paths WHERE the real ones are.
"""
from __future__ import annotations

import numpy as np
from geopandas import GeoDataFrame
from shapely.ops import unary_union

# Matches the permeability corridor, so "agrees" here means the same thing it means there.
DEFAULT_RADIUS_M = 3.0
DEFAULT_STEP_M = 2.0


def buffered_iou(
    proposal: GeoDataFrame, reference: GeoDataFrame, *, r: float = DEFAULT_RADIUS_M
) -> float:
    """Intersection-over-union of the two networks buffered by `r` metres.

    Note the implied scale: buffers stop overlapping once the offset exceeds `2r`, so this reads 0
    for anything more than 6 m apart at the default radius. That is a real property, not a bug --
    but it means a single large-offset test case proves nothing, and callers comparing networks
    that may be far apart should sweep `r`.
    """
    if proposal.empty or reference.empty:
        return 0.0
    a = unary_union(list(proposal.geometry)).buffer(r)
    b = unary_union(list(reference.geometry)).buffer(r)
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union > 0 else 0.0


def _sample(net: GeoDataFrame, step: float) -> np.ndarray:
    """Points along every line at `step` spacing, including both endpoints."""
    points: list[tuple[float, float]] = []
    for geom in net.geometry:
        if geom.is_empty or geom.length == 0:
            continue
        n = max(int(geom.length // step), 1)
        for i in range(n + 1):
            p = geom.interpolate(min(i * step, geom.length))
            points.append((p.x, p.y))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def directional_chamfer(
    proposal: GeoDataFrame, reference: GeoDataFrame, *, step: float = DEFAULT_STEP_M
) -> tuple[float, float]:
    """`(precision_m, recall_m)` — mean nearest-neighbour distance proposal→reference, then
    reference→proposal.

    Reported directionally and never averaged: precision is "paths drawn that aren't there",
    recall is "real paths missed", and a blended score hides which way a prediction fails, which
    is the only thing this measures. `step` imposes a quantization floor of roughly `step / 2`.
    """
    p = _sample(proposal, step)
    q = _sample(reference, step)
    if len(p) == 0 or len(q) == 0:
        return (float("inf"), float("inf"))
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    return (float(d.min(axis=1).mean()), float(d.min(axis=0).mean()))
