"""The fitted donor -> recipient map, and warping donor linework through it.

1. Anchors are parcel centroids. `fit_transport` fits entropic unbalanced GW between the two
   anchor clouds' max-normalized distance matrices -- scale-free shape matching, the normalization
   `signature` also uses -- and its barycentric projection gives every donor anchor one
   recipient-frame target.
2. `transport_lines` moves every vertex of every donor polyline by inverse-square-distance
   (Shepard) interpolation through its nearest anchors' targets, so a polyline warps coherently
   instead of snapping to individual anchors. Vertex count and order are preserved.

What arrives is still the donor's geometry in the recipient's frame, wherever the map sends it --
buildings included. `snap` puts it on the recipient's own substrate.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from reblock.contracts import Block
from reblock.transplant.gw import (
    Arr,
    GWParams,
    barycentric_projection,
    entropic_gw_unbalanced,
    gw_cost,
)

# Squared metres added to every vertex-anchor distance before inverting it, so a vertex sitting
# exactly on an anchor takes that anchor's target instead of dividing by zero.
_IDW_EPS = 1e-6


@dataclass(frozen=True)
class TransportParams:
    gw: GWParams
    idw_k: int      # anchors each transported vertex is interpolated through


@dataclass(frozen=True, eq=False)
class Transport:
    """A fitted donor -> recipient map. Compared by identity: it carries arrays."""

    pi: Arr                 # (n_donor, n_recipient) coupling
    bary: Arr               # (n_donor, 2) each donor anchor's recipient-frame target
    donor_xy: Arr           # (n_donor, 2) donor anchors, donor CRS
    recipient_xy: Arr       # (n_recipient, 2) recipient anchors, recipient CRS
    gw_dist: float          # the exact GW objective of `pi`: the real shape distance, not a proxy
    idw_k: int


def parcel_xy(block: Block) -> Arr:
    """The block's parcel centroids, in parcel order -- the anchors every GW fit matches."""
    c = block.parcels.geometry.centroid
    return np.column_stack([c.x.to_numpy(), c.y.to_numpy()])


def normalized_dist_matrix(xy: Arr) -> Arr:
    """Pairwise Euclidean distances over their own maximum: a scale-free shape descriptor."""
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    dmax = d.max()
    if dmax <= 0:
        raise ValueError("degenerate (coincident) point cloud")
    return d / dmax


def fit_transport(donor_xy: Arr, recipient_xy: Arr, params: TransportParams) -> Transport:
    """Fit the anchor correspondence under uniform target marginals (each cloud carries mass 1).
    `params.gw.tau` sets how strongly the unbalanced solve holds those marginals: finite tau lets
    the coupling concentrate on well-matched anchors instead of spreading evenly."""
    n, m = len(donor_xy), len(recipient_xy)
    c1 = normalized_dist_matrix(donor_xy)
    c2 = normalized_dist_matrix(recipient_xy)
    p = np.full(n, 1.0 / n, dtype=np.float64)
    q = np.full(m, 1.0 / m, dtype=np.float64)
    pi = entropic_gw_unbalanced(c1, c2, p, q, params.gw)
    return Transport(pi=pi, bary=barycentric_projection(pi, recipient_xy), donor_xy=donor_xy,
                     recipient_xy=recipient_xy, gw_dist=gw_cost(pi, c1, c2),
                     idw_k=params.idw_k)


def line_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """A multipart geometry's parts, or the geometry itself."""
    return list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]


def _idw(query_xy: Arr, tree: cKDTree, anchor_targets: Arr, k: int) -> Arr:
    dist, idx = tree.query(query_xy, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / (dist ** 2 + _IDW_EPS)
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("qk,qkd->qd", w, anchor_targets[idx])


def transport_lines(lines: gpd.GeoDataFrame, transport: Transport, *,
                    crs: CRS) -> gpd.GeoDataFrame:
    """Every LineString part of `lines` (donor CRS), each vertex warped into the recipient frame
    (`crs`). A part with fewer than two vertices is dropped; nothing else is."""
    k = min(transport.idw_k, len(transport.donor_xy))
    tree = cKDTree(transport.donor_xy)
    out: list[LineString] = []
    for geom in lines.geometry:
        for part in line_parts(geom):
            coords = np.asarray(part.coords, dtype=np.float64)
            if len(coords) < 2:
                continue
            out.append(LineString(_idw(coords, tree, transport.bary, k)))
    return gpd.GeoDataFrame(geometry=out, crs=crs)
