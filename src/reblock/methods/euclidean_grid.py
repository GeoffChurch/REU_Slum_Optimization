"""EuclideanGridReblocker: overlay a fixed, evenly-spaced orthogonal grid of candidate roads
across a block (an NYC/Manhattan-style grid), as opposed to peel's adaptive parcel-by-parcel
descent. When `seek_density` is set, the grid is phase-shifted so a line runs through the
block's densest parcel cluster (found via a coarse raster over parcel representative points)
rather than being centered on an arbitrary bbox midpoint.

When `adaptive` is set, that same raster is kept as a local density *field* rather than being
collapsed to its peak: the base grid at `spacing` is laid down everywhere as the coarse layer,
and the densest raster cells additionally get an infill layer at `fine_spacing`, clipped to the
cell. The result is hierarchical (quadtree-like) -- coarse blocks where parcels are sparse,
subdivided into finer ones where they cluster -- rather than one uniform spacing everywhere.
`seek_density` and `adaptive` are independent: the former phases the base grid's origin, the
latter overlays extra mesh, and either, both, or neither may be set. Both default on, so the
out-of-the-box behaviour is this density-concentrated grid: finer where parcels cluster, coarser
where they are sparse, phased onto the densest cluster -- combined with the hugging trim below.

Independently of those, every candidate line is trimmed to hug the parcels: after being clipped
to the block's extent it is intersected with the parcel geometry buffered by `parcel_hug_buffer`,
so the portions running through empty gaps (with no parcels nearby) are dropped rather than drawn
edge-to-edge. This is always on -- it is not a toggle. `parcel_hug_buffer` (and `parcel_bridge_gap`
below) default to a multiple of the block's *own* median parcel nearest-neighbour distance, so the
hug is as tight on a dense 5 m-spaced block as on a sparse 50 m-spaced one, with no per-dataset
tuning; both remain explicitly override-able. To avoid needlessly fragmenting the road network, the
trim bridges empty gaps shorter than `parcel_bridge_gap` (keeping a line whole where it crosses a
small gap between two nearby parcel clusters) and only severs a line at larger gaps.

`follow_parcels` (default off) is a distinct mode that abandons the synthetic grid entirely: it
carves roads out of the parcel fabric itself. The candidate roads are the *shared boundary edges*
between adjacent parcels (the lines already visible between them, via `reblock.derive.adjacency`),
split to individual parcel-edge scale. Each edge is scored by the local parcel density (the same
raster used above) and selected with a probability that scales continuously with that density --
a dense mesh of roads where parcels cluster, a light scattering where they are sparse -- then the
selection is stitched to the `block.streets` frontage along real boundary edges so it forms a
connected network rather than floating slivers. All the other flags are ignored in this mode.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely import make_valid, snap, union_all
from shapely.affinity import rotate
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import substring

from reblock.contracts import Block, Proposal
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

# `follow_parcels` density raster cell size, as a multiple of the block's own parcel-spacing scale
# (median parcel nearest-neighbour distance): a few parcel-spacings wide, so a cell in a dense
# cluster holds many parcel points and one in a sparse area few -- a real density gradient rather
# than a flat one-point-per-cell field. Derived per block, never a hardcoded metre value.
_FOLLOW_DENSITY_RES_FACTOR = 4.0
# The smallest selected-edge cluster (total boundary length) worth keeping and wiring to the
# street, as a multiple of the parcel scale. A sparse density floor scatters lone edges all over
# the block; each would otherwise drag its own long connector back to the street, re-densifying
# the very mesh the sparse defaults were meant to thin. Clusters below this are dropped as noise,
# leaving the genuine (dense-area) sub-networks to be stitched -- so few connectors are needed.
_FOLLOW_MIN_COMPONENT_FACTOR = 2.0

# The parcel-hugging trim's two distances default to multiples of the block's own parcel-spacing
# scale (median nearest-neighbour distance between parcel points) rather than to a fixed value or
# a fraction of `spacing`, so a dense 5 m-apart block and a sparse 50 m-apart one each hug at
# their own scale. `_HUG_NN_FACTOR` sets how far past a parcel a road may reach; `_BRIDGE_NN_FACTOR`
# sets the largest empty gap a line will span rather than be severed at (see `_hug_line`).
_HUG_NN_FACTOR = 0.5
_BRIDGE_NN_FACTOR = 3.0


def _median_nn_distance(points: np.ndarray) -> float:
    """Median distance from each parcel point to its nearest *other* parcel point -- a robust
    per-block parcel-spacing scale. Returns 0.0 for fewer than two points (caller falls back)."""
    if len(points) < 2:
        return 0.0
    dists, _ = cKDTree(points).query(points, k=2)   # column 1 is the nearest neighbour (0 is self)
    return float(np.median(dists[:, 1]))


def _hug_line(line: LineString, hug_region: BaseGeometry, bridge_gap: float,
              min_len: float) -> list[LineString]:
    """Trim `line` to just the spans lying within `hug_region` (i.e. near a parcel), returning the
    surviving sub-segments. Holes shorter than `bridge_gap` between two near-parcel spans are kept
    (bridged) so a line crossing a small gap between two nearby parcel clusters is not needlessly
    severed; only larger empty gaps split it, and the dangling ends past the outermost near-parcel
    span are always dropped. `bridge_gap <= 0` disables bridging (every gap splits the line).
    Surviving spans shorter than `min_len` are dropped (the sliver rule, re-applied post-trim)."""
    parts = _line_parts(line.intersection(hug_region), min_len=0.0)
    if not parts:
        return []
    # Reduce to 1-D [start, end] intervals of arc length along `line` (its parts are straight,
    # so their endpoints project back onto it exactly), then merge intervals across small holes.
    intervals: list[tuple[float, float]] = []
    for part in parts:
        cs = list(part.coords)
        a, b = line.project(Point(cs[0])), line.project(Point(cs[-1]))
        intervals.append((min(a, b), max(a, b)))
    intervals.sort()
    merged: list[list[float]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= bridge_gap:   # small hole between near-parcel spans -> bridge
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[LineString] = []
    for a, b in merged:
        if b - a >= min_len and b - a > 0.0:
            seg = substring(line, a, b)   # a != b, so this is a LineString (never a Point)
            if isinstance(seg, LineString):
                out.append(seg)
    return out


def _line_parts(geom: BaseGeometry, min_len: float) -> list[LineString]:
    """Positive-length (>= min_len) LineString parts of `geom` (explode Multi*/GeometryCollection;
    drop non-lines and slivers)."""
    if geom is None or geom.is_empty:
        return []
    geoms: list[BaseGeometry] = (
        list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]
    )
    # geom_type (not just isinstance) so a LinearRing -- a LineString subclass -- stays excluded
    return [g for g in geoms
            if isinstance(g, LineString) and g.geom_type == "LineString" and g.length >= min_len]


def _shared_boundary(gi: BaseGeometry, gj: BaseGeometry, tol: float) -> BaseGeometry:
    """The shared boundary geometry between two adjacent parcels -- the same snap-then-intersect as
    `reblock.derive.adjacency._shared_len`, but keeping the line rather than only its length (with
    the same GEOS-side-location-conflict fallback onto validated operands)."""
    try:
        return snap(gi, gj, tol).intersection(gj)
    except GEOSException:
        return make_valid(gi).intersection(make_valid(gj))


def _straight_segments(line: LineString) -> list[LineString]:
    """Split `line` into its individual straight vertex-to-vertex segments. Keeps candidate roads at
    single parcel-edge scale rather than long multi-vertex runs, so nothing is ever concatenated
    into a long straight chain (a Voronoi edge is already one segment and passes through as-is)."""
    cs = list(line.coords)
    return [LineString([cs[k], cs[k + 1]]) for k in range(len(cs) - 1) if cs[k] != cs[k + 1]]


def _unit_hash(x: float, y: float) -> float:
    """A deterministic pseudo-random value in [0, 1) keyed on a point, so density-weighted edge
    selection is reproducible for a given block (the derivation cache assumes a pure `propose`)
    rather than depending on RNG state or iteration order."""
    digest = hashlib.md5(f"{x:.4f},{y:.4f}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def _connect_to_street(
    edges: list[LineString], selected: set[int], streets: gpd.GeoDataFrame, tol: float,
    min_component: float,
) -> tuple[set[int], int, int]:
    """Stitch the density-selected `edges` (indices into `edges`) into a sparse network that reaches
    the street frontage, returning (final selection, edges added, edges dropped as noise).

    Over the graph of *all* candidate boundary edges (nodes = shared, tol-snapped endpoints):
      1. Drop selected clusters whose total boundary length is below `min_component` -- lone edges
         scattered by the sparse density floor, not worth a road (nor the connector each would drag
         to the street). Only the genuine dense-area sub-networks survive.
      2. Wire the survivors to a *shared backbone* that starts at the street and grows: each cluster
         (largest first) takes the shortest path to whatever is already connected -- the street or
         an earlier cluster's connector -- so connectors are reused into a tree instead of every
         cluster drawing its own path to the frontage. A cluster with no path to the street at
         all is an isolated sliver and is dropped. In the spirit of peel.py's street-reaching
         guarantee."""
    street_geom = union_all(list(streets.geometry)) if len(streets) else None
    node_of: dict[tuple[int, int], int] = {}
    node_pt: dict[int, tuple[float, float]] = {}

    def nid(pt: tuple[float, float]) -> int:
        key = (round(pt[0] / tol), round(pt[1] / tol))   # snap coincident endpoints to one node
        if key not in node_of:
            node_of[key] = len(node_of)
            node_pt[node_of[key]] = pt
        return node_of[key]

    graph: nx.Graph = nx.Graph()
    edge_nodes: list[tuple[int, int]] = []
    for idx, seg in enumerate(edges):
        c0, c1 = seg.coords[0], seg.coords[-1]   # shapely coords are tuple[float, ...]; take x, y
        u, v = nid((c0[0], c0[1])), nid((c1[0], c1[1]))
        edge_nodes.append((u, v))
        if u == v:
            continue
        w = seg.length
        if not graph.has_edge(u, v) or graph[u][v]["weight"] > w:
            graph.add_edge(u, v, weight=w, idx=idx)

    result = set(selected)
    # connected components of the selection, with each one's total boundary length + member edges
    sub: nx.Graph = nx.Graph()
    sub.add_nodes_from(n for i in selected for n in edge_nodes[i]
                       if edge_nodes[i][0] != edge_nodes[i][1])
    sub.add_edges_from(edge_nodes[i] for i in selected if edge_nodes[i][0] != edge_nodes[i][1])
    comps = [c for c in nx.connected_components(sub)]
    comp_id = {n: cid for cid, comp in enumerate(comps) for n in comp}
    comp_len = [0.0] * len(comps)
    comp_edges: list[list[int]] = [[] for _ in comps]
    for i in selected:
        u, v = edge_nodes[i]
        if u == v:
            continue
        comp_len[comp_id[u]] += edges[i].length
        comp_edges[comp_id[u]].append(i)

    dropped = 0
    kept = []
    for cid in range(len(comps)):
        if comp_len[cid] >= min_component:
            kept.append(cid)
        else:                                          # sub-threshold noise cluster: drop it
            for i in comp_edges[cid]:
                result.discard(i)
                dropped += 1

    street_nodes = {n for n, pt in node_pt.items()
                    if street_geom is not None and Point(pt).distance(street_geom) <= tol}
    street_nodes &= set(graph.nodes)
    if not street_nodes:
        return result, 0, dropped   # no frontage to reach (e.g. streets far away): keep survivors

    added = 0
    backbone = set(street_nodes)
    for cid in sorted(kept, key=lambda c: comp_len[c], reverse=True):   # largest anchors first
        comp = comps[cid]
        if comp & backbone:                            # already touches the growing network
            backbone |= comp
            continue
        # shortest path to whatever is already connected (street or an earlier cluster's connector),
        # recomputed as the backbone grows so connectors are shared into one tree
        dist, paths = nx.multi_source_dijkstra(graph, backbone)
        reachable = [(dist[n], n) for n in comp if n in dist]
        if not reachable:                              # no path to the street at all: isolated
            for i in comp_edges[cid]:
                result.discard(i)
                dropped += 1
            continue
        _, target = min(reachable)
        path = paths[target]
        # pairwise over the path: path[1:] is intentionally one shorter, so strict=False
        for a, b in zip(path, path[1:], strict=False):
            idx = graph[a][b]["idx"]
            if idx not in result:
                result.add(idx)
                added += 1
        backbone |= set(path) | comp
    return result, added, dropped


@dataclass(frozen=True)
class _DensityRaster:
    """A coarse count-per-cell raster of parcel representative points (not centroids --
    guaranteed inside the polygon even for non-convex parcels, see peel.py), binned into
    `res`-sized cells anchored at the lower-left of the point cloud.

    `cells[i]` is the (col, row) index pair of the i-th occupied cell and `counts[i]` its
    point count; empty cells are simply absent.
    """

    minx: float
    miny: float
    res: float
    cells: np.ndarray   # (n, 2) int64 (col, row)
    counts: np.ndarray  # (n,) int64

    def center(self, i: int) -> tuple[float, float]:
        col, row = self.cells[i]
        return (self.minx + (float(col) + 0.5) * self.res,
                self.miny + (float(row) + 0.5) * self.res)

    def extent(self, i: int) -> BaseGeometry:
        col, row = self.cells[i]
        x0 = self.minx + float(col) * self.res
        y0 = self.miny + float(row) * self.res
        return box(x0, y0, x0 + self.res, y0 + self.res)

    def peak(self) -> tuple[float, float]:
        return self.center(int(np.argmax(self.counts)))

    def dense_cells(self, percentile: float) -> list[int]:
        """Indices of the cells at or above the `percentile`-th percentile of occupied-cell
        counts -- i.e. the top (100 - percentile)% densest cells. A perfectly flat field has
        no cell denser than the rest, so it yields no dense cells (and adaptive infill then
        degenerates to the plain base grid rather than to a uniformly finer one)."""
        if int(self.counts.max()) == int(self.counts.min()):
            return []
        threshold = float(np.percentile(self.counts, percentile))
        return [int(i) for i in np.flatnonzero(self.counts >= threshold)]

    def threshold(self, percentile: float) -> float:
        return float(np.percentile(self.counts, percentile))


def _density_raster(block: Block, res: float) -> _DensityRaster:
    reps = [g.representative_point() for g in block.parcels.geometry]
    xs = np.array([p.x for p in reps])
    ys = np.array([p.y for p in reps])
    minx, miny = float(xs.min()), float(ys.min())
    res = max(res, 1e-6)   # guard against a degenerate (zero/negative) resolution
    cols = np.floor((xs - minx) / res).astype(np.int64)
    rows = np.floor((ys - miny) / res).astype(np.int64)
    cells, counts = np.unique(np.stack([cols, rows], axis=1), axis=0, return_counts=True)
    return _DensityRaster(minx=minx, miny=miny, res=res, cells=cells, counts=counts)


def _is_multiple(offset: float, base: float, rel_tol: float = 1e-9) -> bool:
    """Is `offset` an integer multiple of `base` (within a relative tolerance)?"""
    rem = math.remainder(offset, base)   # signed distance to the nearest multiple
    return abs(rem) <= rel_tol * max(abs(base), abs(offset), 1.0)


def _grid_lines(
    boundary: BaseGeometry,
    spacing: float,
    angle: float,
    origin: tuple[float, float],
    skip_multiples_of: float | None = None,
) -> list[LineString]:
    """Evenly-spaced orthogonal candidate lines phased through `origin`, rotated by `angle`
    degrees, spanning `boundary`'s extent and clipped to it.

    `skip_multiples_of` drops the lines whose offset from `origin` is a multiple of that
    (coarser) spacing: an infill layer sharing this origin and angle would otherwise re-emit
    every coarse line it nests inside, duplicating geometry.
    """
    if boundary.is_empty:
        return []
    ox, oy = origin
    # Work in the grid's own frame, where the lines are axis-aligned: un-rotate the boundary,
    # cover its extent there, then rotate the lines back. Phasing off a distant `origin` (a
    # density hotspot, or the block origin while infilling one far-off cell) stays exact.
    local = rotate(boundary, -angle, origin=origin) if angle else boundary
    minx, miny, maxx, maxy = local.bounds
    pad = spacing   # margin so every clipped line spans the full extent

    raw: list[LineString] = []
    for k in range(math.floor((minx - ox) / spacing), math.ceil((maxx - ox) / spacing) + 1):
        offset = k * spacing
        if skip_multiples_of and _is_multiple(offset, skip_multiples_of):
            continue
        x = ox + offset
        raw.append(LineString([(x, miny - pad), (x, maxy + pad)]))
    for k in range(math.floor((miny - oy) / spacing), math.ceil((maxy - oy) / spacing) + 1):
        offset = k * spacing
        if skip_multiples_of and _is_multiple(offset, skip_multiples_of):
            continue
        y = oy + offset
        raw.append(LineString([(minx - pad, y), (maxx + pad, y)]))
    if angle:
        raw = [rotate(ln, angle, origin=(ox, oy)) for ln in raw]

    clipped: list[LineString] = []
    for ln in raw:
        clipped.extend(_line_parts(ln.intersection(boundary), min_len=0.0))
    return clipped


@dataclass
class EuclideanGridReblocker:
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M
    spacing: float = 60.0
    angle: float = 0.0
    min_seg_len: float = 1.0
    street_buffer: float = 0.5
    seek_density: bool = True
    # Default-on: the primary behaviour is the density-concentrated grid -- a coarse grid
    # everywhere, subdivided to `fine_spacing` over the densest parcel clusters -- combined with the
    # always-on parcel-hugging trim and bridge-gap stitching below. Set False for a uniform grid.
    adaptive: bool = True
    # None => spacing / 2, i.e. one extra line between every pair of coarse ones (a quadtree
    # split); it cannot be a plain default because it depends on another field.
    fine_spacing: float | None = None
    density_threshold_percentile: float = 75.0   # infill the top 25% densest cells
    # None => derived per block: `_HUG_NN_FACTOR` x the block's median parcel nearest-neighbour
    # distance. After a candidate line is clipped to the block's extent it is trimmed to just the
    # sub-portion within this distance of an actual parcel, so lines hug the parcel geometry rather
    # than running edge-to-edge across empty gaps. Always on. The default derives from the block's
    # own parcel scale (not `spacing` or a fixed metre value) so it adapts to dense vs sparse
    # blocks; it cannot be a plain default because it depends on the block being processed.
    parcel_hug_buffer: float | None = None
    # None => derived per block: `_BRIDGE_NN_FACTOR` x the same parcel scale. The largest empty gap
    # the hugging trim will span (keep a line whole across) rather than sever; bigger gaps still
    # split the line. Also block-derived, so likewise a per-call None default.
    parcel_bridge_gap: float | None = None
    # follow_parcels: carve roads out of the parcel fabric (shared boundary edges selected by local
    # density) instead of overlaying a grid. When False (default) every field above behaves exactly
    # as before and the three params below are inert. Coverage is the fraction of candidate edges
    # selected: `follow_min_coverage` in the sparsest area, `follow_max_coverage` in the densest,
    # interpolated continuously by the (raster) density score raised to `follow_density_gamma`
    # (>1 pushes selection harder toward the densest areas, <1 spreads it out). The defaults are
    # deliberately sparse -- a near-empty floor, a sharp gamma, and a capped ceiling below 1.0 --
    # so only genuinely dense areas approach high coverage and the result is a legible network
    # concentrated in the settlement's core rather than a mesh blanketing the whole block. The
    # connectivity stitch (step 4 in `_propose_follow_parcels`) still wires the sparse remainder
    # to the street, so a low floor thins the mesh without stranding parcels.
    follow_parcels: bool = False
    follow_min_coverage: float = 0.03
    follow_max_coverage: float = 0.45
    follow_density_gamma: float = 3.0
    # Smallest selected-edge cluster (total boundary length) kept and wired to the street; smaller
    # ones are dropped as noise so the network stays sparse (see `_connect_to_street`). None =>
    # derived per block as `_FOLLOW_MIN_COMPONENT_FACTOR` x the parcel scale, so like the hug/bridge
    # distances it adapts to the block and is encoded "auto" in `identity`. Set 0.0 to keep every
    # cluster (pure connectivity, no noise-dropping).
    follow_min_component: float | None = None

    @property
    def effective_fine_spacing(self) -> float:
        if self.fine_spacing is None:
            return float(self.spacing) / 2.0
        return float(self.fine_spacing)

    @property
    def identity(self) -> tuple[str | float | bool, ...]:
        # A None hug/bridge distance is encoded as "auto", not a number: its resolved value depends
        # on the block, so all "auto" configs share one identity (blocks are keyed separately) while
        # an explicit override keys distinctly.
        def _override(v: float | None) -> float | str:
            return float(v) if v is not None else "auto"
        return ("euclidean_grid", float(self.spacing), float(self.angle),
                float(self.min_seg_len), float(self.street_buffer), bool(self.seek_density),
                bool(self.adaptive), self.effective_fine_spacing,
                float(self.density_threshold_percentile),
                _override(self.parcel_hug_buffer), _override(self.parcel_bridge_gap),
                bool(self.follow_parcels), float(self.follow_min_coverage),
                float(self.follow_max_coverage), float(self.follow_density_gamma),
                _override(self.follow_min_component))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; the grid overlay is block-only
        if block.parcels.empty:
            raise ValueError(
                "Block.parcels must be non-empty: a grid overlay has no extent or density "
                "signal to phase against without parcels"
            )
        if block.streets.empty:
            raise ValueError(
                "Block.streets must be non-empty: with no street frontage there is no "
                "existing network to suppress overlap against"
            )
        if self.follow_parcels:
            return self._propose_follow_parcels(block)
        fine_spacing = self.effective_fine_spacing
        if self.adaptive and fine_spacing <= 0.0:
            raise ValueError(f"fine_spacing must be positive when adaptive, got: {fine_spacing}")

        reps = [g.representative_point() for g in block.parcels.geometry]
        reps_xy = np.array([[p.x, p.y] for p in reps], dtype=float)

        parcel_union = union_all(list(block.parcels.geometry))
        minx, miny, maxx, maxy = parcel_union.bounds
        bbox_center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        # The grid is laid down and clipped across the full parcel bounding box (edge to edge);
        # the parcel-hugging trim below then cuts each line back to just the span near parcels.
        extent_region = box(minx, miny, maxx, maxy)

        hotspot: tuple[float, float] | None = None
        origin = bbox_center
        if self.seek_density:
            hotspot = _density_raster(block, self.spacing / 2.0).peak()
            origin = hotspot

        # coarse layer: the base grid, everywhere
        candidates = _grid_lines(extent_region, self.spacing, self.angle, origin)

        # fine layer: supplemental infill, clipped to each dense cell. Phased off the same
        # origin/angle as the coarse grid, so the finer mesh nests inside the coarse cells
        # instead of cutting across them at an arbitrary offset.
        fine_centers: list[tuple[float, float]] = []
        threshold: float | None = None
        if self.adaptive:
            # one raster cell ~ one coarse grid cell, so infilling a cell subdivides it
            raster = _density_raster(block, self.spacing)
            threshold = raster.threshold(self.density_threshold_percentile)
            for i in raster.dense_cells(self.density_threshold_percentile):
                cell = raster.extent(i).intersection(parcel_union)
                if cell.is_empty:
                    continue
                candidates.extend(
                    _grid_lines(cell, fine_spacing, self.angle, origin,
                                skip_multiples_of=self.spacing)
                )
                fine_centers.append(raster.center(i))

        # parcel-hugging trim (always on): cut every candidate down to just the portion within
        # `parcel_hug_buffer` of an actual parcel, so a line running through an empty gap partway
        # along its length is split down to the parts near parcels rather than kept edge-to-edge.
        # Both distances default to a multiple of the block's own parcel-spacing scale (median
        # nearest-neighbour distance) so the hug is as tight on a dense block as on a sparse one.
        parcel_scale = _median_nn_distance(reps_xy)
        if parcel_scale <= 0.0:   # coincident/degenerate points: fall back to a mean parcel width
            parcel_scale = math.sqrt(parcel_union.area / max(len(block.parcels), 1)) or 1.0
        parcel_hug_buffer = (float(self.parcel_hug_buffer) if self.parcel_hug_buffer is not None
                             else _HUG_NN_FACTOR * parcel_scale)
        parcel_bridge_gap = (float(self.parcel_bridge_gap) if self.parcel_bridge_gap is not None
                             else _BRIDGE_NN_FACTOR * parcel_scale)
        # The hug region is the parcels grown by the buffer but clipped back to the block, so a
        # road never spills past the block boundary into empty space outside it (where the street
        # suppression would then sever it into an isolated fragment). On a block whose parcels tile
        # it this is essentially the block itself; on one where parcels sit in separated clusters
        # the buffer lets lines reach across the small gaps between them, within the block.
        hug_region = parcel_union.buffer(parcel_hug_buffer).intersection(block.boundary)
        hug_length_before = float(sum(ln.length for ln in candidates))
        hugged: list[LineString] = []
        for ln in candidates:   # min_seg_len re-applied here so a trim leaves no sub-length span
            hugged.extend(_hug_line(ln, hug_region, parcel_bridge_gap, self.min_seg_len))
        hug_length_after = float(sum(ln.length for ln in hugged))

        street_union = union_all(list(block.streets.geometry))
        served = street_union.buffer(self.street_buffer)

        segments: list[LineString] = []
        for ln in hugged:
            # difference (not an all-or-nothing overlap check) so a line that only partly
            # hugs an existing street keeps its non-overlapping remainder. The min_seg_len
            # sliver-drop here also discards any tiny fragments left by the hugging trim above.
            segments.extend(_line_parts(ln.difference(served), self.min_seg_len))

        roads = gpd.GeoDataFrame(geometry=segments, crs=block.crs)
        params: dict[str, object] = {
            "spacing": self.spacing, "angle": self.angle, "min_seg_len": self.min_seg_len,
            "street_buffer": self.street_buffer, "seek_density": self.seek_density,
            "adaptive": self.adaptive, "segments": len(roads),
            "parcel_scale": parcel_scale,
            "parcel_hug_buffer": parcel_hug_buffer, "parcel_bridge_gap": parcel_bridge_gap,
            "hug_length_before": hug_length_before, "hug_length_after": hug_length_after,
            "hug_length_trimmed": hug_length_before - hug_length_after,
        }
        if hotspot is not None:
            params["density_hotspot"] = hotspot
        if self.adaptive:
            params["fine_spacing"] = fine_spacing
            params["density_threshold_percentile"] = self.density_threshold_percentile
            params["density_threshold"] = threshold
            params["fine_cells"] = len(fine_centers)
            params["fine_cell_centers"] = fine_centers
        pid = (f"euclidean_grid:sp{self.spacing:g}:a{self.angle:g}:msl{self.min_seg_len:g}"
               f":sb{self.street_buffer:g}:sd{int(self.seek_density)}:ad{int(self.adaptive)}"
               f":fs{fine_spacing:g}:dtp{self.density_threshold_percentile:g}"
               f":phb{parcel_hug_buffer:g}:pbg{parcel_bridge_gap:g}")
        return Proposal(block_id=block.block_id, crs=block.crs, edges=None,
                        roads=with_width(roads, self.road_width_m),
                        proposal_id=pid, method="euclidean_grid", params=params,
                        block_identity=block.identity)

    def _propose_follow_parcels(self, block: Block) -> Proposal:
        """Carve roads from the parcel fabric: select shared parcel-boundary edges by local density,
        stitch them to the street frontage, and suppress overlap with existing streets."""
        geoms = list(block.parcels.geometry)
        reps_xy = np.array([[p.x, p.y] for p in (g.representative_point() for g in geoms)],
                           dtype=float)
        parcel_scale = _median_nn_distance(reps_xy)
        if parcel_scale <= 0.0:
            parcel_union = union_all(geoms)
            parcel_scale = math.sqrt(parcel_union.area / max(len(geoms), 1)) or 1.0

        # 1. candidate roads = the shared boundary edges between adjacent parcels, split to single
        #    parcel-edge scale (never merged into long chains, per the granularity requirement)
        adj = parcel_adjacency(geoms, STREET_TOL)
        edges: list[LineString] = []
        for i in range(len(geoms)):
            for j in adj[i]:
                if i < j:
                    for part in _line_parts(_shared_boundary(geoms[i], geoms[j], STREET_TOL), 0.0):
                        edges.extend(_straight_segments(part))

        # 2. score each edge by the local parcel density (reuse the raster; cell holds ~several
        #    parcels), normalised to [0, 1] against the densest cell -> a continuous field
        res = max(_FOLLOW_DENSITY_RES_FACTOR * parcel_scale, 1e-6)
        raster = _density_raster(block, res)
        cell_count = {(int(c), int(r)): int(n)   # cells & counts are same-length by construction
                      for (c, r), n in zip(raster.cells, raster.counts, strict=True)}
        max_count = float(raster.counts.max()) if len(raster.counts) else 1.0

        def density_score(x: float, y: float) -> float:
            col = math.floor((x - raster.minx) / res)
            row = math.floor((y - raster.miny) / res)
            return cell_count.get((col, row), 0) / max_count

        # 3. density-weighted selection: probability rises continuously with local density from
        #    `follow_min_coverage` (sparse) to `follow_max_coverage` (dense) -- gradient, not a cut
        lo, hi, gamma = (self.follow_min_coverage, self.follow_max_coverage,
                         self.follow_density_gamma)
        scores: list[float] = []
        selected: set[int] = set()
        for idx, seg in enumerate(edges):
            mid = seg.interpolate(0.5, normalized=True)
            score = density_score(mid.x, mid.y)
            scores.append(score)
            prob = lo + (hi - lo) * (score ** gamma)
            if _unit_hash(mid.x, mid.y) < prob:
                selected.add(idx)
        n_selected = len(selected)

        # 4. drop noise clusters and stitch the survivors to a shared street-reaching backbone
        min_component = (float(self.follow_min_component) if self.follow_min_component is not None
                         else _FOLLOW_MIN_COMPONENT_FACTOR * parcel_scale)
        selected, added, dropped = _connect_to_street(
            edges, selected, block.streets, STREET_TOL, min_component)

        # 5. suppress overlap with existing streets + drop sub-min_seg_len slivers (as in grid mode)
        served = union_all(list(block.streets.geometry)).buffer(self.street_buffer)
        segments: list[LineString] = []
        for idx in sorted(selected):
            segments.extend(_line_parts(edges[idx].difference(served), self.min_seg_len))
        roads = gpd.GeoDataFrame(geometry=segments, crs=block.crs)

        sel_scores = [scores[i] for i in selected]
        params: dict[str, object] = {
            "follow_parcels": True, "min_seg_len": self.min_seg_len,
            "street_buffer": self.street_buffer, "parcel_scale": parcel_scale,
            "follow_min_coverage": lo, "follow_max_coverage": hi,
            "follow_density_gamma": gamma, "follow_density_res": res,
            "follow_min_component": min_component,
            "boundary_edges_considered": len(edges),
            "boundary_edges_selected": n_selected,
            "connectivity_edges_added": added, "connectivity_edges_dropped": dropped,
            "selected_density_min": min(sel_scores) if sel_scores else 0.0,
            "selected_density_max": max(sel_scores) if sel_scores else 0.0,
            "segments": len(roads),
        }
        pid = (f"euclidean_grid:follow:msl{self.min_seg_len:g}:sb{self.street_buffer:g}"
               f":cov{lo:g}-{hi:g}:g{gamma:g}:mc{min_component:g}")
        return Proposal(block_id=block.block_id, crs=block.crs, edges=None,
                        roads=with_width(roads, self.road_width_m),
                        proposal_id=pid, method="euclidean_grid", params=params,
                        block_identity=block.identity)
    