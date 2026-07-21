from pathlib import Path
from typing import cast

import geopandas as gpd
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pyproj import CRS
from shapely import union_all
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.methods.euclidean_grid import EuclideanGridReblocker

UTM = CRS.from_epsg(32643)


def _parcel_boundary_union(block: Block) -> BaseGeometry:
    return union_all([g.boundary for g in block.parcels.geometry])


def _rect_block(w: float, h: float, streets: list[LineString] | None = None) -> Block:
    """A single rectangular parcel spanning (0, 0)-(w, h), with a street far from the
    parcel (default) or the caller-supplied street lines."""
    poly = box(0, 0, w, h)
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[poly], crs=UTM)
    lines = streets if streets is not None else [LineString([(-100, -100), (-99, -100)])]
    street_gdf = gpd.GeoDataFrame(geometry=lines, crs=UTM)
    return Block(block_id="rect", crs=UTM, boundary=poly, parcels=parcels, streets=street_gdf)


def _clustered_block() -> Block:
    """A dense 3x3 cluster of unit parcels near the origin plus a single far parcel --
    the far parcel drags the bbox center well away from the cluster, so a bbox-centered
    grid phase and a density-seeking one land in visibly different places."""
    cluster = [box(i, j, i + 1, j + 1) for i in range(3) for j in range(3)]
    far = box(90, 90, 91, 91)
    polys = cluster + [far]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all().convex_hull)
    streets = gpd.GeoDataFrame(geometry=[LineString([(200, 200), (201, 200)])], crs=UTM)
    return Block(block_id="cluster", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _dense_cluster_block() -> Block:
    """A 60x60 block tiled by 20x20 parcels, except its lower-left corner ([0, 20]^2), which is
    finely subdivided into 2x2 parcels: with a 10 m raster that corner's four cells hold 25
    parcel points each against 1 per cell elsewhere -- an informal-settlement-like cluster in
    an otherwise sparse block."""
    dense = [box(2 * i, 2 * j, 2 * i + 2, 2 * j + 2) for i in range(10) for j in range(10)]
    sparse = [box(20 * i, 20 * j, 20 * i + 20, 20 * j + 20)
              for i in range(3) for j in range(3) if (i, j) != (0, 0)]
    polys = dense + sparse
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = box(0, 0, 60, 60)
    streets = gpd.GeoDataFrame(geometry=[LineString([(200, 200), (201, 200)])], crs=UTM)
    return Block(block_id="dense", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _dumbbell_block() -> Block:
    """Two 3x3 clusters of unit parcels (x in [0, 3] and x in [50, 53], both y in [0, 3])
    separated by a wide empty gap. The parcel bounding box spans x in [0, 53], so a horizontal
    grid line laid across the block runs edge-to-edge through the empty middle -- exactly the
    kind of line the parcel-hugging trim must cut down to just its two near-parcel ends."""
    left = [box(i, j, i + 1, j + 1) for i in range(3) for j in range(3)]
    right = [box(50 + i, j, 51 + i, j + 1) for i in range(3) for j in range(3)]
    polys = left + right
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = box(0, 0, 53, 3)
    streets = gpd.GeoDataFrame(geometry=[LineString([(200, 200), (201, 200)])], crs=UTM)
    return Block(block_id="dumbbell", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


CLUSTER_AREA = box(0, 0, 20, 20)     # the finely-subdivided corner of _dense_cluster_block
SPARSE_AREA = box(40, 40, 60, 60)    # a far corner of it, one 20x20 parcel


def _length_in(roads: gpd.GeoDataFrame, region: Polygon) -> float:
    return float(sum(g.intersection(region).length for g in roads.geometry))


def _uniform_block() -> Block:
    """A 3x3 tiling of 10x10 parcels: with a 10 m raster every cell holds exactly one parcel
    representative point, so the density field is perfectly flat -- no cluster to refine."""
    polys = [box(10 * i, 10 * j, 10 * i + 10, 10 * j + 10) for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(200, 200), (201, 200)])], crs=UTM)
    return Block(block_id="uniform", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _two_cluster_block(gap: float) -> Block:
    """Two 6x6 clusters of unit parcels (parcel spacing ~ 1 m) separated by a `gap`-metre empty
    strip, inside a block that spans the whole thing. `parcel_union` is a MultiPolygon of the two
    clusters, so a grid line runs across the empty strip between them."""
    left = [box(i, j, i + 1, j + 1) for i in range(6) for j in range(6)]
    rx = 6.0 + gap
    right = [box(rx + i, j, rx + i + 1, j + 1) for i in range(6) for j in range(6)]
    polys = left + right
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = box(0, 0, rx + 6, 6)
    streets = gpd.GeoDataFrame(geometry=[LineString([(500, 500), (501, 500)])], crs=UTM)
    return Block(block_id="two", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _road_components(roads: gpd.GeoDataFrame) -> int:
    """Connected components of the road network: two segments are joined when they touch or cross
    (a grid's perpendicular lines share crossing points), so a fully-connected grid is one."""
    geoms = list(roads.geometry)
    parent = list(range(len(geoms)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            if geoms[i].intersects(geoms[j]):
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(geoms))})


def _no_collinear_overlap(roads: gpd.GeoDataFrame) -> bool:
    """No two segments run along each other (crossings meet at points, so they cost no length):
    the union's length drops below the sum only if some geometry is duplicated."""
    total = float(sum(g.length for g in roads.geometry))
    return abs(union_all(list(roads.geometry)).length - total) <= 1e-6 * max(total, 1.0)


def test_identity_encodes_all_tunable_params() -> None:
    m = EuclideanGridReblocker()
    # the hug buffer and bridge gap default to a per-block-derived value, so identity encodes them
    # as "auto" (their resolved metres depend on the block, which is keyed separately) not a number
    # adaptive defaults on (the density-concentrated grid is the primary behaviour)
    assert m.identity == ("euclidean_grid", 60.0, 0.0, 1.0, 0.5, True, True, 30.0, 75.0,
                          "auto", "auto", False, 0.03, 0.45, 3.0, "auto")
    m2 = EuclideanGridReblocker(spacing=30.0, angle=15.0, min_seg_len=2.0,
                                street_buffer=1.0, seek_density=False)
    assert m2.identity == ("euclidean_grid", 30.0, 15.0, 2.0, 1.0, False, True, 15.0, 75.0,
                           "auto", "auto", False, 0.03, 0.45, 3.0, "auto")
    assert m.identity != m2.identity
    # an explicit hug/bridge override is a distinct config from the auto default and from each other
    assert (EuclideanGridReblocker(parcel_hug_buffer=5.0).identity
            != EuclideanGridReblocker().identity)
    assert (EuclideanGridReblocker(parcel_bridge_gap=5.0).identity
            != EuclideanGridReblocker().identity)
    assert (EuclideanGridReblocker(parcel_hug_buffer=5.0).identity
            != EuclideanGridReblocker(parcel_bridge_gap=5.0).identity)
    # the follow-parcels mode and its coverage / stitch knobs are distinct identity axes too
    assert (EuclideanGridReblocker(follow_parcels=True).identity
            != EuclideanGridReblocker(follow_parcels=False).identity)
    assert (EuclideanGridReblocker(follow_parcels=True, follow_min_coverage=0.3).identity
            != EuclideanGridReblocker(follow_parcels=True).identity)
    assert (EuclideanGridReblocker(follow_parcels=True, follow_min_component=5.0).identity
            != EuclideanGridReblocker(follow_parcels=True).identity)


def test_identity_separates_adaptive_configs() -> None:
    base = EuclideanGridReblocker(spacing=20.0, adaptive=False)   # adaptive now defaults on
    variants = [
        EuclideanGridReblocker(spacing=20.0, adaptive=True),
        EuclideanGridReblocker(spacing=20.0, adaptive=True, fine_spacing=5.0),
        EuclideanGridReblocker(spacing=20.0, adaptive=True, density_threshold_percentile=50.0),
        EuclideanGridReblocker(spacing=20.0, parcel_hug_buffer=3.0),
        EuclideanGridReblocker(spacing=20.0, parcel_bridge_gap=3.0),
        EuclideanGridReblocker(spacing=20.0, follow_parcels=True),
        EuclideanGridReblocker(spacing=20.0, follow_parcels=True, follow_density_gamma=2.0),
    ]
    identities = [base.identity, *(v.identity for v in variants)]
    assert len(set(identities)) == len(identities)   # no cache-key collisions between configs
    # an explicit fine_spacing equal to the default is the same config, not a new one
    assert (EuclideanGridReblocker(spacing=20.0, fine_spacing=10.0).identity
            == EuclideanGridReblocker(spacing=20.0).identity)


def test_basic_grid_generation_on_rectangular_block() -> None:
    block = _rect_block(100.0, 100.0)
    m = EuclideanGridReblocker(spacing=30.0, seek_density=False)
    proposal = m.propose(block)
    roads = proposal.roads
    assert roads is not None and not roads.empty
    boundary = union_all(list(block.parcels.geometry))
    for g in roads.geometry:
        assert g.geom_type == "LineString"
        assert g.length >= m.min_seg_len
        assert boundary.buffer(1e-6).contains(g)   # every candidate is clipped to the block
    assert proposal.method == "euclidean_grid"
    assert proposal.proposal_id.startswith("euclidean_grid:")
    assert proposal.crs == block.crs
    assert proposal.block_identity == block.identity
    assert "density_hotspot" not in proposal.params  # seek_density=False -> no hotspot recorded


def test_overlap_suppression_keeps_the_nonoverlapping_remainder() -> None:
    # bbox-centered phase on a 20x20 block (spacing 10) puts a vertical grid line at x=10
    # spanning y in [0, 20]; a street covers only the bottom half (y in [0, 12]) of it.
    street = LineString([(10, 0), (10, 12)])
    block = _rect_block(20.0, 20.0, streets=[street])
    m = EuclideanGridReblocker(spacing=10.0, street_buffer=0.5, min_seg_len=1.0, seek_density=False)
    proposal = m.propose(block)
    roads = proposal.roads
    assert roads is not None and not roads.empty

    served = street.buffer(0.5)
    all_roads = union_all(list(roads.geometry))
    assert all_roads.intersection(served).length < 1e-6   # overlap fully suppressed

    # the non-overlapping remainder above the street (y roughly 12.5..20) must survive --
    # a partial overlap keeps its clear portion rather than dropping the whole candidate line
    assert any(g.intersects(box(9, 15, 11, 20)) for g in roads.geometry)


def test_empty_streets_raises() -> None:
    block = _rect_block(20.0, 20.0)
    bad = Block(block_id="nostreet", crs=block.crs, boundary=block.boundary,
                parcels=block.parcels, streets=gpd.GeoDataFrame(geometry=[], crs=block.crs))
    with pytest.raises(ValueError, match="streets"):
        EuclideanGridReblocker().propose(bad)


def test_seek_density_shifts_grid_phase_toward_the_cluster() -> None:
    block = _clustered_block()
    # adaptive=False to isolate the phase shift (adaptive now defaults on)
    m_density = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=True,
                                       adaptive=False)
    m_plain = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                     adaptive=False)
    p_density = m_density.propose(block)
    p_plain = m_plain.propose(block)

    assert "density_hotspot" in p_density.params
    hx, hy = cast(tuple[float, float], p_density.params["density_hotspot"])
    assert Point(hx, hy).distance(Point(1.5, 1.5)) < 5.0   # phased near the dense cluster
    assert "density_hotspot" not in p_plain.params

    assert p_density.roads is not None and p_plain.roads is not None
    r1 = sorted(g.wkt for g in p_density.roads.geometry)
    r2 = sorted(g.wkt for g in p_plain.roads.geometry)
    assert r1 != r2   # different phase origin -> different grid alignment


def test_adaptive_infills_a_finer_mesh_over_the_dense_cluster() -> None:
    block = _dense_cluster_block()
    plain = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                   adaptive=False)
    fine = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False, adaptive=True)
    p_plain = plain.propose(block)
    p_fine = fine.propose(block)
    assert p_plain.roads is not None and p_fine.roads is not None

    # only the cluster is refined: its four 10 m cells, and none of the sparse ones
    assert p_fine.params["fine_cells"] == 4
    centers = cast(list[tuple[float, float]], p_fine.params["fine_cell_centers"])
    assert all(CLUSTER_AREA.buffer(1.0).contains(Point(cx, cy)) for cx, cy in centers)
    assert p_fine.params["fine_spacing"] == 5.0            # spacing / 2 by default
    assert p_fine.params["density_threshold_percentile"] == 75.0

    # the coarse layer survives intact underneath, and the infill only adds to it
    assert len(p_fine.roads) > len(p_plain.roads)
    assert p_fine.params["segments"] == len(p_fine.roads)
    assert _no_collinear_overlap(p_fine.roads)   # infill never re-emits a coarse line

    # the extra mesh lands *in the cluster*, not spread over the whole block: the cluster's
    # road density roughly doubles (fine_spacing = spacing / 2) while the sparse corner is
    # left exactly as the coarse grid had it
    assert _length_in(p_fine.roads, CLUSTER_AREA) > 1.5 * _length_in(p_plain.roads, CLUSTER_AREA)
    assert _length_in(p_fine.roads, SPARSE_AREA) == pytest.approx(
        _length_in(p_plain.roads, SPARSE_AREA))

    boundary = union_all(list(block.parcels.geometry))
    for g in p_fine.roads.geometry:   # clip/sliver rules apply to the fine layer too
        assert g.geom_type == "LineString"
        assert g.length >= fine.min_seg_len
        assert boundary.buffer(1e-6).contains(g)


def test_adaptive_on_a_flat_density_field_adds_nothing() -> None:
    block = _uniform_block()
    plain = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                   adaptive=False)
    fine = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False, adaptive=True)
    p_plain, p_fine = plain.propose(block), fine.propose(block)
    assert p_plain.roads is not None and p_fine.roads is not None

    assert p_fine.params["fine_cells"] == 0   # no cell is denser than the rest
    assert (sorted(g.wkt for g in p_fine.roads.geometry)
            == sorted(g.wkt for g in p_plain.roads.geometry))


def test_finer_fine_spacing_and_lower_percentile_add_more_mesh() -> None:
    block = _dense_cluster_block()
    base = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False, adaptive=True)
    finer = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                   adaptive=True, fine_spacing=2.5)
    looser = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                    adaptive=True, density_threshold_percentile=0.0)
    p_base, p_finer, p_looser = base.propose(block), finer.propose(block), looser.propose(block)
    assert p_base.roads is not None and p_finer.roads is not None and p_looser.roads is not None

    # a tighter infill mesh over the same four cells
    assert p_finer.params["fine_cells"] == p_base.params["fine_cells"]
    assert _length_in(p_finer.roads, CLUSTER_AREA) > _length_in(p_base.roads, CLUSTER_AREA)
    assert _no_collinear_overlap(p_finer.roads)

    # percentile 0 -> every occupied cell qualifies, so the infill spills into the sparse areas
    assert cast(int, p_looser.params["fine_cells"]) > cast(int, p_base.params["fine_cells"])
    assert _length_in(p_looser.roads, SPARSE_AREA) > _length_in(p_base.roads, SPARSE_AREA)


def test_seek_density_and_adaptive_are_independent_and_composable() -> None:
    block = _dense_cluster_block()
    p_seek = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1,
                                    seek_density=True, adaptive=False).propose(block)
    p_both = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1,
                                    seek_density=True, adaptive=True).propose(block)
    p_adapt = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1,
                                     seek_density=False, adaptive=True).propose(block)
    assert p_seek.roads is not None and p_both.roads is not None and p_adapt.roads is not None

    # both flags active: the base grid is still phased on the hotspot, and infill still lands
    assert "density_hotspot" in p_both.params and p_both.params["fine_cells"] == 4
    assert "density_hotspot" not in p_adapt.params      # adaptive alone does not phase-shift
    assert _length_in(p_both.roads, CLUSTER_AREA) > _length_in(p_seek.roads, CLUSTER_AREA)
    assert _no_collinear_overlap(p_both.roads)
    # the phase still matters under adaptive: same infill cells, different alignment for both
    # the coarse layer and the fine mesh nested in it
    assert (sorted(g.wkt for g in p_both.roads.geometry)
            != sorted(g.wkt for g in p_adapt.roads.geometry))


def test_adaptive_false_ignores_the_infill_params() -> None:
    block = _dense_cluster_block()
    p_plain = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=True,
                                     adaptive=False).propose(block)
    p_inert = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=True,
                                     adaptive=False, fine_spacing=1.0,
                                     density_threshold_percentile=10.0).propose(block)
    assert p_plain.roads is not None and p_inert.roads is not None
    assert (sorted(g.wkt for g in p_inert.roads.geometry)
            == sorted(g.wkt for g in p_plain.roads.geometry))
    assert p_plain.params["adaptive"] is False
    for key in ("fine_cells", "fine_cell_centers", "fine_spacing", "density_threshold"):
        assert key not in p_plain.params


def test_default_is_the_adaptive_hugging_bridging_grid() -> None:
    # the out-of-the-box method (no flags) is the density-concentrated straight-line grid with the
    # hugging trim and bridge-gap stitching -- NOT follow_parcels, NOT a uniform grid
    m = EuclideanGridReblocker()
    assert m.adaptive is True and m.seek_density is True and m.follow_parcels is False

    # only spacing/min_seg_len set (to suit the small test block); every mode flag stays default
    p = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1).propose(_dense_cluster_block())
    roads = p.roads
    assert roads is not None and not roads.empty
    assert "follow_parcels" not in p.params                 # the grid path, not the parcel path
    for g in roads.geometry:                                # straight orthogonal lines, not veins
        assert g.geom_type == "LineString"

    # adaptive infill fired over the dense cluster, seek_density phased onto it, hug + bridge ran
    # (this block is fully parcel-tiled, so the hug trims nothing here -- its bite is covered by the
    # dedicated hug tests; what matters is that the stage is wired into the default pipeline)
    assert cast(int, p.params["fine_cells"]) > 0
    assert "density_hotspot" in p.params
    assert "parcel_hug_buffer" in p.params and "parcel_bridge_gap" in p.params
    assert cast(float, p.params["hug_length_before"]) > 0.0
    # density concentration: more grid road length over the dense cluster than the sparse corner
    assert _length_in(roads, CLUSTER_AREA) > _length_in(roads, SPARSE_AREA)


def test_parcel_hug_trims_lines_tightly_to_the_parcel_geometry() -> None:
    block = _dumbbell_block()
    parcel_union = union_all(list(block.parcels.geometry))
    # no hug/bridge parameter is set: the trim is unconditional, always-on default behaviour
    # (adaptive=False to isolate the hugging trim from the density infill, now default-on)
    m = EuclideanGridReblocker(spacing=3.0, min_seg_len=0.1, seek_density=False, adaptive=False)
    proposal = m.propose(block)
    roads = proposal.roads
    assert roads is not None and not roads.empty

    # the buffer is derived from the block's OWN parcel spacing (unit parcels -> nn ~ 1 m), not from
    # `spacing`: it lands well below the old flat spacing / 3 (= 1.0 m) default
    scale = cast(float, proposal.params["parcel_scale"])
    hug = cast(float, proposal.params["parcel_hug_buffer"])
    assert scale == pytest.approx(1.0, abs=0.2)
    assert hug == pytest.approx(0.5 * scale)
    assert hug < 1.0

    # a genuinely TIGHT trim: the majority of candidate length (the wide empty middle) is removed,
    # not just a small fraction
    before = cast(float, proposal.params["hug_length_before"])
    after = cast(float, proposal.params["hug_length_after"])
    trimmed = cast(float, proposal.params["hug_length_trimmed"])
    assert trimmed == pytest.approx(before - after)
    assert trimmed > 0.5 * before

    # every road hugs the parcels: even its endpoints stay within the buffer of some parcel, so no
    # segment reaches out across the empty middle where there are no parcels nearby
    for g in roads.geometry:
        for x, y in g.coords:
            assert Point(x, y).distance(parcel_union) <= hug + 1e-6
    # concretely, nothing survives in the wide empty gap between the two clusters...
    assert all(g.intersection(box(8, -5, 45, 8)).length < 1e-9 for g in roads.geometry)
    # ...yet both clusters keep roads, so the trim did not simply drop everything
    assert any(g.intersects(box(0, 0, 3, 3)) for g in roads.geometry)
    assert any(g.intersects(box(50, 0, 53, 3)) for g in roads.geometry)


def test_parcel_hug_buffer_scales_with_the_blocks_own_parcel_spacing() -> None:
    def block_at(step: float) -> Block:
        polys = [box(step * i, step * j, step * i + 0.9 * step, step * j + 0.9 * step)
                 for i in range(5) for j in range(5)]
        parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
        boundary = box(-step, -step, 6 * step, 6 * step)
        streets = gpd.GeoDataFrame(geometry=[LineString([(1e4, 1e4), (1e4 + 1, 1e4)])], crs=UTM)
        return Block(block_id=f"s{step}", crs=UTM, boundary=boundary, parcels=parcels,
                     streets=streets)

    # identical layout, one 10x sparser than the other, SAME spacing -> the derived hug buffer
    # tracks the block's own parcel scale, not `spacing` (dataset-agnostic across dense/sparse)
    dense = EuclideanGridReblocker(spacing=5.0, seek_density=False).propose(block_at(1.0))
    sparse = EuclideanGridReblocker(spacing=5.0, seek_density=False).propose(block_at(10.0))
    assert cast(float, dense.params["parcel_scale"]) == pytest.approx(1.0, abs=0.2)
    assert cast(float, sparse.params["parcel_scale"]) == pytest.approx(
        10.0 * cast(float, dense.params["parcel_scale"]), rel=0.2)
    assert cast(float, sparse.params["parcel_hug_buffer"]) == pytest.approx(
        10.0 * cast(float, dense.params["parcel_hug_buffer"]), rel=0.2)
    assert (cast(float, dense.params["parcel_hug_buffer"])
            < cast(float, sparse.params["parcel_hug_buffer"]))


def test_hug_bridges_small_gaps_but_severs_wide_ones() -> None:
    # unit parcels -> nn ~ 1 m, so the default buffer ~ 0.5 m and bridge gap ~ 3 m: gaps up to
    # roughly bridge + 2*buffer (~ 4 m) are spanned to keep the network whole, wider ones are cut.
    def propose(gap: float, **kw: object) -> gpd.GeoDataFrame:
        p = EuclideanGridReblocker(spacing=2.0, min_seg_len=0.1, seek_density=False,
                                   adaptive=False, **kw).propose(_two_cluster_block(gap))
        assert p.roads is not None
        return p.roads

    # a small (3 m) gap: bridging keeps the two clusters connected as one network, where turning
    # bridging off (parcel_bridge_gap=0) severs them into separate components
    assert _road_components(propose(3.0)) < _road_components(propose(3.0, parcel_bridge_gap=0.0))

    # a wide (12 m) gap is never bridged: no road runs through the empty strip between the clusters
    wide = propose(12.0)
    assert all(g.intersection(box(8, -1, 16, 7)).length < 1e-9 for g in wide.geometry)   # x 6..18


def _street_bounded_tiling(n: int) -> Block:
    """An n x n tiling of unit parcels whose block boundary IS the street frontage (as a real
    street-bounded kblock face is) -- so follow_parcels roads have a frontage to reach."""
    polys = [box(i, j, i + 1, j + 1) for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = box(0, 0, n, n)
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="tiling", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_follow_parcels_follows_boundary_edges_weighted_by_density() -> None:
    block = _dense_cluster_block()
    proposal = EuclideanGridReblocker(follow_parcels=True).propose(block)
    roads = proposal.roads
    assert roads is not None and not roads.empty
    assert proposal.method == "euclidean_grid"
    assert proposal.params["follow_parcels"] is True

    # every road segment lies on an actual shared parcel boundary -- not an arbitrary grid line
    boundaries = _parcel_boundary_union(block).buffer(1e-6)
    for g in roads.geometry:
        assert boundaries.contains(g)

    # density gradient: the dense corner gets far more road length per unit area than a sparse one
    # (CLUSTER_AREA and SPARSE_AREA are equal-area 20x20 squares)
    assert _length_in(roads, CLUSTER_AREA) > 3.0 * _length_in(roads, SPARSE_AREA)

    # traceability params: edges considered vs selected, and the density range across selected edges
    assert (0 < cast(int, proposal.params["boundary_edges_selected"])
            <= cast(int, proposal.params["boundary_edges_considered"]))
    assert (cast(float, proposal.params["selected_density_max"])
            > cast(float, proposal.params["selected_density_min"]))


def test_follow_parcels_defaults_select_a_sparse_minority_of_edges() -> None:
    # the tuned-sparse defaults (low floor, sharp gamma, sub-1.0 ceiling) must pick only a minority
    # of candidate edges, so the network is legible rather than a mesh blanketing the whole block
    block = _dense_cluster_block()
    default = EuclideanGridReblocker(follow_parcels=True).propose(block)
    full = EuclideanGridReblocker(follow_parcels=True, follow_min_coverage=1.0,
                                  follow_max_coverage=1.0).propose(block)
    considered = cast(int, default.params["boundary_edges_considered"])
    selected = cast(int, default.params["boundary_edges_selected"])
    assert cast(int, full.params["boundary_edges_selected"]) == considered   # coverage 1.0 -> all
    assert selected < 0.5 * considered                                       # a sparse minority
    assert selected < 0.5 * cast(int, full.params["boundary_edges_selected"])


def test_follow_parcels_segments_stay_at_individual_parcel_edge_scale() -> None:
    # roads follow real parcel edges and are never concatenated into long chains: every emitted
    # segment is a single straight span (exactly two vertices), bounded by one parcel edge
    block = _dense_cluster_block()
    roads = EuclideanGridReblocker(follow_parcels=True).propose(block).roads
    assert roads is not None and not roads.empty
    for g in roads.geometry:
        assert g.geom_type == "LineString"
        assert len(g.coords) == 2


def test_follow_parcels_network_reaches_the_street_frontage() -> None:
    block = _street_bounded_tiling(8)
    # a deliberately sparse, uniform coverage fragments the raw selection; the connectivity stitch
    # must then wire every surviving piece back to the street frontage. follow_min_component=0.0
    # keeps noise-dropping out of the way so this isolates the street-reaching stitch guarantee.
    m = EuclideanGridReblocker(follow_parcels=True, follow_min_coverage=0.3,
                               follow_max_coverage=0.3, follow_min_component=0.0,
                               street_buffer=0.1, min_seg_len=0.1)
    proposal = m.propose(block)
    roads = proposal.roads
    assert roads is not None and not roads.empty

    assert cast(int, proposal.params["connectivity_edges_added"]) > 0   # stitching happened
    # all road length lives in components that reach a street (no floating interior slivers)
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == pytest.approx(1.0)
    boundaries = _parcel_boundary_union(block).buffer(1e-6)
    for g in roads.geometry:
        assert boundaries.contains(g)


def test_follow_parcels_min_component_drops_noise_for_a_sparser_network() -> None:
    block = _dense_cluster_block()
    # the default noise-drop (follow_min_component derived from parcel scale) removes the lone edges
    # the sparse floor scatters through low-density areas, leaving a sparser network than keeping
    # every cluster (follow_min_component=0.0) would
    default = EuclideanGridReblocker(follow_parcels=True).propose(block)
    unfiltered = EuclideanGridReblocker(
        follow_parcels=True, follow_min_component=0.0).propose(block)
    assert cast(int, default.params["connectivity_edges_dropped"]) > 0
    assert cast(float, default.params["follow_min_component"]) > 0.0
    assert len(default.roads) < len(unfiltered.roads)          # a genuinely sparser final network
    # what survives still concentrates in the dense cluster, not the sparse corner
    assert (_length_in(default.roads, CLUSTER_AREA)
            > 3.0 * _length_in(default.roads, SPARSE_AREA))
    boundaries = _parcel_boundary_union(block).buffer(1e-6)   # still real parcel-boundary geometry
    for g in default.roads.geometry:
        assert boundaries.contains(g)


def test_follow_parcels_is_deterministic() -> None:
    block = _street_bounded_tiling(6)
    m = EuclideanGridReblocker(follow_parcels=True, street_buffer=0.1, min_seg_len=0.1)
    a = sorted(g.wkt for g in m.propose(block).roads.geometry)
    b = sorted(g.wkt for g in m.propose(block).roads.geometry)
    assert a == b and len(a) > 0


def test_follow_parcels_off_preserves_grid_behaviour() -> None:
    block = _dense_cluster_block()
    # follow_parcels defaults off; the proposal is the ordinary grid, with no follow-mode params
    grid = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False)
    assert grid.follow_parcels is False
    p_default = grid.propose(block)
    p_explicit = EuclideanGridReblocker(spacing=10.0, min_seg_len=0.1, seek_density=False,
                                        follow_parcels=False).propose(block)
    assert "follow_parcels" not in p_default.params        # grid path records no follow-mode keys
    assert "spacing" in p_default.params
    assert (sorted(g.wkt for g in p_default.roads.geometry)
            == sorted(g.wkt for g in p_explicit.roads.geometry))


def test_euclidean_grid_method_yaml_instantiates_with_defaults() -> None:
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=["method=euclidean_grid"])
    method = instantiate(cfg.method)
    assert isinstance(method, EuclideanGridReblocker)
    assert method.identity == (
        "euclidean_grid", 60.0, 0.0, 1.0, 0.5, True, True, 30.0, 75.0, "auto", "auto",
        False, 0.03, 0.45, 3.0, "auto")
