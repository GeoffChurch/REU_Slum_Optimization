"""Clearance reblocker — the repulsion knob on ONE auto-detected deep Cape Town region.

True end-to-end: screen the full metro (DenseCompactScreen, memoized), auto-pick the deepest
seed whose own building_count is in a tractable window, grow it into a small multi-block
neighborhood (DenseClusterRegionBuilder), build the region, then reblock at five repulsions
(the two extremes, two moderates, and the balanced default) and render each. Reproduces the
committed PNGs. Run: PYTHONPATH=. pixi run python examples/clearance-repulsion/generate.py
"""
from __future__ import annotations

from pathlib import Path

from reblock.budget import displacement_count
from reblock.data.kblock import KblockSource
from reblock.data.provision import cached_kblock_source
from reblock.derivations import access_before
from reblock.derive.access import parcel_access_layers
from reblock.methods.clearance import ClearanceReblocker
from reblock.region import DenseClusterRegionBuilder, region_block
from reblock.render import frame_bbox, render_after, render_before, save_render
from reblock.screen.dense_compact import DenseCompactScreen

OUT = Path(__file__).parent
REPULSIONS = [-6.0, -2.0, 0.0, 2.0, 6.0]     # extremes, moderates, balanced default
SEED_MIN, SEED_MAX = 40, 90                  # tractable deep seed (giants alone are 1000-3000)
REGION_MAX = 100                             # region buildings budget -> ~a dozen roads/panel


def main() -> None:
    source = cached_kblock_source(city="capetown")
    screen = DenseCompactScreen(max_depth_min=6.0)         # deep informal fabric, deepest-first
    ranked = screen.select(source)                          # memoized -> instant on rerun
    geoms = source.block_geometries()                      # block_id + geometry + building_count
    print(f"screen: flagged {len(ranked)} of {len(geoms)} blocks (dense, deep informal)")
    count = dict(zip(geoms["block_id"].astype(str), geoms["building_count"], strict=True))

    seed = next(b for b in ranked if SEED_MIN <= count.get(b, 0) <= SEED_MAX)
    members = DenseClusterRegionBuilder(max_buildings=REGION_MAX).build(geoms, [[seed]])[0]
    print(f"auto-detected region: seed={seed}  members={members}")

    blocks = list(KblockSource(source.blocks_path, source.buildings_path, region_id="clearance",
                               block_ids=members).region().blocks)
    region = region_block(blocks)
    print(f"region: {len(region.parcels)} parcels, {len(region.building_points)} buildings")

    before = access_before(region)
    vmax = int(before.max())
    frame = frame_bbox(region.parcels)
    own_pts = region.building_points

    # Dimmed context = the FULL RAW block set in the render frame (NOT the screened selection),
    # minus the region's own members -- the surrounding formal grid the informal fabric sits in.
    # Mirrors reblock.run's render path (emit.py): block_geometries()/building_points() are the
    # raw source, windowed to `frame`; the screen never enters here.
    outlines = source.block_geometries(frame)
    is_member = outlines["block_id"].astype(str).isin(members)
    context_outlines = outlines[~is_member]
    member_union = outlines[is_member].geometry.union_all()
    ctx_pts = source.building_points(frame)
    context_points = ctx_pts[~ctx_pts.geometry.within(member_union)]

    save_render(
        render_before(region, before, vmax=vmax, own_points=own_pts, frame=frame,
                      context_outlines=context_outlines, context_points=context_points),
        OUT / "before.png")

    print(f"\n{'repulsion':>9} {'roads':>5} {'length_m':>9} {'displaced':>9} {'max_depth':>9}")
    for s in REPULSIONS:
        proposal = ClearanceReblocker(repulsion=s).propose(region)
        roads = proposal.roads
        after = parcel_access_layers(region, roads)
        length = float(sum(g.length for g in roads.geometry)) if len(roads) else 0.0
        displaced = displacement_count(region.building_points, roads, 3.0)
        # buildings displaced (within the 3 m road corridor) -- clearance cuts through parcels, so
        # unlike a frontage method this IS meaningful; mark them red on the road.
        corridor = roads.geometry.buffer(3.0).union_all() if len(roads) else None
        displaced_pts = (region.building_points[region.building_points.geometry.within(corridor)]
                         if corridor is not None else region.building_points.iloc[:0])
        fig = render_after(region, proposal, after, vmax=vmax, own_points=own_pts, frame=frame,
                           context_outlines=context_outlines, context_points=context_points,
                           displaced_points=displaced_pts)
        tag = f"{s:+.0f}".replace("+0", "0")
        save_render(fig, OUT / f"after_s{tag}.png")
        print(f"{s:>9.0f} {len(roads):>5d} {length:>9.0f} {displaced:>9d} {int(after.max()):>9d}")


if __name__ == "__main__":
    main()
