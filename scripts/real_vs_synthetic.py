"""Does anything separate REAL footpath networks from synthesized ones?

The standing intuition is that real networks carry qualities permeability and displacement do not
capture, and there is evidence for it: real OSM networks are the most displacement-efficient thing
measured (0.160 vs clearance 0.205 vs consensus 0.248 for comparable permeability), and IoU@10 m
between a synthesized network and the real one sits at 0.27 -- our networks look nothing like
theirs. Something is being rewarded that we do not score.

That intuition is testable rather than arguable, because the census found 16,497 blocks whose real
network is known. For each block this scores the real network and several synthesized ones on
candidate statistics, then asks which statistic separates them. A statistic that ranks real above
synthetic is a metric candidate; one that does not, is not.

Candidates, each with a reason to exist:

  cycle_ratio        real networks are not trees. Every greedy drainage method here builds a
                     strict tree; loop_closure exists precisely because loops matter. Measured as
                     (edges - nodes + components) / edges on the network's own planar graph.
  mean_leg_m         mean distance from a parcel to its nearest road. Displacement counts homes
                     destroyed; this counts walking imposed on everyone else.
  road_per_parcel_m  metres of road per parcel served -- the parsimony a worn path has and a
                     planned one may not.
  frontage_frac      fraction of parcels with road frontage (within STREET_TOL). A tree can serve
                     a parcel via a neighbour's frontage; real paths tend to front more.
  deadend_frac       fraction of network endpoints that are dead ends. High in a pure drainage
                     tree, low in a network people actually circulate through.
  straightness       mean (chord / path length) over the network's own segments -- do paths run
                     direct, or wander around parcel corners?

Reported as a paired comparison against the block's own real network, since blocks differ wildly
and only the within-block contrast is meaningful.

    pixi run python -m scripts.real_vs_synthetic --recipients 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.geometric_access import geometric_access_distances
from reblock.derive.network_metrics import meshedness, node_network
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.loop_closure import LoopClosureRefiner
from scripts.consensus_sweep import displacement_matched_prefix
from scripts.pair_matrix import (
    desire_source,
    displacement_fraction,
    evenly_spaced,
    fetch_donor_lines,
    iso_of,
    load_pools,
)

SNAP_TOL = 0.5      # metres; endpoints closer than this are the same node


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of the walk-leg distribution: 0 = every parcel walks the same distance,
    1 = the whole burden falls on one parcel. The distributional statement a mean cannot make."""
    v = np.sort(np.asarray(x, dtype=float))
    n = len(v)
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * total))


def _noded(roads: gpd.GeoDataFrame) -> list[LineString]:
    """Segments split at every intersection, so crossings become shared nodes.

    Load-bearing, and the first version of this file got it wrong: keying the graph on raw segment
    ENDPOINTS found no shared nodes at all in real OSM networks (every endpoint degree 1, so
    cycle_ratio read 0.000 for every network including the real ones -- the statistic measured
    nothing). Real footpaths cross mid-segment and are not drawn to share endpoints. `unary_union`
    nodes the collection at intersections first.
    """
    if roads.empty:
        return []
    merged = unary_union(list(roads.geometry))
    geoms = getattr(merged, "geoms", [merged])
    return [g for g in geoms if isinstance(g, LineString) and g.length > 0]


def _nodes_edges(roads: gpd.GeoDataFrame) -> tuple[int, int, int]:
    """(nodes, edges, components) of the NODED network graph, endpoints snapped at SNAP_TOL."""
    segs = _noded(roads)
    if not segs:
        return (0, 0, 0)
    key: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    for geom in segs:
        coords = list(geom.coords)
        ids = []
        for x, y in (coords[0], coords[-1]):
            k = (int(round(x / SNAP_TOL)), int(round(y / SNAP_TOL)))
            ids.append(key.setdefault(k, len(key)))
        if ids[0] != ids[1]:
            edges.append((ids[0], ids[1]))
    parent = list(range(len(key)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = len({find(i) for i in range(len(key))}) if key else 0
    return (len(key), len(edges), comps)


def metrics_for(block: Block, roads: gpd.GeoDataFrame) -> dict[str, float]:
    if roads.empty:
        return {k: float("nan") for k in
                ("cycle_ratio", "meshedness", "walk_mean_m", "walk_p95_m", "walk_max_m",
                 "walk_gini", "mean_leg_m", "road_per_parcel_m", "frontage_frac",
                 "deadend_frac", "straightness", "road_len_m", "displacement")}
    n_nodes, n_edges, comps = _nodes_edges(roads)
    cycles = max(n_edges - n_nodes + comps, 0)

    # The repo's own primitive, not a crow-flies distance: geometric_access_distances walks the
    # parcel-adjacency graph to the nearest road or street, so a parcel behind three others pays
    # for crossing them. StructureEval already exposes its p95 as "B equity" -- this measurement
    # exists to ask whether that retired metric is the one that separates real from synthetic.
    legs = geometric_access_distances(block, roads).to_numpy().astype(float)

    degree: dict[tuple[int, int], int] = {}
    for geom in _noded(roads):
        cs = list(geom.coords)
        for x, y in (cs[0], cs[-1]):
            k = (int(round(x / SNAP_TOL)), int(round(y / SNAP_TOL)))
            degree[k] = degree.get(k, 0) + 1
    deadends = sum(1 for v in degree.values() if v == 1)

    straight = [g.length and Point(g.coords[0]).distance(Point(g.coords[-1])) / g.length
                for g in roads.geometry]
    total_len = float(roads.geometry.length.sum())
    graph = node_network(roads, block.streets)
    return {
        "cycle_ratio": cycles / n_edges if n_edges else 0.0,
        "meshedness": float(meshedness(graph)),
        "walk_mean_m": float(legs.mean()),
        "walk_p95_m": float(np.quantile(legs, 0.95)),
        "walk_max_m": float(legs.max()),
        # Gini over per-parcel walk legs: 0 = everyone walks the same, 1 = all burden on a few.
        # The distributional reading the means cannot give.
        "walk_gini": _gini(legs),
        "mean_leg_m": float(legs.mean()),
        "road_per_parcel_m": total_len / max(len(block.parcels), 1),
        "frontage_frac": float((legs <= STREET_TOL).mean()),
        "deadend_frac": deadends / max(len(degree), 1),
        "straightness": float(np.mean([s for s in straight if s])) if straight else 0.0,
        "road_len_m": total_len,
        "displacement": displacement_fraction(block, roads),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipients", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("scratchpad/ot/real_vs_synthetic.parquet"))
    args = ap.parse_args()

    pools = load_pools()
    blocks = pools.blocks
    source = desire_source("pbf", iso_of(blocks))
    usable = sorted(set(pools.recipients) & set(pools.donors))
    counts = [float(len(b.parcels)) for b in blocks]
    chosen = evenly_spaced(usable, counts, args.recipients)
    print(f"  scoring {len(chosen)} blocks with a known real network", flush=True)

    rows: list[dict[str, object]] = []
    for n, i in enumerate(chosen, 1):
        block = blocks[i]
        status, own = fetch_donor_lines(source, block)
        if status != "ok" or own is None:
            continue
        # Matched on DISPLACEMENT, not length. A length prefix cuts loop_closure's connectors --
        # which are appended AFTER its base tree -- off entirely, so looped_tree and clearance
        # came out identical on every statistic in the first run. Displacement is also the fairer
        # budget: it is the cost that is actually paid in homes.
        target_disp = displacement_fraction(block, own)

        def matched(roads: gpd.GeoDataFrame, blk: Block = block,
                    t: float = target_disp) -> gpd.GeoDataFrame:
            return displacement_matched_prefix(blk, roads, t)

        nets = {"real": own}
        nets["clearance"] = matched(ClearanceReblocker(depth_target=1).propose(block).roads)
        nets["looped_tree"] = matched(
            LoopClosureRefiner(base=ClearanceReblocker(depth_target=1)).propose(block).roads)
        nets["demand_greedy"] = matched(
            DemandGreedyReblocker(desire_source=source, depth_target=1).propose(block).roads)
        for name, roads in nets.items():
            rows.append({"block": block.block_id, "network": name, **metrics_for(block, roads)})
        print(f"  [{n}/{len(chosen)}] {block.block_id}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out)

    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows)\n")
    if df.empty:
        return
    stat_cols = ["walk_mean_m", "walk_p95_m", "walk_max_m", "walk_gini", "cycle_ratio",
                 "meshedness", "road_per_parcel_m", "frontage_frac", "displacement"]
    real = df[df.network == "real"].set_index("block")
    print(f"{'statistic':>18} {'real':>8} | " +
          " | ".join(f"{n:>22}" for n in ("clearance", "looped_tree", "demand_greedy")))
    for c in stat_cols:
        cells = []
        for name in ("clearance", "looped_tree", "demand_greedy"):
            syn = df[df.network == name].set_index("block")
            common = real.index.intersection(syn.index)
            a, b = real.loc[common, c].astype(float), syn.loc[common, c].astype(float)
            ok = a.notna() & b.notna()
            if ok.sum() < 3:
                cells.append(f"{'--':>22}")
                continue
            p = stats.wilcoxon(a[ok], b[ok]).pvalue
            cells.append(f"{b[ok].median():>10.3f} p={p:>7.4f}")
        print(f"{c:>18} {real[c].astype(float).median():>8.3f} | " + " | ".join(cells))
    print("\n(p is a paired Wilcoxon of real vs that method, same blocks; small p = separates)")


if __name__ == "__main__":
    main()
