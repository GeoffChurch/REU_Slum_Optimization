"""Holdout support for the footpath-prediction eval.

Leave-one-BLOCK-out leaks: donors are immediate neighbours, frequently the same continuous OSM way
clipped at a block edge, often one mapper in one session -- a live explanation for a high score
that requires no generalization at all. Leave-one-SETTLEMENT-out is the obvious fix but is not
well-defined: measured on Cape Town's 1,136 qualified blocks, a 100 m threshold yields 417
components with 23% singletons and a 150-block component spanning 5.7 km (Gugulethu inside it;
Nyanga, Langa, Delft each alone). Transitive chaining has no natural stopping point, and there is
no free label to fall back on -- `gadm_code` is the block_id prefix and `urban_id` is metro-scale.

So `exclusion_holdout` (a hard metric radius) is the fold definition, and `settlement_labels` is a
stratification/reporting label whose threshold must be stated wherever it appears.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely import STRtree


def settlement_labels(blocks: gpd.GeoDataFrame, *, tol_m: float = 100.0) -> list[int]:
    """Connected-component label per block under `tol_m` boundary proximity.

    REPORTING ONLY -- not a fold definition. Chains transitively, so the label depends on the
    whole corpus and on `tol_m`; always state the threshold alongside any number stratified by it.
    """
    geoms = list(blocks.geometry)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    if len(geoms) > 1:
        tree = STRtree(geoms)
        left, right = tree.query(geoms, predicate="dwithin", distance=tol_m)
        graph.add_edges_from((i, j) for i, j in zip(left.tolist(), right.tolist(), strict=True)
                             if i != j)
    labels = [0] * len(geoms)
    for label, component in enumerate(nx.connected_components(graph)):
        for node in component:
            labels[node] = label
    return labels


def exclusion_holdout(
    blocks: gpd.GeoDataFrame, recipient_idx: int, *, radius_m: float
) -> list[int]:
    """Indices eligible as donors for `recipient_idx`: everything strictly beyond `radius_m`.

    Monotone in `radius_m`, no chaining, one interpretable number, sweepable -- which is why this
    and not a component label is the primary fold definition. The recipient is always excluded.
    """
    recipient = blocks.geometry.iloc[recipient_idx]
    distances = blocks.geometry.distance(recipient)
    return [i for i in range(len(blocks))
            if i != recipient_idx and float(distances.iloc[i]) > radius_m]
