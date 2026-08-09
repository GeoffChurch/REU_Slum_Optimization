"""Ground truth: where the informal settlements actually are.

Every screen metric this project ships is a GUESS at informality from block geometry. Until
2026-08-08 nothing measured how good those guesses are, and the answer turned out to matter: the
then-default `density_compactness` floor selects 1,644 Cape Town blocks of which only **24.5%** are
really informal settlement.

The source is the City of Cape Town's own survey -- **117,336 informal dwelling polygons** digitised
from February 2018 aerial photography at 1:200, published via University of Edinburgh DataShare
([doi:10.7488/ds/2758](https://doi.org/10.7488/ds/2758)). Median structure area 29.5 m², which is
shack-scale and independently reproduced by Open Buildings on the same blocks (28.9 m²).

## Cape Town only, and that is not an oversight

No equivalent layer was found for Nairobi. Searched: general web, the City of Cape Town's ArcGIS
open-data portal (which has no informal-settlement boundary layer of its own -- only Informal
*Trading*), openAFRICA (Cloudflare-blocked), HDX (2,627 hits, all tabular indicators), and OSM
Overpass for `informal=yes` / `residential=informal` / named settlement polygons over the Nairobi
bbox (27 elements, 12 of them nodes; the major settlements are POINTS). See
`notes/2026-08-08-c17-nairobi-has-no-published-settlement-layer-but-the-floor-transfers.md`.

## Extents are derived, because the file has no settlement field

`FID_1` puts 115,327 of 117,336 rows in a single group, so it is not a grouping. Settlement extents
are therefore clustered from the structures themselves: DBSCAN over structure centroids at
`EPS_M`, then a `BUFFER_M` buffered union per cluster, keeping clusters of at least
`MIN_STRUCTURES`. On the shipped parameters that yields 189 settlements covering 15.4 km² and
retaining 97.9% of all structures.

The clustering is DBSCAN, implemented on scipy rather than imported from scikit-learn, which is not
a declared dependency of this project. Core points (>= `MIN_SAMPLES` neighbours within `EPS_M`) are
connected-componented over core-to-core edges; non-core points are dropped rather than assigned to a
border's nearest core. Verified against scikit-learn on the shipped parameters: identical settlement
count (189) and area (15.4 km²), with structure retention differing by 12 of 114,909 (0.01%) from
the border tie-break described in `_dbscan`.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.ops import unary_union

DEFAULT_CACHE = Path.home() / ".cache" / "reblock"

# University of Edinburgh DataShare, doi:10.7488/ds/2758 -- "Dwelling outline - Informal Settlements
# of Cape Town", CoCT_IS_STRUCTURES_201802.zip (17.4 MB).
_URL = "https://datashare.ed.ac.uk/bitstreams/ef718096-b0ad-40e9-bc6c-5585a0a64ab6/download"
_SHP = "CoCT_IS_STRUCTURES_201802.shp"

EPS_M = 30.0            # neighbourhood radius: two shacks within 30 m are the same settlement
MIN_SAMPLES = 10
BUFFER_M = 20.0         # buffered union per cluster -> a settlement polygon, not a point cloud
MIN_STRUCTURES = 20     # below this a cluster is a hamlet or a digitising stray, not a settlement
_KNOWN = {"capetown"}


def ensure_informal_structures(city: str = "capetown",
                               *, cache_dir: Path = DEFAULT_CACHE) -> Path:
    """Path to the informal-structure shapefile, downloading and extracting once if needed."""
    if city not in _KNOWN:
        raise ValueError(
            f"no informal-settlement ground truth for {city!r}; known: {sorted(_KNOWN)}. "
            "No published boundary layer was found for Nairobi -- see this module's docstring.")
    out_dir = cache_dir / "coct_is"
    shp = out_dir / _SHP
    if shp.exists():
        return shp
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "CoCT_IS_STRUCTURES_201802.zip"
    if not zip_path.exists():
        with urllib.request.urlopen(_URL, timeout=600) as r, zip_path.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    if not shp.exists():
        raise FileNotFoundError(f"{_SHP} missing from {zip_path}")
    return shp


def settlement_extents(city: str = "capetown", *, epsg: int = 32734,
                       cache_dir: Path = DEFAULT_CACHE) -> gpd.GeoDataFrame:
    """Settlement polygons clustered from the structure survey, with a `n_structures` column.

    Cached as a parquet keyed on the clustering parameters, so changing any of them produces a new
    file rather than silently reusing extents built under different settings.
    """
    key = f"{city}_eps{EPS_M:g}_ms{MIN_SAMPLES}_buf{BUFFER_M:g}_min{MIN_STRUCTURES}"
    cached = cache_dir / f"informal_extents_{key}.parquet"
    if cached.exists():
        return gpd.read_parquet(cached)

    g = gpd.read_file(ensure_informal_structures(city, cache_dir=cache_dir)).to_crs(epsg)
    cent = g.geometry.centroid
    xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    lab = _dbscan(xy)
    geoms = np.asarray(list(g.geometry), dtype=object)
    polys, counts = [], []
    for k in range(int(lab.max()) + 1):
        sel = geoms[lab == k]
        if len(sel) < MIN_STRUCTURES:
            continue
        polys.append(unary_union([s.buffer(BUFFER_M) for s in sel]))
        counts.append(int(len(sel)))
    out = gpd.GeoDataFrame({"n_structures": counts}, geometry=polys, crs=epsg)
    out.to_parquet(cached)
    return out


def _dbscan(xy: NDArray[np.float64]) -> NDArray[np.int64]:
    """DBSCAN(eps=EPS_M, min_samples=MIN_SAMPLES) on scipy: -1 for noise, else a cluster index.

    Core points are those with at least `MIN_SAMPLES` neighbours within `EPS_M` (self included, as
    scikit-learn counts it). Clusters are the connected components of the core-to-core neighbour
    graph, and each border point -- non-core but within `EPS_M` of a core -- joins its NEAREST core
    point's cluster.

    Border assignment is load-bearing, which is worth stating because a first version skipped it on
    the reasoning that `MIN_STRUCTURES` would discard anything affected. That was wrong: border
    points push clusters OVER the threshold, and dropping them cost 21 of 189 settlements
    (114,909 -> 112,250 structures). The core clustering was identical either way -- 244 components,
    100% partition agreement with scikit-learn on points both call non-noise -- so the fringe was
    the entire difference.

    Nearest-core assignment is deterministic; scikit-learn attaches a border point to whichever core
    its scan reaches first, which is arbitrary but equally valid DBSCAN. Both satisfy the definition.
    """
    tree = cKDTree(xy)
    pairs = np.asarray(list(tree.query_pairs(EPS_M)), dtype=np.int64)
    n = len(xy)
    deg = np.ones(n, dtype=np.int64)                     # self-count, matching scikit-learn
    if len(pairs):
        np.add.at(deg, pairs[:, 0], 1)
        np.add.at(deg, pairs[:, 1], 1)
    core = deg >= MIN_SAMPLES
    lab = np.full(n, -1, dtype=np.int64)
    if not core.any():
        return lab
    keep = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]] if len(pairs) else pairs
    adj = coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
    _ncomp, comp = connected_components(adj, directed=False)
    comp_of_core = np.unique(comp[core])
    remap = {int(c): i for i, c in enumerate(comp_of_core)}
    lab[core] = [remap[int(c)] for c in comp[core]]

    # border points: non-core, but within EPS_M of a core -> join that core's cluster
    border = ~core
    if border.any():
        core_idx = np.flatnonzero(core)
        d, j = cKDTree(xy[core]).query(xy[border], distance_upper_bound=EPS_M)
        hit = np.isfinite(d)
        assigned = np.flatnonzero(border)[hit]
        lab[assigned] = lab[core_idx[j[hit]]]
    return lab


def label_blocks(blocks: gpd.GeoDataFrame, extents: gpd.GeoDataFrame,
                 *, cover_frac: float = 0.30,
                 ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Per-block share of area inside a settlement extent, and the boolean label at `cover_frac`.

    Returns `(cover, label)`. `cover_frac` is 0.30 because that is what C13 measured against; the
    metric ORDERING was verified stable for every threshold from 0.10 to 0.90, so the exact value is
    not load-bearing (`notes/2026-08-08-c13-...`).
    """
    from shapely import STRtree

    tree = STRtree(list(extents.geometry))
    cover = np.zeros(len(blocks))
    for i, geom in enumerate(blocks.geometry):
        hit = tree.query(geom, predicate="intersects")
        if not len(hit) or geom.area <= 0:
            continue
        inter = unary_union([extents.geometry.iloc[j] for j in hit]).intersection(geom)
        cover[i] = inter.area / geom.area
    return cover, cover >= cover_frac
