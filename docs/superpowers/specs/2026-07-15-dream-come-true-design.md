# `dream_come_true` — Phase 1: OSM desire-line baseline (Design)

**Date:** 2026-07-15
**Status:** approved design, ready for implementation plan
**Scope:** Phase 1 only — the OSM (vector) baseline. The satellite-imagery detector is a **separate
later spec** (Phase 2), which slots into the same seam this spec defines.

## Goal

A reblocker method, `dream_come_true`, whose "proposed roads" are the **real informal circulation
network people already walk** — pulled from OpenStreetMap for the region — instead of a synthesized
one. It implements the existing `Method` protocol and is graded by the existing compare/eval suite
unchanged, so it answers: *how do the actual worn paths score against dijkstra / clearance /
arterial on the four lenses + displacement?*

Why OSM first (owner decision, 2026-07-15, "OSM baseline first, then imagery"): HOT (Humanitarian
OSM) has already digitized informal footpaths from imagery. A live coverage probe over block 5810's
flagship region found **185 `path/footway/track/steps` ways + 343 `service/residential/…` ways** — a
rich, ready-made desire-line network. This delivers a working method with no ML and full offline
reproducibility, and establishes the seam the imagery phase will reuse.

## Where it fits

- `Method.propose(block, prior=None) -> Proposal` — a method receives a `Block` (with `boundary`,
  `crs`, `streets`, `parcels`, `building_points`) and returns a `Proposal(roads=GeoDataFrame)`.
- The region bbox is derivable from `block.boundary` reprojected to EPSG:4326 (the same
  `to_crs(4326)` path `render.google_maps_url` already uses), so no new plumbing is needed to know
  *where* to fetch.
- No imagery/raster/CV/ML/HTTP libraries exist in the stack today (geopandas, shapely, scipy,
  numpy, networkx). Phase 1 adds **no new dependency**: Overpass is reached with stdlib `urllib`,
  parsed with stdlib `json`, and geometry built with shapely (already present).

## Architecture — one pluggable seam so Phase 2 slots in

Mirrors the existing pluggable-substrate pattern (clearance's `substrate` via `conf/substrate/`).

```
DesireLineSource (Protocol):
    desire_lines(bbox_wgs84: tuple[float,float,float,float], crs: CRS) -> GeoDataFrame[LineString]
        # returns desire-line geometries reprojected into `crs` (the block's projected CRS)

    ├─ OSMDesireLines(...)          ← Phase 1 (this spec)
    └─ ImageryDesireLines(...)      ← Phase 2 (later spec), same signature

DreamComeTrueReblocker(source: DesireLineSource, corridor_m: float = 3.0)
    propose(block) -> Proposal      # the Method; source-agnostic block integration
```

The **source** owns "get desire-lines for this bbox in this CRS"; the **method** owns block-specific
integration (clip, dedupe against existing streets). Phase 2 replaces only the source.

Files:
- `src/reblock/methods/desire_lines.py` — the `DesireLineSource` Protocol + `OSMDesireLines`.
- `src/reblock/methods/dream_come_true.py` — `DreamComeTrueReblocker`.
- `conf/desire_source/osm.yaml` — the pluggable source config group (like `conf/substrate/`).
- `conf/method/dream_come_true.yaml` — single-method run config (`method=dream_come_true`).
- `conf/compare_config.yaml` — an `all_methods.dream_come_true` entry.

## The Phase-1 pipeline

**`OSMDesireLines.desire_lines(bbox_wgs84, crs)`:**
1. Build the Overpass QL query: `way["highway"~"<tags>"](south,west,north,east); out geom;` where
   `<tags>` is the `|`-joined tag list and the bbox is `(miny,minx,maxy,maxx)` in lat/lon.
2. **Cache check** — key = hash of (rounded bbox, sorted tag list). If
   `~/.cache/reblock/osm/<key>.geojson` exists, load it (offline). Else fetch and write it. Mirrors
   the parquet-dataset auto-download-on-first-use pattern.
3. **Fetch** via stdlib `urllib.request` with a real `User-Agent` header (default curl UA gets a 406
   from Overpass — verified). Timeout + a single retry against a fallback endpoint.
4. **Parse** `out geom` JSON: each `way` carries `geometry: [{lat,lon}, …]` → a shapely `LineString`
   of `(lon, lat)` points (EPSG:4326 x=lon, y=lat). Drop ways with < 2 nodes.
5. Return a `GeoDataFrame` in 4326, **reprojected to `crs`**.

**`DreamComeTrueReblocker.propose(block)`:**
1. `bbox = GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds`.
2. `lines = self.source.desire_lines(bbox, block.crs)`.
3. **Clip** to `block.boundary` (intersection); explode to single LineStrings; drop empties.
4. **Dedupe against `block.streets`** — the perimeter/inter-block network is already egress, not part
   of the intervention (the same "added roads vs existing network" convention `compare` uses). Drop
   any desire-line segment that runs within a small tolerance of an existing street.
5. Return `Proposal(block_id, crs=block.crs, roads=<interior desire-lines>, method="dream_come_true",
   params={"corridor_m": ...})`. Empty roads (no coverage) is a valid, non-crashing result.

`identity` (for the derivation cache): `("dream_come_true", source.identity, corridor_m)` where
`OSMDesireLines.identity = ("osm", tuple(sorted(tags)), snapshot_key_or_None)`.

## Configuration

`conf/desire_source/osm.yaml`:
```yaml
_target_: reblock.methods.desire_lines.OSMDesireLines
tags: [path, footway, track, steps, pedestrian, living_street]   # the worn desire-paths (default)
endpoint: https://overpass-api.de/api/interpreter
cache_dir: null          # null -> ~/.cache/reblock/osm
snapshot: null           # a committed GeoJSON path -> load it directly, skip Overpass (see below)
```
- **Default tags** = the six worn-path classes (owner-approved). `service, residential, unclassified`
  are addable via config for a fuller circulation network but are **not** in the default.
- `all_methods.dream_come_true`: `{_target_: …DreamComeTrueReblocker, source: "${desire_source}",
  corridor_m: 3.0}` with `desire_source: osm` in the compare defaults list.

## Reproducibility (owner-approved: snapshot the committed example, cache-and-date otherwise)

OSM drifts, so a live method is not byte-reproducible. Two mechanisms:
- **Arbitrary regions:** fetch → cache to `~/.cache/reblock/osm` → the run log records the fetch;
  README notes "OSM as of `<date>`". First run needs network (like the datasets); offline after.
- **Committed flagship example:** a `snapshot` path on `OSMDesireLines`. When set and present, the
  source **loads the committed GeoJSON directly** (no Overpass, no cache) so the example reproduces
  byte-for-byte forever. We commit `examples/dream-come-true/desire_lines_<region>.geojson` (a small
  file — a few hundred LineStrings) and point the example's method config at it via `snapshot=`.

## Testing

- **`OSMDesireLines` parse** (no network): feed a captured Overpass JSON fixture → assert the right
  number of LineStrings, correct coordinates, `< 2`-node ways dropped, reprojected to the target CRS.
- **Cache hit** (no network): pre-write a cache file → assert `desire_lines` returns it without
  fetching (patch/guard the fetch to raise if called).
- **Snapshot load** (no network): `snapshot=<fixture.geojson>` → returns those lines, no fetch.
- **`DreamComeTrueReblocker.propose`** on a synthetic block + a stub `DesireLineSource`: clip to
  boundary, dedupe against streets, empty-coverage → empty roads (no crash), `identity` stable.
- **Method-protocol conformance**: instantiates from config, `propose` returns a valid `Proposal`.
- A **live smoke test** is out of scope for CI (network) — coverage is exercised via fixtures.

## Integration with the example comparisons

- Add `dream_come_true` to the **multiblock** flagship comparison (region 5810) — where OSM coverage
  is rich (185 paths) — with a committed desire-line snapshot, and report where the *real* paths land
  on the four lenses + displacement. It gets a registry color automatically.
- For **method-comparison** (single block 40972): include it **iff** the block has adequate OSM
  desire-path coverage (a plan task verifies this first). If coverage is thin at single-block scale,
  feature `dream_come_true` in multiblock only and say so — no silent empty curve.
- This triggers one example regeneration; fold it into the plan's final task.

## Out of scope (Phase 2, later spec)

- The satellite-imagery `ImageryDesireLines` source: imagery acquisition + a road/path detector.
  Phase 2 is where the CV/ML lives; it reuses `DesireLineSource` unchanged.

## Risks / open items

- **Overpass availability/rate-limits** for the first fetch of an arbitrary region — mitigated by the
  cache + snapshot; the committed example never hits the network.
- **Single-block OSM coverage** (block 40972) unknown — resolved by a plan task before wiring it into
  method-comparison.
- **Street dedupe tolerance** — needs tuning so genuine interior paths aren't dropped as "existing
  street"; the plan will pick a tolerance consistent with `network_metrics.STREET_TOL`.
