# `dream_come_true_cv` — Phase 2: imagery wide-corridor desire-lines (Design)

**Date:** 2026-07-15
**Status:** approved design (spike-validated), ready for implementation plan
**Depends on:** Phase 1 (`dream_come_true` / `DesireLineSource` / `OSMDesireLines`, merged).

## Goal

A second `DesireLineSource` — `ImageryDesireLines` — that derives desire-lines by **detecting the
wide bare-earth corridors directly from satellite imagery** (Esri World Imagery), landing a genuinely
*imagery-derived* reblocker variant, `dream_come_true_cv`, beside the OSM one. It plugs into the same
seam Phase 1 built, so `DreamComeTrueReblocker` is unchanged.

## Scope (deliberately narrow — set by the spike)

A feasibility spike (2026-07-15, `scratchpad/spike_*.py`) established what is and isn't tractable
without a trained model:
- **Imagery is trivial:** Esri World Imagery (ArcGIS REST tiles, free, no key) gives 0.25 m/px at
  zoom 19 and shows the fabric + paths clearly.
- **The *fine interior* footpath network does NOT detect cheaply** — it's low-contrast greyish worn
  earth, and the building points are too sparse to mask roofs. Recovering it needs sustained CV
  engineering or a trained segmentation model. **OUT OF SCOPE.**
- **The *wide* bare-earth corridors DO detect reliably** (colour + smoothness both catch them). This
  method captures **those**, and is honest about being a *subset* of what OSM maps.

Out of scope: the fine interior network; any deep/trained model; live-imagery detection quality
tuning beyond "recovers the wide corridors on the two flagship regions".

## Architecture — a new source behind the existing seam

```
ImageryDesireLines(DesireLineSource):
    desire_lines(bbox_wgs84, crs) -> GeoDataFrame[LineString]
        # snapshot set  -> load committed detected-lines GeoJSON (offline, byte-stable)
        # else (live)   -> fetch Esri tile mosaic for bbox, detect, vectorize
    identity -> None when live (uncacheable); ("imagery", <params>, snapshot-hash) with a snapshot
```

`DreamComeTrueReblocker` (Phase 1) is reused verbatim — it is source-agnostic. Files:
- `src/reblock/methods/imagery.py` — the mosaic fetch + the detector + `ImageryDesireLines`
  (kept out of `desire_lines.py` so the CV/scikit-image weight is isolated in one module).
- `conf/desire_source/imagery.yaml` — the pluggable source config group.

## Detection pipeline (spike-validated; production-cleaned)

`ImageryDesireLines._detect(mosaic_rgb, transform) -> list[LineString]`:
1. **Mosaic:** fetch Esri World Imagery tiles (zoom 19 default) covering the bbox (+1 tile pad),
   stitch to one RGB array; record the EPSG:3857 affine (pixel ↔ world). stdlib `urllib` + `PIL`.
2. **Bare-earth likelihood** (`[0,1]`): `brightness · smoothness · not-green · not-shadow`, where
   `smoothness = clip(1 - local_std(gray, w≈7px)/σ, 0, 1)` (corrugated roofs are textured, worn
   earth is smooth), `not-green` excludes vegetation (HSV hue/sat), `not-shadow` drops low value.
3. **Wide-corridor mask:** threshold the likelihood, then morphological **opening with a wide disk**
   (radius = `min_corridor_m/2` in px, default ~3 m) so only *wide* bare-earth survives — this is
   what drops the thin fragments and roof snags the spike showed. `skimage.morphology`.
4. **Skeletonize → vectorize:** `skimage.morphology.skeletonize` the mask → 1-px centerlines; build
   an 8-neighbour pixel graph (`networkx`, already a dep), extract junction-to-junction/endpoint
   polylines → `LineString`s in pixel space; map pixel → 3857 → `crs`; `shapely.simplify` to drop
   pixel jitter; drop segments shorter than `min_len_m`.

The `DreamComeTrueReblocker` then clips to the block and dedupes against streets, exactly as for OSM.

## Dependency

Add **`scikit-image`** (`skimage`) — `morphology` (opening/skeletonize) + `measure`. It is the
standard image-processing library; the imagery detector genuinely needs it. (No `torch`/`cv2` — this
is classical CV.) Everything else (`PIL`, `numpy`, `scipy.ndimage`, `networkx`, `shapely`) is present.

## Reproducibility (snapshot = committed detected-lines GeoJSON, parallel to OSM)

`ImageryDesireLines(snapshot=<path>)` loads a committed GeoJSON of **already-detected LineStrings** —
so the examples reproduce offline + byte-stable with no imagery fetch or detection at example time,
exactly like `OSMDesireLines`' snapshot. `scripts/fetch_desire_lines_snapshot.py` gains a mode (or a
sibling) that runs the *live* detection once per flagship region and writes the GeoJSON. The detector
itself is unit-tested on a small committed fixture image, so the CV code has real coverage.

## The rename (fold in here, per the agreed plan)

Now that a sibling exists, rename for the `_{osm,cv}` convention — one migration, no compat shim:
- `all_methods.dream_come_true` (source: osm) → **`all_methods.dream_come_true_osm`**; add
  **`all_methods.dream_come_true_cv`** (source: imagery). **Inline** each variant's source dict
  (rather than `source: ${desire_source}`) so a per-variant `all_methods.dream_come_true_*.source.
  snapshot=<path>` override works directly — this also retires the interpolation quirk that forced
  the awkward `desire_source.snapshot=` override in Phase 1.
- Migrate the reproduce commands, both example READMEs, and any test referring to the bare
  `dream_come_true` key.
- `DreamComeTrueReblocker`/`Proposal.method` stay `"dream_come_true"` (the method *type*); the config
  key is the variant label the compare renders (mirrors clearance vs clearance_grid).

## Testing (all fixture-based; no network, no live imagery in CI)

- **Mosaic math:** lon/lat→tile and tile→3857-extent round-trips (pure, no fetch).
- **Detector on a fixture image:** a small synthetic RGB (a bright smooth "corridor" band through
  textured "roof" noise) → assert the detector returns a LineString tracing the corridor, and that a
  pure-roof fixture returns none. Exercises likelihood + opening + skeletonize + vectorize.
- **Snapshot load:** `snapshot=<fixture.geojson>` → returns those lines, no fetch.
- **`identity`:** None when live; stable + snapshot-hashed with a snapshot.
- **Config conformance:** `dream_come_true_cv` (and renamed `_osm`) instantiate from `compare_config`
  and `method=dream_come_true desire_source=imagery`.

## Example integration (one combined regeneration)

Both flagship examples regenerate once with **both** variants in the comparison: `dream_come_true_osm`
+ `dream_come_true_cv` (each with its committed snapshot), dijkstra/mesh still absent, frontier metric.
Commit the two `desire_lines_cv_*.geojson` snapshots + an `after_dream_come_true_cv.jpg` render.
READMEs frame the honest contrast: the OSM variant has the fuller network; the CV variant captures the
main bare-earth corridors it can see from orbit.

## Risks / open items

- **Detection quality on the region vs block** — the wide corridors detect on both, but coverage/
  vectorization quality is confirmed during snapshot creation (a plan task eyeballs each region's
  detected lines before committing, exactly as the coverage gate did for OSM).
- **Vectorization robustness** — skeleton→polyline on messy masks can spawn spurs; the plan prunes
  short spurs and simplifies. If a region's detection is too noisy to vectorize cleanly, that's
  reported (like OSM's coverage gate), not silently shipped.
- **scikit-image install** — first CV dep; the plan adds it via pixi and confirms the import + the
  full suite before building on it.
