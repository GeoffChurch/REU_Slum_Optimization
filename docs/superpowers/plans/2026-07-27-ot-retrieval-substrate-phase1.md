# OT Retrieval Substrate Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared corpus substrate that makes the OT/transplant direction measurable at scale — a country-wide OSM footpath census, targeted building-point provisioning, geometric agreement primitives, and a reusable GW pair-matrix benchmark.

**Architecture:** Three data tiers with different requirements. T1 (the census) needs only the country blocks parquet plus OSM linework — no `Block`, no Voronoi, no building points — which is unlocked by refactoring one function to a pure-geometry signature. T2 (retrieval) needs building points, provisioned only for blocks the census says matter. T3 (scoring) is the expensive tier and stays small in Phase 1. Nothing here builds an index or a reblocker; Phase 1 produces measurements and reusable artifacts.

**Tech Stack:** Python 3.11, geopandas/shapely/pyogrio (GDAL `OSM` driver, already present), pandas/pyarrow, networkx, scipy. Hydra configs under `conf/`. pytest + mypy --strict + ruff. Package manager is pixi.

**Spec:** `docs/superpowers/specs/2026-07-27-ot-retrieval-substrate-phase1-design.md`

## Global Constraints

- **No backward-compatibility shims.** When an interface changes, migrate every call site and delete the old path. No aliases, no dual-path code, no deprecated-but-supported parameters. (Owner directive.)
- **Line length 100.** ruff `select = ["E", "F", "I", "UP", "B"]`, `target-version = "py311"`.
- **mypy `--strict`** covers `src`, `tests`, and two named scripts. New `src/` and `tests/` files must pass strict.
- **Run everything through pixi:** `pixi run test`, `pixi run typecheck`, `pixi run lint`, `pixi run check`. Invoking `.pixi/envs/default/bin/python` directly prints a spurious PROJ error.
- **Interiority tolerances are `0.5 / 2 / 5` m.** `STREET_TOL = 0.5` (`src/reblock/derive/access.py:30`) is the default, not the only value.
- **Footpath tags are `("path", "footway", "track", "steps", "pedestrian", "living_street")`** — one shared definition, never re-declared.
- **Near-miss tags are `("service", "residential", "unclassified")`** — counted separately, never mixed into primary counts.
- **Open Buildings confidence floor is `OB_MIN_CONFIDENCE = 0.7`.**
- **Corpus is ZAF + KEN.** `~/.cache/reblock/{ZAF,KEN}_geodata.parquet`, 1,813,575 blocks total, single-row-group (must be read with `iter_batches`).
- **No silent truncation.** Any step that drops rows, tiles, or blocks reports the count and the reason.

---

## File Structure

| path | status | responsibility |
|---|---|---|
| `src/reblock/methods/osm_footpaths.py` | modify | `interior_desire_lines` becomes pure-geometry; the reblocker calls it with block fields |
| `src/reblock/data/osm_extract.py` | create | `PbfDesireLines` + `interiority_row` + `census_rows`; the T1 engine |
| `src/reblock/data/settlements.py` | create | `settlement_labels` (stratification) + `exclusion_holdout` (the fold definition) |
| `src/reblock/data/provision.py` | modify | multi-tile Open Buildings enumeration + per-polygon filter |
| `src/reblock/eval/agreement.py` | create | `buffered_iou` + `directional_chamfer`; plain functions, not an `Eval` |
| `scripts/osm_census.py` | create | census driver: UTM batching, area guard, streaming, parquet out |
| `conf/desire_source/_footpath_tags.yaml` | create | the single shared tag list |
| `conf/desire_source/osm.yaml` | modify | interpolate the shared list instead of re-declaring |
| `conf/desire_source/pbf.yaml` | create | `PbfDesireLines` config |
| `pyproject.toml` | modify | register the `network` pytest marker |
| `tests/methods/test_osm_footpaths.py` | modify | cover the pure-geometry signature |
| `tests/data/test_osm_extract.py` | create | interiority sweep, near-miss, PbfDesireLines |
| `tests/data/test_settlements.py` | create | clustering + holdout |
| `tests/data/test_provision.py` | create | tile enumeration + polygon filter |
| `tests/eval/test_agreement.py` | create | agreement primitives |

Task 9 (the pair matrix) is a scratchpad experiment whose deliverable is a committed parquet, not `src/` code.

---

### Task 1: Make `_interior_desire_lines` pure-geometry

This is the change that unlocks T1: the census must compute interiority for 1.8M blocks that have no building points and therefore cannot be constructed as `Block`s (`Block.__post_init__` raises `ValueError("Block.parcels must be non-empty")`, `src/reblock/contracts.py:54-56`).

**Files:**
- Modify: `src/reblock/methods/osm_footpaths.py:23-41` (the function) and `:64` (the call site)
- Test: `tests/methods/test_osm_footpaths.py`

**Interfaces:**
- Consumes: nothing
- Produces: `interior_desire_lines(lines: GeoDataFrame, boundary: BaseGeometry, streets: BaseGeometry, crs: CRS, *, tol: float = STREET_TOL) -> GeoDataFrame` — public, importable from `reblock.methods.osm_footpaths`. Returns interior LineStrings longer than `tol`.

- [ ] **Step 1: Write the failing test**

Add to `tests/methods/test_osm_footpaths.py`:

```python
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.methods.osm_footpaths import interior_desire_lines


def test_interior_desire_lines_needs_no_block() -> None:
    """The census path: boundary + streets + crs only, no Block, no parcels."""
    crs = CRS.from_epsg(32734)
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(10, 50), (90, 50)]),   # interior: kept
            LineString([(0, 0), (100, 0)]),     # on the boundary: dropped
        ],
        crs=crs,
    )
    out = interior_desire_lines(lines, boundary, boundary.boundary, crs)
    assert len(out) == 1
    assert out.geometry.iloc[0].length == pytest.approx(80.0)


def test_interior_desire_lines_tolerance_trims_length_not_count() -> None:
    """A path running just inside the boundary survives 0.5 m and dies at 5 m."""
    crs = CRS.from_epsg(32734)
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    lines = gpd.GeoDataFrame(geometry=[LineString([(10, 2), (90, 2)])], crs=crs)
    assert len(interior_desire_lines(lines, boundary, boundary.boundary, crs, tol=0.5)) == 1
    assert len(interior_desire_lines(lines, boundary, boundary.boundary, crs, tol=5.0)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/methods/test_osm_footpaths.py -k interior_desire_lines -v`
Expected: FAIL with `ImportError: cannot import name 'interior_desire_lines'`

- [ ] **Step 3: Replace the function**

In `src/reblock/methods/osm_footpaths.py`, replace `_interior_desire_lines` entirely (do not keep the old name — see Global Constraints):

```python
def interior_desire_lines(
    lines: gpd.GeoDataFrame,
    boundary: BaseGeometry,
    streets: BaseGeometry,
    crs: CRS,
    *,
    tol: float = STREET_TOL,
) -> gpd.GeoDataFrame:
    """Clip `lines` to `boundary`, subtract the `streets` corridor (a `tol` buffer), and keep the
    interior LineString remainder longer than `tol` -- the added intervention, excluding the
    perimeter/inter-block streets that are already egress.

    Pure geometry: takes boundary/streets/crs rather than a Block, so the country-wide OSM census
    can call it for blocks that have no building points (and therefore no Voronoi parcels, and
    therefore cannot be constructed as a Block at all). `tol` is exposed because the census sweeps
    it -- OSM ways are digitized against different imagery than the kblock outlines, so a
    boundary-running path more than `tol` off the outline reads as interior.
    """
    empty = gpd.GeoDataFrame(geometry=[], crs=crs)
    if lines.empty:
        return empty
    clipped = lines.clip(boundary)
    if clipped.empty:
        return empty
    remainder = clipped.geometry.difference(streets.buffer(tol)).explode(index_parts=False)
    mask = ((~remainder.is_empty)
            & (remainder.geom_type == "LineString")
            & (remainder.length > tol))
    # geopandas-stubs' GeoSeries.__getitem__ resolves boolean-mask indexing to the
    # scalar-return overload (-> BaseGeometry) instead of the array-return one; cast to
    # correct it, mirroring the same fixup in reblock.data.shapefile._prepared.
    kept = cast(gpd.GeoSeries, remainder[mask])
    return gpd.GeoDataFrame(geometry=list(kept), crs=crs)
```

Add `from pyproj import CRS` and `from shapely.geometry.base import BaseGeometry` to the imports.

- [ ] **Step 4: Update the one call site**

In `OsmFootpathsReblocker.propose`, replace `roads = _interior_desire_lines(lines, block)` with:

```python
        roads = interior_desire_lines(
            lines, block.boundary, unary_union(list(block.streets.geometry)), block.crs)
```

- [ ] **Step 5: Run the full suite to verify nothing regressed**

Run: `pixi run test`
Expected: PASS. The `osm_footpaths` behaviour is unchanged — same clip, same corridor subtraction, same filter — so every existing test must still pass. If any fails, the refactor changed semantics; fix it rather than updating the assertion.

- [ ] **Step 6: Typecheck and lint**

Run: `pixi run typecheck && pixi run lint`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/methods/osm_footpaths.py tests/methods/test_osm_footpaths.py
git commit -m "refactor: interior_desire_lines takes geometry, not a Block

Unlocks the country-wide OSM census: 1.81M ZAF+KEN blocks have no building
points, so no Voronoi parcels, so Block.__post_init__ rejects them. Interiority
never needed a Block -- only boundary, streets and crs. Exposes tol because the
census sweeps 0.5/2/5m."
```

---

### Task 2: Interiority row with tolerance sweep and near-miss tags

**Files:**
- Create: `src/reblock/data/osm_extract.py`
- Test: `tests/data/test_osm_extract.py`

**Interfaces:**
- Consumes: `interior_desire_lines` (Task 1)
- Produces:
  - `FOOTPATH_TAGS: tuple[str, ...]` and `NEAR_MISS_TAGS: tuple[str, ...]`
  - `TOLERANCES: tuple[float, ...] = (0.5, 2.0, 5.0)`
  - `interiority_row(block_id: str, boundary: BaseGeometry, footpaths: GeoDataFrame, near_miss: GeoDataFrame, crs: CRS) -> dict[str, object]` — one census row

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_osm_extract.py`:

```python
import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.data.osm_extract import TOLERANCES, interiority_row

CRS_M = CRS.from_epsg(32734)
BOUNDARY = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def _lines(*geoms: LineString) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geoms), crs=CRS_M)


def test_interiority_row_reports_count_and_length_at_every_tolerance() -> None:
    row = interiority_row(
        "b1", BOUNDARY, _lines(LineString([(10, 50), (90, 50)])), _lines(), CRS_M)
    assert row["block_id"] == "b1"
    for tol in TOLERANCES:
        assert row[f"n_interior_segments_{tol}"] == 1
        assert row[f"interior_length_m_{tol}"] == pytest.approx(80.0)


def test_interiority_row_count_gate_is_robust_where_length_is_not() -> None:
    """A path crossing the interior but touching the edge: length is trimmed by tolerance,
    the count is not. This is the spike's central finding and the reason both are reported."""
    row = interiority_row(
        "b2", BOUNDARY, _lines(LineString([(0, 50), (90, 50)])), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == row["n_interior_segments_5.0"] == 1
    assert row["interior_length_m_0.5"] > row["interior_length_m_5.0"]


def test_interiority_row_keeps_near_miss_separate() -> None:
    row = interiority_row(
        "b3", BOUNDARY,
        _lines(LineString([(10, 50), (90, 50)])),
        _lines(LineString([(10, 20), (90, 20)])),
        CRS_M)
    assert row["n_interior_segments_0.5"] == 1
    assert row["n_near_miss_segments_0.5"] == 1
    assert row["interior_length_m_0.5"] == pytest.approx(80.0)


def test_interiority_row_uncovered_block_is_all_zero() -> None:
    row = interiority_row("b4", BOUNDARY, _lines(), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == 0
    assert row["interior_length_m_0.5"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/data/test_osm_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reblock.data.osm_extract'`

- [ ] **Step 3: Write the implementation**

Create `src/reblock/data/osm_extract.py`:

```python
"""Country-wide OSM footpath census (tier T1): per-block interior-footpath coverage over the
whole ZAF+KEN block corpus, computed from the blocks parquet + OSM linework alone -- no
building points, no Voronoi parcels, no Block. See
docs/superpowers/specs/2026-07-27-ot-retrieval-substrate-phase1-design.md.
"""
from __future__ import annotations

import geopandas as gpd
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from reblock.methods.osm_footpaths import interior_desire_lines

# The shipped osm_footpaths tag set, imported by conf/desire_source/_footpath_tags.yaml so the
# census and the method can never disagree about what a footpath is.
FOOTPATH_TAGS: tuple[str, ...] = (
    "path", "footway", "track", "steps", "pedestrian", "living_street")
# Tags informal paths are SOMETIMES mapped under. Counted separately and never mixed into the
# primary columns, so the cost of widening the filter is visible before anyone re-extracts.
NEAR_MISS_TAGS: tuple[str, ...] = ("service", "residential", "unclassified")
# OSM ways are digitized against different imagery than the kblock outlines, so a boundary-running
# path more than STREET_TOL off the outline reads as interior. Measured: the count gate moves only
# 2.6 points across this range while total length drops ~18%, so tolerance matters for
# donor-quality ranking, not for the coverage census -- but report both and let the data say so.
TOLERANCES: tuple[float, ...] = (0.5, 2.0, 5.0)


def interiority_row(
    block_id: str,
    boundary: BaseGeometry,
    footpaths: gpd.GeoDataFrame,
    near_miss: gpd.GeoDataFrame,
    crs: CRS,
) -> dict[str, object]:
    """One census row: interior segment count and length at every tolerance, for the primary
    footpath tags and (separately) the near-miss tags.

    For a kblock block `streets` IS the outline, so the street corridor is `boundary.boundary`.
    """
    streets = boundary.boundary
    row: dict[str, object] = {"block_id": block_id, "boundary_length_m": float(streets.length)}
    for label, lines in (("interior", footpaths), ("near_miss", near_miss)):
        for tol in TOLERANCES:
            kept = interior_desire_lines(lines, boundary, streets, crs, tol=tol)
            row[f"n_{label}_segments_{tol}"] = int(len(kept))
            row[f"{label}_length_m_{tol}"] = (
                float(kept.geometry.length.sum()) if len(kept) else 0.0)
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/data/test_osm_extract.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Typecheck, lint, commit**

```bash
pixi run typecheck && pixi run lint
git add src/reblock/data/osm_extract.py tests/data/test_osm_extract.py
git commit -m "feat: interiority_row -- census row with tolerance sweep and near-miss tags

Reports segment COUNT and length at 0.5/2/5m. Both matter: coverage is gated on
count, donor quality on length, and the spike measured that tolerance moves
length ~18% while barely moving the count gate."
```

---

### Task 3: One shared footpath tag list

`conf/desire_source/osm.yaml` currently re-declares the tag list, so the shipped method's effective tags come from Hydra, not from Python. Importing `FOOTPATH_TAGS` would therefore guarantee nothing. Fix the divergence at the source.

**Files:**
- Create: `conf/desire_source/_footpath_tags.yaml`
- Modify: `conf/desire_source/osm.yaml`
- Test: `tests/data/test_osm_extract.py`

**Interfaces:**
- Consumes: `FOOTPATH_TAGS` (Task 2)
- Produces: a Hydra-resolvable `${footpath_tags}` list

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_osm_extract.py`:

```python
from pathlib import Path

import yaml

from reblock.data.osm_extract import FOOTPATH_TAGS


def test_config_tag_list_matches_python_definition() -> None:
    """conf/ and Python must not be able to drift: one list, one place."""
    shared = yaml.safe_load(Path("conf/desire_source/_footpath_tags.yaml").read_text())
    assert tuple(shared["footpath_tags"]) == FOOTPATH_TAGS

    osm_cfg = yaml.safe_load(Path("conf/desire_source/osm.yaml").read_text())
    assert osm_cfg["tags"] == "${footpath_tags}", (
        "osm.yaml must interpolate the shared list, not re-declare it")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/data/test_osm_extract.py -k config_tag_list -v`
Expected: FAIL — the file does not exist.

- [ ] **Step 3: Create the shared list**

Create `conf/desire_source/_footpath_tags.yaml`:

```yaml
# @package _global_
# The ONE footpath tag list. reblock.data.osm_extract.FOOTPATH_TAGS is the Python mirror, and
# tests/data/test_osm_extract.py::test_config_tag_list_matches_python_definition fails if they
# drift. Every DesireLineSource config interpolates ${footpath_tags} rather than re-declaring.
footpath_tags: [path, footway, track, steps, pedestrian, living_street]
```

- [ ] **Step 4: Point osm.yaml at it**

In `conf/desire_source/osm.yaml`, replace the `tags:` line with:

```yaml
defaults:
  - _footpath_tags
tags: ${footpath_tags}
```

- [ ] **Step 5: Run test and a config smoke check**

Run: `pixi run pytest tests/data/test_osm_extract.py -k config_tag_list -v`
Expected: PASS

Run: `pixi run python -c "import hydra; from hydra import compose, initialize; initialize(config_path='conf', version_base=None); print(compose(config_name='config', overrides=['desire_source=osm']).desire_source.tags)"`
Expected: prints the six tags. If Hydra cannot resolve `${footpath_tags}` from a `defaults` entry in a group config, move the key into `conf/config.yaml` instead and update the test's second assertion to match.

- [ ] **Step 6: Commit**

```bash
git add conf/desire_source/ tests/data/test_osm_extract.py
git commit -m "fix: one footpath tag list, interpolated not re-declared

osm.yaml re-declared the tags, so the shipped method's effective set came from
Hydra while Python held a separate default -- they could silently diverge. A test
now fails if they do."
```

---

### Task 4: `PbfDesireLines`

**Files:**
- Modify: `src/reblock/data/osm_extract.py`
- Create: `conf/desire_source/pbf.yaml`
- Modify: `pyproject.toml` (register the `network` marker)
- Test: `tests/data/test_osm_extract.py`

**Interfaces:**
- Consumes: `FOOTPATH_TAGS`, `NEAR_MISS_TAGS`
- Produces: `PbfDesireLines(pbf_path: Path, tags: Sequence[str] = FOOTPATH_TAGS)` implementing `DesireLineSource`, with `.desire_lines(bbox_wgs84, crs) -> GeoDataFrame` and `.identity -> tuple[str, str, tuple[str, ...]]`; plus `read_pbf_lines(pbf_path: Path, tags: Sequence[str]) -> GeoDataFrame` for the batch path.

- [ ] **Step 1: Register the network marker**

In `pyproject.toml` under `[tool.pytest.ini_options]`, add:

```toml
markers = [
    "network: hits the network (Geofabrik/Overpass/Open Buildings); deselect with -m 'not network'",
]
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/data/test_osm_extract.py`:

```python
from reblock.data.osm_extract import PbfDesireLines


def test_pbf_identity_is_stable_and_keys_on_content_and_tags(tmp_path: Path) -> None:
    """Unlike OSMDesireLines (identity None when live), a PBF source is cacheable -- which is
    what flips osm_footpaths from uncacheable to cacheable, so the identity must be content-keyed."""
    pbf = tmp_path / "x.osm.pbf"
    pbf.write_bytes(b"not-a-real-pbf-but-hashable")
    a = PbfDesireLines(pbf)
    b = PbfDesireLines(pbf)
    assert a.identity == b.identity
    assert a.identity is not None

    pbf2 = tmp_path / "y.osm.pbf"
    pbf2.write_bytes(b"different-content")
    assert PbfDesireLines(pbf2).identity != a.identity
    assert PbfDesireLines(pbf, tags=("footway",)).identity != a.identity


def test_pbf_conforms_to_desire_line_source_protocol() -> None:
    """Structural conformance is enforced STATICALLY by this annotated binding -- mypy --strict
    fails if PbfDesireLines does not satisfy the Protocol. Do NOT rewrite this as
    `isinstance(..., DesireLineSource)`: DesireLineSource is a bare Protocol, not
    @runtime_checkable, so isinstance raises TypeError rather than returning False."""
    from reblock.methods.desire_lines import DesireLineSource

    source: DesireLineSource = PbfDesireLines(Path("nonexistent.osm.pbf"))
    assert callable(source.desire_lines)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run pytest tests/data/test_osm_extract.py -k pbf -v`
Expected: FAIL with `ImportError: cannot import name 'PbfDesireLines'`

- [ ] **Step 4: Implement**

Append to `src/reblock/data/osm_extract.py`:

```python
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyogrio


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pbf_lines(pbf_path: Path, tags: Sequence[str] = FOOTPATH_TAGS) -> gpd.GeoDataFrame:
    """Every `highway` way of the given tag classes in a .osm.pbf, as EPSG:4326 LineStrings.

    Filters OGR-side via `where` so a country extract does not materialize every South African
    `highway=track` in Python. NOTE: this reduces what crosses into pandas; it does NOT avoid the
    GDAL OSM driver building its multi-GB temp SQLite database, which happens regardless.

    The census must call this ONCE per UTM batch and query an STRtree per block -- not once per
    block. `DesireLineSource.desire_lines` is a per-bbox API and there are 1.81M blocks.
    """
    quoted = ", ".join(f"'{t}'" for t in tags)
    return cast(gpd.GeoDataFrame, pyogrio.read_dataframe(
        pbf_path, layer="lines", where=f"highway IN ({quoted})", use_arrow=True))


@dataclass
class PbfDesireLines:
    """A DesireLineSource backed by a local Geofabrik .osm.pbf extract.

    A second implementation alongside OSMDesireLines, not a replacement: the operating ranges are
    disjoint (a PBF covers its extract; Overpass covers any bbox). At 1.81M blocks a bulk extract
    is the only workable option -- one 0.25-degree Overpass tile is ~29 MB / 40k ways / 7 s, and
    ZAF+KEN is ~4,598 such tiles against Overpass's ~1 GB/day fair-use policy, versus 766 MB of
    PBF once.

    `identity` is stable (unlike OSMDesireLines' None-when-live), so osm_footpaths becomes
    cacheable when driven by this source.
    """

    pbf_path: Path
    tags: Sequence[str] = FOOTPATH_TAGS
    _cache: gpd.GeoDataFrame | None = field(default=None, init=False, repr=False)

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame:
        if self._cache is None:
            self._cache = read_pbf_lines(self.pbf_path, self.tags)
        minx, miny, maxx, maxy = bbox_wgs84
        window = cast(gpd.GeoDataFrame, self._cache.cx[minx:maxx, miny:maxy])
        return cast(gpd.GeoDataFrame, window.to_crs(crs))

    @property
    def identity(self) -> tuple[str, str, tuple[str, ...]]:
        return ("pbf", _file_sha256(self.pbf_path), tuple(self.tags))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/data/test_osm_extract.py -k pbf -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Add the config**

Create `conf/desire_source/pbf.yaml`:

```yaml
# Local Geofabrik .osm.pbf desire-line source -- the bulk path used by the country-wide census.
# OSMDesireLines (conf/desire_source/osm.yaml) remains for bboxes outside the extracts.
defaults:
  - _footpath_tags
_target_: reblock.data.osm_extract.PbfDesireLines
pbf_path: ???        # e.g. ~/.cache/reblock/osm_pbf/south-africa-latest.osm.pbf
tags: ${footpath_tags}
```

- [ ] **Step 7: Write the source-agreement test (network-marked)**

Append to `tests/data/test_osm_extract.py`:

```python
@pytest.mark.network
def test_pbf_and_overpass_agree_on_a_pinned_bbox() -> None:
    """Two sources for the same data WILL disagree (Geofabrik extract timestamp vs live Overpass;
    GDAL `lines` layer vs Overpass `out geom`). Without this test, two sources is accommodation
    rather than a Strategy. Tolerance is loose because the snapshots differ in date, not content."""
    from reblock.methods.desire_lines import OSMDesireLines

    pbf = Path.home() / ".cache" / "reblock" / "osm_pbf" / "south-africa-latest.osm.pbf"
    if not pbf.exists():
        pytest.skip("run scripts/osm_census.py --fetch first")

    bbox = (18.55, -33.99, 18.58, -33.96)   # a Cape Flats window with dense footpath mapping
    crs = CRS.from_epsg(32734)
    a = PbfDesireLines(pbf).desire_lines(bbox, crs)
    b = OSMDesireLines(timeout_s=180.0).desire_lines(bbox, crs)
    assert a.geometry.length.sum() == pytest.approx(b.geometry.length.sum(), rel=0.25)
```

- [ ] **Step 8: Check the committed-example churn this causes**

A stable `PbfDesireLines.identity` flips `osm_footpaths` from uncacheable to cacheable, which
changes `proposal_id` (it embeds a hash of the source identity — `osm_footpaths.py:56-70`). There
are **6 committed `desire_lines_*.geojson` snapshots** under `examples/`, and their regenerated
outputs would move if any example switched to this source.

Run: `git status --short examples/ && pixi run pytest tests/test_gen_examples.py -v`
Expected: clean and PASS. Nothing should have changed — **no example config switches to
`PbfDesireLines` in this task**, so this step is a guard, not a migration. If `examples/` is dirty
or that test fails, stop: the identity change leaked into the shipped example outputs and needs
resolving before this lands.

Note for later: the spec says `PbfDesireLines` becomes the default source for ZAF/KEN after Phase
1. That switch is deliberately **not** made here — it would regenerate all six snapshots on the
strength of a source the census has not yet validated. Do it as its own change, after the census
run, with the regeneration reviewed.

- [ ] **Step 9: Run the non-network suite, typecheck, lint, commit**

```bash
pixi run pytest -m "not network"
pixi run typecheck && pixi run lint
git add src/reblock/data/osm_extract.py conf/desire_source/pbf.yaml pyproject.toml tests/data/test_osm_extract.py
git commit -m "feat: PbfDesireLines -- bulk .osm.pbf DesireLineSource

A second DesireLineSource, not a replacement: disjoint operating ranges. At 1.81M
blocks Overpass is not viable (one 0.25deg tile is 29MB/40k ways/7s; ZAF+KEN is
~4598 tiles against a ~1GB/day fair-use policy) versus 766MB of PBF once.
Content-keyed identity makes osm_footpaths cacheable when driven by this source.
Adds a network-marked pinned-bbox agreement test between the two sources."
```

---

### Task 5: Census driver

**Files:**
- Create: `scripts/osm_census.py`
- Modify: `src/reblock/data/osm_extract.py` (add `utm_zone_epsg`, `census_rows`)
- Test: `tests/data/test_osm_extract.py`

**Interfaces:**
- Consumes: `interiority_row`, `read_pbf_lines`
- Produces:
  - `utm_zone_epsg(lon: float, lat: float) -> int`
  - `assert_zone_fit(lon: float, epsg: int) -> None` — raises if the block is >3.5° from the central meridian
  - `census_rows(blocks: GeoDataFrame, footpaths: GeoDataFrame, near_miss: GeoDataFrame, epsg: int) -> list[dict[str, object]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_osm_extract.py`:

```python
from reblock.data.osm_extract import assert_zone_fit, census_rows, utm_zone_epsg


def test_utm_zone_epsg_picks_hemisphere_and_zone() -> None:
    assert utm_zone_epsg(18.5, -33.9) == 32734      # Cape Town, zone 34 south
    assert utm_zone_epsg(36.8, -1.3) == 32737       # Nairobi, zone 37 south
    assert utm_zone_epsg(36.8, 1.3) == 32637        # just north of the equator


def test_assert_zone_fit_is_loud_about_a_forgotten_batch() -> None:
    """A single country-wide UTM does not crash -- it silently biases lengths by up to 3.5%.
    The assertion is what makes a missed batch loud instead of a quiet drift."""
    assert_zone_fit(18.5, 32734)                    # zone 34 central meridian is 21E
    with pytest.raises(ValueError, match="outside UTM zone"):
        assert_zone_fit(41.9, 32734)


def test_census_rows_emits_one_row_per_block() -> None:
    blocks = gpd.GeoDataFrame(
        {"block_id": ["a", "b"]},
        geometry=[
            Polygon([(18.50, -33.95), (18.51, -33.95), (18.51, -33.94), (18.50, -33.94)]),
            Polygon([(18.52, -33.95), (18.53, -33.95), (18.53, -33.94), (18.52, -33.94)]),
        ],
        crs=CRS.from_epsg(4326))
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS.from_epsg(4326))
    rows = census_rows(blocks, empty, empty, 32734)
    assert [r["block_id"] for r in rows] == ["a", "b"]
    assert all(r["n_interior_segments_0.5"] == 0 for r in rows)
    assert all(r["area_m2"] > 0 for r in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/data/test_osm_extract.py -k "utm or zone or census_rows" -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

Append to `src/reblock/data/osm_extract.py`:

```python
from shapely import STRtree


def utm_zone_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx north, 327xx south."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def assert_zone_fit(lon: float, epsg: int) -> None:
    """Raise if `lon` is more than 3.5 degrees from the zone's central meridian.

    Load-bearing: `estimate_utm_crs()` on a whole-country extent returns a single zone with NO
    error, and transverse Mercator stays conformal, so nothing crashes -- you just get a silent
    scale bias (measured +0.72% at Cape Town, +1.23% at lon 16.5, +3.46% at lon 41.9 under one
    country-wide UTM). That biases interior_length_m by 1-3%. This makes a forgotten batch loud.
    """
    zone = epsg - (32600 if epsg < 32700 else 32700)
    central = 6 * zone - 183
    if abs(lon - central) > 3.5:
        raise ValueError(
            f"longitude {lon} is outside UTM zone {zone} (central meridian {central}); "
            f"batch blocks by zone via utm_zone_epsg before projecting")


def census_rows(
    blocks: gpd.GeoDataFrame,
    footpaths: gpd.GeoDataFrame,
    near_miss: gpd.GeoDataFrame,
    epsg: int,
) -> list[dict[str, object]]:
    """Census rows for one UTM batch. `blocks` is in EPSG:4326; everything is reprojected to
    `epsg` once, then each block queries an STRtree rather than re-reading the layer."""
    crs_m = CRS.from_epsg(epsg)
    blocks_m = blocks.to_crs(crs_m)
    fp_m = footpaths.to_crs(crs_m) if len(footpaths) else footpaths.set_crs(crs_m, allow_override=True)
    nm_m = near_miss.to_crs(crs_m) if len(near_miss) else near_miss.set_crs(crs_m, allow_override=True)
    fp_tree = STRtree(list(fp_m.geometry)) if len(fp_m) else None
    nm_tree = STRtree(list(nm_m.geometry)) if len(nm_m) else None

    rows: list[dict[str, object]] = []
    for block_id, geom in zip(blocks_m["block_id"], blocks_m.geometry, strict=True):
        near_fp = (fp_m.iloc[fp_tree.query(geom)] if fp_tree is not None
                   else gpd.GeoDataFrame(geometry=[], crs=crs_m))
        near_nm = (nm_m.iloc[nm_tree.query(geom)] if nm_tree is not None
                   else gpd.GeoDataFrame(geometry=[], crs=crs_m))
        row = interiority_row(str(block_id), geom, near_fp, near_nm, crs_m)
        # The qualified filter is a building-count band, which does NOT bound block AREA: the
        # spike found 5 of 251 covered blocks carrying >5 km of "interior" footpath on 90-293
        # buildings (max 26.5 km on 258 buildings, vs a 356 m median) -- huge polygons where the
        # clip captures a whole neighbourhood. Emit area so the guard is applied downstream on
        # data rather than guessed here.
        row["area_m2"] = float(geom.area)
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/data/test_osm_extract.py -k "utm or zone or census_rows" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the driver script**

Create `scripts/osm_census.py`:

```python
"""Country-wide OSM footpath census driver (Phase 1, unit 1a).

Streams the ZAF/KEN blocks parquet, batches blocks by UTM zone, reads the country footpath layer
ONCE per batch, and writes one row per block to ~/.cache/reblock/osm_coverage_{iso}.parquet.

Budget, measured: 3.31 ms/block for clip + corridor difference + filter, so ~1.67 single-core
hours per tolerance over 1.81M blocks -- about 5 h for the 0.5/2/5 m sweep and ~10 h once the
near-miss tag set is included. Use --limit for a smoke run first.

Usage:
    pixi run python -m scripts.osm_census --iso ZAF --limit 5000
    pixi run python -m scripts.osm_census --iso ZAF
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from reblock.data.osm_extract import (
    FOOTPATH_TAGS,
    NEAR_MISS_TAGS,
    assert_zone_fit,
    census_rows,
    read_pbf_lines,
    utm_zone_epsg,
)

CACHE = Path.home() / ".cache" / "reblock"
PBF = {"ZAF": "south-africa-latest.osm.pbf", "KEN": "kenya-latest.osm.pbf"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", choices=sorted(PBF), required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after N blocks (smoke run)")
    ap.add_argument("--batch-size", type=int, default=50_000)
    args = ap.parse_args()

    pbf_path = CACHE / "osm_pbf" / PBF[args.iso]
    if not pbf_path.exists():
        raise SystemExit(
            f"missing {pbf_path}\n"
            f"download it from https://download.geofabrik.de/ "
            f"(south-africa 417 MB, kenya 349 MB)")

    print(f"reading footpath layer from {pbf_path.name} ...", flush=True)
    t0 = time.time()
    footpaths = read_pbf_lines(pbf_path, FOOTPATH_TAGS)
    near_miss = read_pbf_lines(pbf_path, NEAR_MISS_TAGS)
    print(f"  {len(footpaths):,} footpath ways, {len(near_miss):,} near-miss "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Single-row-group parquets (833 MB / 386 MB): gpd.read_parquet will not stream a column.
    src = CACHE / f"{args.iso}_geodata.parquet"
    pf = pq.ParquetFile(src)
    rows: list[dict[str, object]] = []
    seen = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=args.batch_size,
                                 columns=["block_id", "building_count", "k_complexity",
                                          "geometry"]):
        blocks = gpd.GeoDataFrame.from_arrow(batch)
        if blocks.crs is None:
            blocks = blocks.set_crs(4326)
        by_zone: dict[int, list[int]] = defaultdict(list)
        reps = blocks.geometry.representative_point()
        for i, pt in enumerate(reps):
            by_zone[utm_zone_epsg(pt.x, pt.y)].append(i)

        for epsg, idx in by_zone.items():
            sub = blocks.iloc[idx]
            assert_zone_fit(float(reps.iloc[idx[0]].x), epsg)
            bounds = sub.total_bounds
            fp = footpaths.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            nm = near_miss.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
            batch_rows = census_rows(sub, fp, nm, epsg)
            for r, bc, kc in zip(batch_rows, sub["building_count"], sub["k_complexity"],
                                 strict=True):
                r["building_count"] = int(bc) if pd.notna(bc) else 0
                r["k_complexity"] = int(kc) if pd.notna(kc) else 0
            rows.extend(batch_rows)

        seen += len(blocks)
        rate = seen / max(time.time() - t0, 1e-9)
        print(f"  {seen:,} blocks  {rate:.0f}/s", flush=True)
        if args.limit and seen >= args.limit:
            print(f"stopping at --limit {args.limit} ({seen:,} blocks processed)", flush=True)
            break

    out = CACHE / f"osm_coverage_{args.iso}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    covered = sum(1 for r in rows if int(r["n_interior_segments_0.5"]) > 0)
    print(f"\nwrote {out}  ({len(rows):,} rows)")
    print(f"blocks with >=1 interior footpath segment: {covered:,} "
          f"({covered/max(len(rows),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Confirm the area guard is deferred deliberately, not forgotten**

The spec requires 1a to add an area or density guard, because the `building_count` band does not
bound block area. This plan **emits `area_m2` and applies no threshold**, on the grounds that the
right cut is not yet known — the spike saw 5 of 251 covered blocks over 5 km of interior footpath
against a 356 m median, which locates the outliers but does not fix a threshold. Choosing one from
the full census distribution is better than guessing one now.

Run: `pixi run python -c "
import pandas as pd, pathlib
d = pd.read_parquet(pathlib.Path.home()/'.cache/reblock/osm_coverage_ZAF.parquet')
c = d[d['n_interior_segments_0.5'] > 0]
print(c[['area_m2','interior_length_m_0.5']].describe(percentiles=[.5,.9,.99]).round(0))
print('m of footpath per building, p99:',
      (c['interior_length_m_0.5']/c['building_count'].clip(lower=1)).quantile(0.99).round(1))
"`
Expected: a distribution to pick the guard from. Record the chosen threshold in the census note.
Do **not** proceed to Task 7 without setting one — the shortlist it downloads for is exactly where
an unbounded block wastes tiles.

- [ ] **Step 7: Smoke-run the driver**

Run: `pixi run python -m scripts.osm_census --iso ZAF --limit 5000`
Expected: prints a per-batch rate and a final coverage percentage. If the PBF is absent it exits with the Geofabrik URL — download it and retry. Sanity-check the coverage percentage against the spike's 65.5% on qualified Cape Town blocks; a whole-country figure will be **lower** because it includes rural and formal blocks.

- [ ] **Step 8: Typecheck, lint, commit**

```bash
pixi run typecheck && pixi run lint
git add scripts/osm_census.py src/reblock/data/osm_extract.py tests/data/test_osm_extract.py
git commit -m "feat: country-wide OSM footpath census driver

Streams the single-row-group blocks parquet via iter_batches, batches by UTM zone
(estimate_utm_crs on a country extent silently biases lengths up to 3.5% rather
than failing -- assert_zone_fit makes that loud), reads the footpath layer once per
batch and queries an STRtree per block. Emits area_m2 so the area guard the spike
showed is needed can be applied on data downstream."
```

---

### Task 6: Settlement labels and the exclusion-radius holdout

Leave-one-block-out is unsafe: donors are immediate neighbours, often the same OSM way clipped at a block edge or one mapper's session. Leave-one-settlement-out is the obvious fix but is not well-defined — measured on Cape Town's 1,136 qualified blocks, a 100 m threshold gives 417 components, 23% singletons, and a 150-block component spanning 5.7 km that swallows Gugulethu while Nyanga, Langa and Delft are singletons. So the **exclusion radius is primary** and `settlement_id` is a reporting label only.

**Files:**
- Create: `src/reblock/data/settlements.py`
- Test: `tests/data/test_settlements.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `settlement_labels(blocks: GeoDataFrame, *, tol_m: float = 100.0) -> list[int]`
  - `exclusion_holdout(blocks: GeoDataFrame, recipient_idx: int, *, radius_m: float) -> list[int]` — indices eligible as donors

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_settlements.py`:

```python
import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import box

from reblock.data.settlements import exclusion_holdout, settlement_labels

CRS_M = CRS.from_epsg(32734)


def _blocks(*offsets: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[box(x, 0, x + 50, 50) for x in offsets], crs=CRS_M)


def test_settlement_labels_group_near_blocks_and_split_far_ones() -> None:
    blocks = _blocks(0, 60, 5000)          # first two within 100 m, third far away
    labels = settlement_labels(blocks, tol_m=100.0)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_settlement_labels_chain_transitively() -> None:
    """Documents the pathology that demotes this to a reporting label: A-B and B-C within
    tolerance puts A and C in one settlement however far apart they are."""
    blocks = _blocks(0, 60, 120, 180)
    assert len(set(settlement_labels(blocks, tol_m=100.0))) == 1


def test_exclusion_holdout_drops_everything_inside_the_radius() -> None:
    blocks = _blocks(0, 60, 5000)
    donors = exclusion_holdout(blocks, recipient_idx=0, radius_m=100.0)
    assert donors == [2]


def test_exclusion_holdout_never_returns_the_recipient() -> None:
    blocks = _blocks(0, 5000)
    assert 0 not in exclusion_holdout(blocks, recipient_idx=0, radius_m=0.0)


def test_exclusion_holdout_is_monotone_in_radius() -> None:
    """The property that makes this a defensible fold definition: more radius, never more donors."""
    blocks = _blocks(0, 60, 200, 5000)
    counts = [len(exclusion_holdout(blocks, 0, radius_m=r)) for r in (0.0, 100.0, 500.0, 10_000.0)]
    assert counts == sorted(counts, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/data/test_settlements.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/reblock/data/settlements.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/data/test_settlements.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Typecheck, lint, commit**

```bash
pixi run typecheck && pixi run lint
git add src/reblock/data/settlements.py tests/data/test_settlements.py
git commit -m "feat: exclusion-radius holdout + settlement labels

Exclusion radius is the fold definition (monotone, no chaining, sweepable);
settlement_labels is a reporting label only, because measurement showed component
labelling cannot separate Cape Flats settlements at any single threshold."
```

---

### Task 7: Multi-tile Open Buildings provisioning

Current `download_capetown_buildings` picks the single tile containing the bbox *centroid* and filters with a lon/lat rectangle. A rectangle around a ZAF+KEN shortlist is both countries, which would retain essentially every row and defeat the point of being query-driven. Measured: `tiles.geojson` has 333 features, of which **20 cover ZAF+KEN** at **3.78 GB** gzipped.

**Files:**
- Modify: `src/reblock/data/provision.py`
- Test: `tests/data/test_provision.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `tiles_for(shortlist: GeoDataFrame, tiles: GeoDataFrame) -> list[str]` — point-tile URLs intersecting the shortlist
  - `filter_to_shortlist(points: GeoDataFrame, shortlist: GeoDataFrame) -> GeoDataFrame` — per-polygon join

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_provision.py`:

```python
import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Point, box

from reblock.data.provision import filter_to_shortlist, tiles_for

WGS = CRS.from_epsg(4326)


def _tiles() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"tile_url": ["a.csv.gz", "b.csv.gz", "c.csv.gz"]},
        geometry=[box(18, -34, 19, -33), box(36, -2, 37, -1), box(0, 0, 1, 1)],
        crs=WGS)


def test_tiles_for_returns_only_intersecting_tiles() -> None:
    shortlist = gpd.GeoDataFrame(
        geometry=[box(18.5, -33.9, 18.6, -33.8), box(36.8, -1.3, 36.9, -1.2)], crs=WGS)
    assert sorted(tiles_for(shortlist, _tiles())) == ["a.csv.gz", "b.csv.gz"]


def test_tiles_for_is_not_fooled_by_the_bounding_rectangle() -> None:
    """The bug this replaces: a bbox around a ZAF+KEN shortlist spans everything between them."""
    shortlist = gpd.GeoDataFrame(
        geometry=[box(18.5, -33.9, 18.6, -33.8), box(36.8, -1.3, 36.9, -1.2)], crs=WGS)
    assert "c.csv.gz" not in tiles_for(shortlist, _tiles())


def test_filter_to_shortlist_keeps_only_points_inside_a_block() -> None:
    shortlist = gpd.GeoDataFrame(geometry=[box(18.5, -33.9, 18.6, -33.8)], crs=WGS)
    points = gpd.GeoDataFrame(
        {"confidence": [0.9, 0.9]},
        geometry=[Point(18.55, -33.85), Point(25.0, -30.0)],   # inside, then far away
        crs=WGS)
    kept = filter_to_shortlist(points, shortlist)
    assert len(kept) == 1
    assert kept.geometry.iloc[0].x == pytest.approx(18.55)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/data/test_provision.py -v`
Expected: FAIL with `ImportError: cannot import name 'tiles_for'`

- [ ] **Step 3: Implement**

Append to `src/reblock/data/provision.py`:

```python
def tiles_for(shortlist: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> list[str]:
    """Open Buildings point-tile URLs whose S2 cell intersects any shortlist block.

    Measured: tiles.geojson has 333 features, 20 of which cover ZAF+KEN (3.78 GB gzipped as
    points; the polygon variants are 14.09 GB). The existing single-centroid-tile lookup is
    correct only for a bbox smaller than one cell.
    """
    joined = gpd.sjoin(tiles.to_crs(shortlist.crs), shortlist[["geometry"]],
                       how="inner", predicate="intersects")
    return sorted(set(joined["tile_url"]))


def filter_to_shortlist(
    points: gpd.GeoDataFrame, shortlist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Keep only building points falling inside a shortlist block POLYGON.

    Load-bearing: filtering to the shortlist's bounding rectangle would, for a ZAF+KEN shortlist,
    retain essentially every Open Buildings row in the download and make the targeted provisioning
    country-wide by accident.
    """
    joined = gpd.sjoin(points, shortlist[["geometry"]], how="inner", predicate="within")
    return cast(gpd.GeoDataFrame, joined.drop(columns=["index_right"]))
```

Add `from typing import cast` if not already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/data/test_provision.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Typecheck, lint, commit**

```bash
pixi run typecheck && pixi run lint
git add src/reblock/data/provision.py tests/data/test_provision.py
git commit -m "feat: multi-tile Open Buildings enumeration + per-polygon filter

tiles.geojson has 333 features, 20 covering ZAF+KEN (3.78 GB gzipped). The
single-centroid-tile lookup only works for a bbox inside one cell, and a
bounding-rectangle filter around a two-country shortlist retains everything --
so the filter is a per-polygon sjoin, not a rectangle."
```

---

### Task 8: Agreement primitives

Plain functions, **not** an `Eval`: `Eval.score(block, proposal)` (`src/reblock/contracts.py:124`) has no slot for a reference network, and agreement is proposal-vs-reference.

**Files:**
- Create: `src/reblock/eval/agreement.py`
- Test: `tests/eval/test_agreement.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `buffered_iou(proposal: GeoDataFrame, reference: GeoDataFrame, *, r: float = 3.0) -> float`
  - `directional_chamfer(proposal: GeoDataFrame, reference: GeoDataFrame, *, step: float = 2.0) -> tuple[float, float]` — `(precision_m, recall_m)`: mean proposal→reference distance, then reference→proposal

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_agreement.py`:

```python
import geopandas as gpd
import pytest
from pyproj import CRS
from shapely import affinity
from shapely.geometry import LineString

from reblock.eval.agreement import buffered_iou, directional_chamfer

CRS_M = CRS.from_epsg(32734)


def _net(*geoms: LineString) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geoms), crs=CRS_M)


def test_identical_networks_score_perfectly() -> None:
    net = _net(LineString([(0, 0), (100, 0)]), LineString([(50, -50), (50, 50)]))
    assert buffered_iou(net, net) == pytest.approx(1.0)
    precision, recall = directional_chamfer(net, net)
    assert precision == pytest.approx(0.0, abs=1e-6)
    assert recall == pytest.approx(0.0, abs=1e-6)


def test_disjoint_networks_score_zero_iou() -> None:
    a = _net(LineString([(0, 0), (100, 0)]))
    b = _net(LineString([(0, 10_000), (100, 10_000)]))
    assert buffered_iou(a, b) == pytest.approx(0.0)


def test_iou_decays_monotonically_with_offset() -> None:
    """A single far-offset case is vacuous -- with r=3 the buffers stop overlapping past 6 m and
    score 0 for ANY metric. The graded series is what actually tests the metric.

    Deliberately NOT parametrized: the property under test is the ORDERING across offsets, which
    a per-offset parametrization cannot express (each case would only see its own score)."""
    ref = _net(LineString([(0, 0), (100, 0)]))
    scores = {
        o: buffered_iou(_net(affinity.translate(ref.geometry.iloc[0], yoff=o)), ref, r=3.0)
        for o in (0.0, 1.0, 3.0, 6.0, 12.0)
    }
    assert scores[0.0] > scores[1.0] > scores[3.0] > scores[6.0]
    assert scores[0.0] == pytest.approx(1.0)
    assert scores[6.0] == pytest.approx(0.0)
    assert scores[12.0] == pytest.approx(0.0)


def test_iou_at_offset_equal_to_radius_is_a_pinned_value() -> None:
    ref = _net(LineString([(0, 0), (100, 0)]))
    prop = _net(affinity.translate(ref.geometry.iloc[0], yoff=3.0))
    assert buffered_iou(prop, ref, r=3.0) == pytest.approx(0.33, abs=0.03)


def test_chamfer_is_asymmetric_for_a_strict_subset() -> None:
    """The only thing directional Chamfer exists to expose. A proposal covering half the
    reference has near-zero precision error but large recall error; the transpose flips it."""
    reference = _net(LineString([(0, 0), (100, 0)]))
    proposal = _net(LineString([(0, 0), (50, 0)]))

    precision, recall = directional_chamfer(proposal, reference)
    assert precision == pytest.approx(0.0, abs=0.5)
    assert recall > 5.0

    precision_t, recall_t = directional_chamfer(reference, proposal)
    assert recall_t == pytest.approx(0.0, abs=0.5)
    assert precision_t > 5.0


def test_chamfer_densification_step_is_a_quantization_floor() -> None:
    """A 2 m step puts a ~1 m floor on Chamfer; a finer step must not raise the score."""
    ref = _net(LineString([(0, 0), (100, 0)]))
    prop = _net(affinity.translate(ref.geometry.iloc[0], yoff=4.0))
    coarse, _ = directional_chamfer(prop, ref, step=2.0)
    fine, _ = directional_chamfer(prop, ref, step=0.5)
    assert fine <= coarse + 1e-6
    assert fine == pytest.approx(4.0, abs=0.2)


def test_empty_proposal_scores_zero_iou() -> None:
    assert buffered_iou(_net(), _net(LineString([(0, 0), (100, 0)]))) == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/eval/test_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reblock.eval.agreement'`

- [ ] **Step 3: Implement**

Create `src/reblock/eval/agreement.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/eval/test_agreement.py -v`
Expected: PASS (11 tests, counting the parametrized cases)

- [ ] **Step 5: Typecheck, lint, commit**

```bash
pixi run typecheck && pixi run lint
git add src/reblock/eval/agreement.py tests/eval/test_agreement.py
git commit -m "feat: geometric agreement primitives (buffered IoU + directional Chamfer)

Plain functions, not an Eval -- Eval.score(block, proposal) has no reference slot.
Directional Chamfer is never averaged: precision and recall answer different
questions and a blend hides which way a prediction fails. Tests use a graded
offset series rather than one large offset, which with r=3 would score 0 for any
metric and prove nothing."
```

---

### Task 9: GW pair-matrix pilot

Scratchpad experiment, but the deliverable is a **committed parquet** — the most reusable thing Phase 1 produces. An earlier draft named scratchpad loss as a risk and then guaranteed it by making the deliverable a note.

**Files:**
- Create: `scripts/pair_matrix.py`
- Create: `data/benchmarks/gw_pair_matrix.parquet` (committed)
- No tests: this is an experiment; the parquet and a note are the artifacts.

**Interfaces:**
- Consumes: `exclusion_holdout` (Task 6)
- Produces: a parquet with columns `recipient, donor, donor_type, real_gw_dist, feature_dist, perm_gap, displacement_proposal, displacement_direct, road_len_m, wall_clock_s`

- [ ] **Step 1: Salvage the prior OT code**

The 2026-07-23 scratchpad survives in another session's directory and is one `/tmp` reclaim from being lost. Copy it first:

```bash
SRC=/tmp/claude-1641171234/-home-gchurchill-src-reblock/27c82570-a74d-47e6-9e87-e53987507f6d/scratchpad
mkdir -p scratchpad/ot
cp "$SRC"/{ot_gw.py,transplant.py,select_donor.py,barycenter_amortization.py,osm_barycenter.py,gap_snap_fix.py} scratchpad/ot/ 2>/dev/null
cp "$SRC"/rsc_*.py scratchpad/ot/ 2>/dev/null
ls scratchpad/ot/
```

Expected: `ot_gw.py`, `transplant.py`, `select_donor.py`, `barycenter_amortization.py`, `osm_barycenter.py` at minimum. If the directory is gone, the GW implementation must be rebuilt from `notes/2026-07-23-ot-road-transplant.md` §1 (entropic GW, projected-gradient outer loop + log-domain Sinkhorn inner, ε = 0.01, τ = 1.0) — budget a day rather than an hour.

- [ ] **Step 2: Run a 20-pair timing pilot before anything else**

Nothing downstream should be sized before this number exists. Each pair needs a real entropic GW fit at ε ≤ 0.01 (the note's ablation makes that mandatory — at ε = 0.05 the network collapses to ~3% of its length — and log-domain Sinkhorn converges slowly there), plus transplant, snap, a length-matched clearance solve, and permeability on both.

Create `scripts/pair_matrix.py`. The salvaged API (verified present in the scratchpad) is:

```python
# scratchpad/ot/ot_gw.py
entropic_gw_unbalanced(c1, c2, p, q, *, eps=0.01, tau=1.0, ...) -> Arr    # the coupling pi
gw_cost(pi, c1, c2) -> float
barycentric_projection(pi, y, fallback=None) -> Arr
# scratchpad/ot/transplant.py
fit_transport(donor_xy, recipient_xy, *, eps=0.01, tau=1.0, ...) -> GWTransportResult
transport_lines(lines, result, ...) -> gpd.GeoDataFrame
gap_snap(lines, block, ...) -> gpd.GeoDataFrame        # NOTE: takes a Block (tier T3)
_normalized_dist_matrix(xy) -> Arr
# scratchpad/ot/select_donor.py
signature(xy, n_sub=30, seed=..., n_boot=5) -> np.ndarray
```

`gap_snap` takes a `Block`, so recipients must be constructible — i.e. tier T3, with building
points provisioned. Draw them from the shortlist Task 7 downloaded, not from the raw census.

```python
"""GW pair-matrix benchmark (Phase 1, unit 1d).

Per (recipient, donor): fit real entropic GW, transplant the donor's linework, snap it to the
recipient's substrate, and score it against a length-matched direct clearance solve. The output
parquet is a retrieval benchmark -- any future featurization or donor material can be scored
against it without re-solving anything.

Usage:
    pixi run python scripts/pair_matrix.py --pairs 20 --timing-only
    pixi run python scripts/pair_matrix.py --pairs 100 --out data/benchmarks/gw_pair_matrix.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scratchpad/ot")
from ot_gw import gw_cost                                    # noqa: E402
from select_donor import signature                           # noqa: E402
from transplant import _normalized_dist_matrix, fit_transport, gap_snap, transport_lines  # noqa: E402

from reblock.budget import building_radii, displacement      # noqa: E402
from reblock.data.settlements import exclusion_holdout       # noqa: E402
from reblock.methods.clearance import ClearanceReblocker     # noqa: E402
from reblock.permeability import permeability                # noqa: E402

CORRIDOR_M = 3.0


def displacement_fraction(block, roads) -> float:
    """Expected homes displaced as a fraction of the block's buildings.

    `budget.displacement` takes `(building_points, radii, roads, corridor_m)` and returns a COUNT,
    not a fraction and not a Block -- this mirrors the normalization in `emit.py:92`.
    """
    pts = block.building_points
    n = len(pts)
    if n == 0:
        return 0.0
    radii = building_radii(pts, CORRIDOR_M)
    return float(displacement(pts, radii, roads, CORRIDOR_M) / n)


def score_pair(recipient, donor, donor_lines, timings: dict[str, float]) -> dict[str, object]:
    """One matrix row. `recipient`/`donor` are Blocks; `donor_lines` is the donor's material."""
    r_xy = np.c_[recipient.parcels.geometry.centroid.x, recipient.parcels.geometry.centroid.y]
    d_xy = np.c_[donor.parcels.geometry.centroid.x, donor.parcels.geometry.centroid.y]

    t = time.time()
    result = fit_transport(d_xy, r_xy, eps=0.01, tau=1.0)
    timings["gw"] += time.time() - t
    dist = gw_cost(result.pi, _normalized_dist_matrix(d_xy), _normalized_dist_matrix(r_xy))

    t = time.time()
    moved = gap_snap(transport_lines(donor_lines, result), recipient)
    timings["transplant"] += time.time() - t

    t = time.time()
    road_len = float(moved.geometry.length.sum())
    direct = ClearanceReblocker().propose(recipient).roads
    # Length-match the baseline by truncating to a prefix of comparable total length.
    cum = direct.geometry.length.cumsum()
    direct = direct[cum <= road_len] if road_len > 0 else direct.iloc[:0]
    timings["clearance"] += time.time() - t

    t = time.time()
    perm_prop = permeability(recipient, moved)
    perm_direct = permeability(recipient, direct)
    timings["permeability"] += time.time() - t

    return {
        "recipient": recipient.block_id,
        "donor": donor.block_id,
        "donor_type": "osm_footpaths",
        "real_gw_dist": float(dist),
        "feature_dist": float(np.linalg.norm(signature(d_xy) - signature(r_xy))),
        "perm_gap": float(perm_prop - perm_direct),
        "perm_proposal": float(perm_prop),
        "perm_direct": float(perm_direct),
        "displacement_proposal": displacement_fraction(recipient, moved),
        "displacement_direct": displacement_fraction(recipient, direct),
        "road_len_m": road_len,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--exclusion-radius-m", type=float, default=2000.0)
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/gw_pair_matrix.parquet"))
    args = ap.parse_args()

    # Build the recipient/donor pools from the Task 7 shortlist. Donor eligibility comes from
    # exclusion_holdout so no donor is a near-neighbour of its recipient (see Task 6 for why
    # leave-one-block-out leaks).
    recipients, donors, donor_lines, blocks_gdf = load_pools()   # implement against your shortlist

    timings = {"gw": 0.0, "transplant": 0.0, "clearance": 0.0, "permeability": 0.0}
    rows: list[dict[str, object]] = []
    t0 = time.time()

    for i, recipient in enumerate(recipients):
        eligible = exclusion_holdout(blocks_gdf, i, radius_m=args.exclusion_radius_m)
        for j in eligible:
            if len(rows) >= args.pairs:
                break
            rows.append(score_pair(recipient, donors[j], donor_lines[j], timings))
        if len(rows) >= args.pairs:
            break

    elapsed = time.time() - t0
    print(f"\n{len(rows)} pairs in {elapsed:.0f}s -- {elapsed/max(len(rows),1):.1f}s/pair")
    for stage, secs in timings.items():
        print(f"  {stage:14s} {secs:7.1f}s  ({secs/max(elapsed,1e-9)*100:.0f}%)")

    if args.timing_only:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}")
```

`load_pools()` is the one piece that depends on how Task 7's shortlist landed on disk — implement
it to return `(recipient Blocks, donor Blocks, donor linework, a GeoDataFrame of donor geometries
in the same order)` from the provisioned shortlist. Then run:

```bash
pixi run python scripts/pair_matrix.py --pairs 20 --timing-only
```

Expected: a per-stage breakdown and a total per-pair second count. **Record it in the note.** If a pair costs more than ~90 s, reduce the matrix below 100 pairs rather than letting the run blow out.

- [ ] **Step 3: Run the ~100-pair matrix**

Recipients span the parcel-count range; donors span the GW-distance range. Donor eligibility comes from `exclusion_holdout`, so no donor is a near-neighbour of its recipient. Phase 1 populates `donor_type = "osm_footpaths"` only — the column exists so Phase 3 extends this same matrix rather than starting a new one.

```bash
pixi run python scripts/pair_matrix.py --pairs 100 --out data/benchmarks/gw_pair_matrix.parquet
```

- [ ] **Step 4: Report the three measurements**

No pre-committed decision rule — report and discuss:

1. The fidelity-vs-GW-distance relationship, and where (if anywhere) `perm_gap` crosses zero.
2. The pool-size → rank-1-distance exponent, measured by subsampling the pool at 10/30/100/300/1000 and fitting — **not** assumed from `N^(-1/d)`.
3. Per-pair wall clock.

- [ ] **Step 5: Commit the artifact and the note**

Write the findings note at `docs/superpowers/notes/<today>-gw-pair-matrix-findings.md` (use the
actual run date), covering the three measurements from Step 4 plus the per-pair cost.

```bash
mkdir -p data/benchmarks
git add data/benchmarks/gw_pair_matrix.parquet scripts/pair_matrix.py
git add docs/superpowers/notes/*-gw-pair-matrix-findings.md
git commit -m "feat: GW pair-matrix benchmark (100 pairs, osm_footpaths donors)

A retrieval benchmark any future featurization or donor material can be scored
against without re-solving anything. donor_type is a column so Phase 3 extends
this matrix rather than starting a new one. Committed as parquet rather than left
in scratchpad -- the earlier draft named scratchpad loss as a risk and then
guaranteed it by making the deliverable a note."
```

---

## Task Dependency Order

```
Task 1 (refactor) ──┬─► Task 2 (interiority_row) ──► Task 3 (shared tags) ──► Task 4 (PbfDesireLines) ──► Task 5 (census driver)
                    │                                                                                          │
                    │                                                                                          ▼
                    │                                                                            Task 7 (multi-tile provisioning)
                    │
                    └─► Task 8 (agreement primitives)          Task 6 (holdout) ──► Task 9 (pair matrix)
```

Tasks 6 and 8 are independent of the 1a chain and can run in parallel with it. Task 7 depends on Task 5 only because the shortlist it filters to is the census output. Task 9 depends on Task 6 for donor eligibility.

## What Phase 1 does NOT build

Stated so no task drifts into it: no FFT/masked-NCC retrieval (Phase 2 — and per the spec it is an O(N) scan, not an index), no shape-standardizing region builder (Phase 3 prerequisite), no patch quilting, no policy transplant, no supervised desire-line detection, and no reblocker of any kind. Phase 1 produces measurements and reusable artifacts.
