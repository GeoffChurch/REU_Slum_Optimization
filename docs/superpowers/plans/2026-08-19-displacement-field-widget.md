# DisplacementField Widget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Displacement page its first figure and its first widget — a live-draggable road over the real building field, with `Σcᵢ` computed exactly in the browser — and give the whole widget substrate real reflow.

**Architecture:** Displacement is exactly computable from point-to-segment distance alone (spec §1), so the widget computes the project's metric rather than reading a baked table. Python gains a `render_field` figure mode and a bake script; TypeScript gains a DOM-free metric module, a canvas draw, the widget, and one shared `ResizeObserver` that replaces both existing widgets' `window` resize listeners.

**Tech Stack:** Python 3.12 (numpy, geopandas, shapely, matplotlib), TypeScript (esbuild bundle + `tsc --noEmit`), `node:test`, MkDocs Material, pixi.

**Spec:** `docs/superpowers/specs/2026-08-19-displacement-field-widget-design.md`

## Global Constraints

- **`scripts/gen_site_pages.py` stays stdlib-only and MUST NEVER import `reblock`.** Every number it prints comes from an artifact already on disk.
- **Bundles and their `.d.ts` are generated and committed, never hand-edited.** A `.d.ts` is written by its own bake script from a `DTS_TEMPLATE` in that script.
- **The width slider floors at 7 m.** `PermeabilityParams.min_road_width_m = 7.0` (`src/reblock/permeability.py:125`) and `:205-209` raises below it. Range 7–20 m, step 0.5, default 7 — all baked, never TypeScript literals.
- **Widget-owned elements are sized with an inline style, never a presentation attribute.** Material ships `.md-typeset svg{height:auto;max-width:100%}` and presentation attributes lose that cascade. This cost D1 a Critical at its last gate.
- **No `viewBox`.** Spec §7 rejects it with the reason: it would scale 11 px axis labels to ~5 px at 320 px wide.
- **The fallback `<img>` is removed only after a successful first draw.** A widget that removes it and then fails leaves a blank figure and an honest-looking page.
- **No `# type: ignore`, no mypy excludes, no unreachable guards.** A default that cannot be reached is a silencer, not a defence.
- **Never reach into a closed, known-at-authoring-time set with a runtime string.** Frozen dataclass or named field, not `d["key"]`.
- **Every guard must be shown to fail before it counts.** Break it, observe red, restore. An injection that will not go red is reported, not tuned.
- **`import type` for every `.d.ts` import in widget code**; relative imports carry the `.js` extension.
- Run Python as `pixi run python -m scripts.<name>` — `pythonpath` is pytest-only.
- **`pixi run lint` is a gate on every task, not just the Python ones.** Task 1 shipped a
  101-column line green because its brief listed only pytest and typecheck. If a task's own gate
  list below omits it, it is still required.

## File Structure

**Python**

| file | responsibility |
|---|---|
| `src/reblock/render.py` | *modify:* add `render_field` — the disks/corridor figure, no choropleth |
| `scripts/gen_displacement_field.py` | *create:* the road rule, the PNG, `field.json`, `field.d.ts`, 5 parity fixtures |
| `scripts/gen_site_pages.py` | *modify:* `_displacement_field_figure()` + `MARKERS` entry + asset copy; delete `data-block` |
| `docs/_partials/displacement.md` | *modify:* the `<!-- DISPFIELD -->` marker, and the §8 prose correction |
| `tests/test_render.py` | *modify:* `render_field` layer/colour assertions |
| `tests/test_displacement_closed_form.py` | *create:* the closed-form identity, all 8 methods |
| `tests/test_displacement_field_bundle.py` | *create:* road determinism, schema, `.d.ts` equality, pixel parity |

**TypeScript**

| file | responsibility |
|---|---|
| `web/src/model/displacement.ts` | *create:* DOM-free metric — `corridorDistance`, `sumC` |
| `web/src/render/field.ts` | *create:* the canvas draw, spec §3's four layers in order |
| `web/src/widgets/displacement-field.ts` | *create:* handles, slider, toggle, readout |
| `web/src/field.d.ts` | *generated:* the bundle's type |
| `web/src/dom/resize.ts` | *create:* `observeSize` — the one `ResizeObserver` |
| `web/src/dom/fallback.ts` | *create:* `removeFallbackImage` — the `<img>` *and* its glightbox anchor |
| `web/src/widgets/perm-graph.ts`, `web/src/widgets/frontier.ts` | *modify:* adopt `observeSize`, `removeFallbackImage`; drop `window` resize |
| `web/src/mount.ts` | *modify:* register `displacement-field` |
| `web/test/displacement-model.test.ts` | *create:* Python↔TS parity against the baked fixtures |
| `web/test/resize.test.ts` | *create:* the fake observer, width sweep, zero-width skip |
| `web/test/field-boot.test.ts` | *create:* the widget boots, drags, and reports |
| `web/test/svg.test.ts`, `web/test/transform.test.ts` | *modify:* width-sweep containment; `fitBbox` minimum |

---

### Task 1: `render_field` — the figure that makes fallback parity possible

**Files:**
- Modify: `src/reblock/render.py` (add after `render_graph`, which ends at `:363`)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `budget.building_radii`, `budget.corridor_distance`, `render.py`'s `_CONTEXT_OUTLINE`, `_ROAD_COLOR`, `_DISPLACED_PT`, `_draw_boundary_and_streets`, `frame_bbox`, `BBox`
- Produces: `render_field(block, roads, radii, *, frame=None) -> Figure`, and `field_contributions(block, roads, radii) -> NDArray[np.float64]` — the per-building `cᵢ` the figure shades by, returned rather than kept private so a test can assert on the shading directly. **It is not baked into the bundle:** the widget computing `cᵢ` live is the entire point of the piece, and a baked copy of the boot state's values would be a second source of truth for a number the browser derives anyway.

- [ ] **Step 1: Write the failing test**

In `tests/test_render.py`:

```python
def test_render_field_draws_every_building_not_only_the_displaced_ones():
    """The point of the figure: a reader must be able to see that a road THREADED a gap, which
    means seeing the disks it missed. render_after draws only the displaced ones."""
    block = _tiny_block()                      # existing helper in this module
    roads = _one_road(block)
    radii = building_radii(block.building_points)
    fig = render_field(block, roads, radii)
    ax = fig.axes[0]
    # One PatchCollection per disk group; every building appears exactly once.
    disk_counts = [len(c.get_paths()) for c in ax.collections
                   if len(c.get_paths()) == len(block.building_points)]
    assert disk_counts, (
        f"no collection covers all {len(block.building_points)} buildings; "
        f"collection sizes were {[len(c.get_paths()) for c in ax.collections]}")


def test_render_field_shades_by_c_and_uses_the_named_constant():
    block = _tiny_block()
    roads = _one_road(block)
    radii = building_radii(block.building_points)
    c = field_contributions(block, roads, radii)
    fig = render_field(block, roads, radii)
    alphas = sorted({round(float(a), 6) for coll in fig.axes[0].collections
                     for a in np.atleast_1d(coll.get_alpha() or 1.0)})
    assert any(a not in (0.0, 1.0) for a in alphas), (
        "no partial alpha anywhere: the disks are not shaded by c")
    assert _DISPLACED_PT in {to_hex(col) for coll in fig.axes[0].collections
                             for col in np.atleast_2d(coll.get_facecolor())}, (
        f"the grazed disks must use the named constant {_DISPLACED_PT}, not an inline literal")
    assert 0.0 < c.max() <= 1.0


def test_render_field_never_fills_parcels():
    """Piece B's finding, restated: filling parcels states one quantity twice and drowns the
    subject. The parcel collection must be face-transparent."""
    block = _tiny_block()
    fig = render_field(block, _one_road(block), building_radii(block.building_points))
    parcel_faces = [coll.get_facecolor() for coll in fig.axes[0].collections
                    if len(coll.get_paths()) == len(block.parcels)]
    assert parcel_faces, "no collection matches the parcel count"
    assert all(np.asarray(f).size == 0 or float(np.asarray(f)[0][3]) == 0.0
               for f in parcel_faces), "parcels are filled"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_render.py -k render_field -v`
Expected: FAIL with `ImportError`/`NameError: render_field`.

- [ ] **Step 3: Implement**

Add to `src/reblock/render.py`, after `render_graph`:

```python
def field_contributions(block: Block, roads: gpd.GeoDataFrame,
                        radii: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-building displacement contribution `c_i = clip(1 - d_i/r_i, 0, 1)`, in
    `block.building_points` order.

    Returned rather than left private inside `render_field` so a test can assert on the shading
    without reading pixels. NOT baked into the widget's bundle: the widget derives `c` itself from
    the road position, which is what makes the road draggable at all.
    """
    from reblock.budget import corridor_distance   # deferred: budget imports render's siblings
    n = len(block.building_points)
    if n == 0 or roads is None or roads.empty:
        return np.zeros(n, dtype=np.float64)
    d = corridor_distance(block.building_points, roads)
    r = np.asarray(radii, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(r > 0.0, 1.0 - d / r, np.where(d <= 0.0, 1.0, 0.0))
    return np.clip(c, 0.0, 1.0).astype(np.float64)


def render_field(
    block: Block,
    roads: gpd.GeoDataFrame,
    radii: NDArray[np.float64],
    *,
    frame: BBox | None = None,
) -> Figure:
    """The displacement model, drawn literally: every building a disk of its own radius, the road
    corridor over it, each disk shaded by how much of it the corridor takes.

    Differs from `render_after` in the one way that matters for this page: no choropleth underneath,
    and EVERY building drawn, not only the displaced ones. `render_after` shades displaced disks at
    `alpha = c` on top of the depth fill (`_draw_heatmap`), so disk shading and parcel fill compete
    in the same pixels -- which makes it impossible for a widget drawing disks over a wireframe to
    match, and impossible for a reader to see that a road threaded a GAP rather than merely missing
    some homes. The gap is the subject: `c` clips to exactly 0 at `d = r`, so a road in a gap is
    free, and only the disks it missed show that.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    block.parcels.plot(ax=ax, facecolor="none", edgecolor=_CONTEXT_OUTLINE, linewidth=0.4)

    view = frame if frame is not None else frame_bbox(block.parcels)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # Dissolve per width group BEFORE buffering -- render.py:304-319's rule, and load-bearing here
    # rather than merely tidy: this figure exists to show overlapping corridors as CHEAP, and a
    # translucent patch per road compounds toward opaque exactly where they overlap, drawing the
    # opposite of the claim.
    if roads is not None and not roads.empty:
        road_w = roads["width_m"].to_numpy(dtype=float)
        corridor = gpd.GeoDataFrame(
            geometry=[unary_union(list(roads.geometry[road_w == w])).buffer(float(w) / 2.0)
                      for w in np.unique(road_w)],
            crs=block.crs)
        corridor.plot(ax=ax, color=_ROAD_COLOR, alpha=0.25, zorder=2, linewidth=0)

    _draw_boundary_and_streets(ax, block)

    # Every building as its own disk. Two collections: the ones the corridor reaches, filled at
    # alpha = c, and the ones it does not, as a thin outline. `_DISPLACED_PT` rather than
    # render_after's inline `(1.0, 0.0, 0.0, c)` -- a named constant is a thing the bake can put in
    # the widget's bundle, where a literal in a function body would have to be retyped in
    # TypeScript and could then drift.
    c = field_contributions(block, roads, radii)
    disks = gpd.GeoDataFrame(
        geometry=block.building_points.geometry.buffer(np.asarray(radii, dtype=np.float64)),
        crs=block.crs)
    grazed = c > 0.0
    if (~grazed).any():
        disks[~grazed].plot(ax=ax, facecolor="none", edgecolor=_DISPLACED_PT,
                            linewidth=0.5, zorder=5)
    if grazed.any():
        rgba = to_rgba(_DISPLACED_PT)
        disks[grazed].plot(ax=ax, color=[(*rgba[:3], float(ci)) for ci in c[grazed]],
                           linewidth=0, zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig
```

Add `from matplotlib.colors import to_rgba` to the imports.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_render.py -v`, then `pixi run typecheck` and `pixi run lint`.
Expected: PASS, and no existing `render_before`/`render_after`/`render_graph` test regresses.

- [ ] **Step 5: Fault-inject each new guard**

For each of the three tests: make the change it forbids (draw only `disks[grazed]`; use `(1,0,0,c)` instead of `_DISPLACED_PT`; fill parcels with a colour), run, **confirm red**, restore. Report any injection that stays green rather than adjusting the test to suit.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/render.py tests/test_render.py
git commit -m "feat: render_field -- the displacement model drawn literally, no choropleth"
```

---

### Task 2: The road rule, and the closed form pinned in Python

**Files:**
- Create: `scripts/_default_road.py`
- Create: `tests/test_displacement_closed_form.py`

**Interfaces:**
- Consumes: `scripts._example_block.load_example_block`, `reblock.budget.{building_radii, displacement, displacement_from_distance}`
- Produces: `default_roads(block, width_m) -> GeoDataFrame` (two rows, `geometry` + `width_m`); `segments(roads) -> NDArray` of shape `(k, 5)` as `(x0, y0, x1, y1, half_width)`; `closed_form_distance(px, py, segs) -> NDArray` — the exact reference the TypeScript mirrors

A separate module from `gen_displacement_field.py` because Task 3's bake and this task's tests both need it, and because `_example_block.py` set the precedent: a thing two callers need is declared once.

- [ ] **Step 1: Write the failing test**

`tests/test_displacement_closed_form.py`:

```python
"""The identity the TypeScript widget implements, pinned in the language where ground truth lives.

    dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)

A buffer IS the set of points within w/2 of the line, and distance to a union is the minimum over
its parts, so this is exact -- the residual against shapely is shapely's own inscribed-polygon
discretisation, which is why the closed form comes out HIGHER every time. Measured worst case over
the eight methods on the pinned block: 4.4e-04 relative.

If someone changes `corridor_distance`, this fails HERE, not in a browser nobody re-runs.
"""
import numpy as np
import pytest

from reblock.budget import building_radii, displacement, displacement_from_distance
from scripts._default_road import closed_form_distance, segments
from scripts._example_block import load_example_block

TOL = 1e-3   # 2.3x the measured worst case (4.4e-04), which is shapely's error, not ours


@pytest.fixture(scope="module")
def pinned():
    block, roads_by_method = load_example_block(None)
    return block, roads_by_method, building_radii(block.building_points)


def test_closed_form_matches_shapely_for_every_method(pinned):
    block, roads_by_method, radii = pinned
    px = block.building_points.geometry.x.to_numpy()
    py = block.building_points.geometry.y.to_numpy()
    assert len(roads_by_method) == 8, f"expected the eight example methods, got {sorted(roads_by_method)}"
    for name, roads in sorted(roads_by_method.items()):
        truth = displacement(block.building_points, radii, roads)
        mine = displacement_from_distance(
            radii, closed_form_distance(px, py, segments(roads)))
        assert truth > 0, f"{name} displaces nothing, so this comparison proves nothing"
        assert abs(mine - truth) / truth < TOL, f"{name}: closed {mine} vs shapely {truth}"


def test_the_closed_form_is_the_higher_of_the_two(pinned):
    """Direction matters: shapely's buffer is an INSCRIBED polygon, so it is slightly small, so it
    reports slightly larger distances and slightly smaller c. A closed form that came out LOWER
    would mean the formula is wrong rather than shapely being discrete."""
    block, roads_by_method, radii = pinned
    px = block.building_points.geometry.x.to_numpy()
    py = block.building_points.geometry.y.to_numpy()
    for name, roads in sorted(roads_by_method.items()):
        truth = displacement(block.building_points, radii, roads)
        mine = displacement_from_distance(
            radii, closed_form_distance(px, py, segments(roads)))
        assert mine >= truth, f"{name}: closed form {mine} below shapely {truth}"


def test_default_roads_are_deterministic_and_inside_the_block(pinned):
    from scripts._default_road import default_roads
    block, _, _ = pinned
    a = default_roads(block, 7.0)
    b = default_roads(block, 7.0)
    assert len(a) == 2
    assert a.geometry.iloc[0].equals(b.geometry.iloc[0]), "road 1 is not reproducible"
    assert a.geometry.iloc[1].equals(b.geometry.iloc[1]), "road 2 is not reproducible"
    hull = block.parcels.union_all()
    for g in a.geometry:
        assert g.length > 0
        assert hull.buffer(1e-6).contains(g), "a default road leaves the block"
    assert not (a.geometry.iloc[0].buffer(3.5)
                .intersects(a.geometry.iloc[1].buffer(3.5))), (
        "the two default corridors already overlap, so merging them is not something the reader does")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_displacement_closed_form.py -v`
Expected: FAIL, `ModuleNotFoundError: scripts._default_road`.

- [ ] **Step 3: Implement**

`scripts/_default_road.py`:

```python
"""The two default roads, and the closed-form corridor distance the widget implements.

Declared once because three callers need it: the bake (scripts/gen_displacement_field.py), the
identity test (tests/test_displacement_closed_form.py), and the fixture generator inside the bake.
`scripts/_example_block.py` set this precedent -- when each caller declared its own copy, changing
one left the others describing something else while every test still passed.
"""
from __future__ import annotations

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from reblock.contracts import Block


def default_roads(block: Block, width_m: float) -> GeoDataFrame:
    """Two straight roads, derived by rule so the PNG, the bundle and the caption agree.

    Road 1 runs along the building field's PRINCIPAL AXIS through its centroid, clipped to the
    block. Road 2 is road 1 shifted perpendicular by `3 * width_m` -- far enough that the two
    corridors start disjoint, so merging them is something the reader DOES rather than something
    they arrive to find already done.

    This is a REFERENCE LINE, not a discovered structural axis, and the docstring must not imply
    otherwise: measured on the pinned block the field is nearly isotropic (singular values 567.4 and
    523.0, anisotropy 1.085), so there is no meaningful "long axis" of this settlement to follow.

    Do not "improve" this to the convex-hull diameter or the longest interior chord. Both were
    measured and both are far WORSE conditioned: the hull diameter beats its runner-up pair by 0.07%
    (161.19 m against 161.07 m) and swings 3.28 degrees under 10 cm of coordinate jitter, where the
    principal axis swings 0.23 degrees -- because it averages 263 points while a diameter is decided
    by exactly two extreme vertices. The two alternatives also agree with each other to 0.0 degrees
    here, so they are one idea, not two.

    A rule rather than a hand-placed line: the widget's boot state and the committed PNG have to be
    the same road for fallback parity to mean anything, and the caption's numbers have to be
    measurements of it.
    """
    pts = block.building_points
    xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    centre = xy.mean(axis=0)
    # First principal component. `np.linalg.svd` on the centred cloud; the sign of a singular
    # vector is arbitrary, so normalise it -- otherwise the "deterministic" road flips between runs
    # of the same code on the same data.
    _, _, vt = np.linalg.svd(xy - centre, full_matrices=False)
    axis = vt[0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])

    hull = block.parcels.union_all()
    return GeoDataFrame(
        {"width_m": [float(width_m), float(width_m)]},
        geometry=[_chord(hull, centre, axis),
                  _chord(hull, centre + normal * (3.0 * float(width_m)), axis)],
        crs=block.crs)


def _chord(hull: BaseGeometry, through: NDArray[np.float64],
           direction: NDArray[np.float64]) -> LineString:
    """The longest piece of the infinite line `through + t*direction` that lies inside `hull`.

    Longest, not first: a concave block cuts the line into several pieces and only the longest is
    the road a reader would recognise as crossing the settlement.
    """
    span = float(np.hypot(*(np.asarray(hull.bounds[2:]) - np.asarray(hull.bounds[:2])))) * 2.0
    line = LineString([through - direction * span, through + direction * span])
    inside = line.intersection(hull)
    parts = list(inside.geoms) if inside.geom_type.startswith("Multi") else [inside]
    longest = max(parts, key=lambda g: g.length)
    return LineString([longest.coords[0], longest.coords[-1]])


def segments(roads: GeoDataFrame) -> NDArray[np.float64]:
    """Every road flattened to `(x0, y0, x1, y1, half_width)` -- exactly what the widget receives.

    Flattening here rather than in the bake means the identity test and the widget consume the same
    shape, so a parity failure is a failure of the FORMULA and never of two different flattenings.
    """
    out: list[tuple[float, float, float, float, float]] = []
    for geom, w in zip(roads.geometry, roads["width_m"].to_numpy(dtype=float), strict=True):
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            coords = np.asarray(part.coords, dtype=np.float64)
            for a, b in zip(coords[:-1], coords[1:], strict=True):
                out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(w) / 2.0))
    return np.asarray(out, dtype=np.float64).reshape(-1, 5)


def closed_form_distance(px: NDArray[np.float64], py: NDArray[np.float64],
                         segs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-point distance to the corridor, without ever building the corridor.

        dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)

    This is the reference `web/src/model/displacement.ts` mirrors line for line. Kept in numpy here
    and per-point there; same arithmetic.
    """
    if len(segs) == 0:
        return np.full(len(px), np.inf)
    x0, y0, x1, y1, hw = segs.T
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = ((px[:, None] - x0) * dx + (py[:, None] - y0) * dy) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)      # a zero-length road is its own endpoint
    d = np.hypot(px[:, None] - (x0 + t * dx), py[:, None] - (y0 + t * dy)) - hw
    return np.maximum(0.0, d).min(axis=1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_displacement_closed_form.py -v`, then `pixi run typecheck` and
`pixi run lint`.
Expected: PASS. First run loads the pinned block (minutes if the derivation cache is cold).

- [ ] **Step 5: Fault-inject**

Drop the `- hw` from `closed_form_distance` → the per-method test must go red. Remove the sign normalisation in `default_roads` → the determinism test will *not* reliably go red (SVD is deterministic for fixed input), so instead assert the normalisation directly: negate `axis` unconditionally and confirm the "corridors do not already overlap" or containment assertion moves. **If neither injection reddens, say so** — the sign normalisation may be unfalsifiable by test, in which case it stays as a comment-documented invariant and the plan's claim about it is wrong.

- [ ] **Step 6: Commit**

```bash
git add scripts/_default_road.py tests/test_displacement_closed_form.py
git commit -m "feat: the default road rule, and the closed form pinned against shapely"
```

---

### Task 3: The bake — one PNG, one bundle, one `.d.ts`, five fixtures

**Files:**
- Create: `scripts/_bundle_io.py`
- Modify: `scripts/gen_web_bundle.py` (delete its private quantisers, import the shared ones)
- Create: `scripts/gen_displacement_field.py`
- Create: `examples/displacement-field/field.png`, `examples/displacement-field/field.json` (generated, committed)
- Create: `web/src/field.d.ts` (generated, committed)
- Test: `tests/test_displacement_field_bundle.py`

**Interfaces:**
- Consumes: `scripts._default_road.{default_roads, segments}`, `reblock.render.{render_field, field_contributions, save_render}`, `reblock.budget.{building_radii, displacement}`
- Produces: `field.json` per spec §4; `FieldBundle` in `web/src/field.d.ts`; `REFERENCE_CASES` — the five named fixtures

**Why the quantisers move.** Spec §4 says this bake reuses `gen_web_bundle.py`'s `_r`/`_c`/`_line_coords` verbatim. They are private to that module, so the choice is import-a-private, copy them, or extract them. Copying is how the coordinate-precision trap gets re-introduced (6 significant figures on a 6,240,000 northing quantises to 10 m). Extract, and delete the originals — no aliases left behind.

- [ ] **Step 1: Write the failing test**

`tests/test_displacement_field_bundle.py`:

```python
"""The committed field bundle, and whether it still describes what Python computes."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path("examples/displacement-field/field.json")
PNG = Path("examples/displacement-field/field.png")
DTS = Path("web/src/field.d.ts")


@pytest.fixture(scope="module")
def bundle():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_the_committed_dts_is_what_the_generator_writes():
    """Piece C left this open: nothing asserted the committed .d.ts equalled the generator's own
    template, so a hand edit was caught only for keys the recursive guard happened to walk."""
    from scripts.gen_displacement_field import DTS_TEMPLATE
    assert DTS.read_text(encoding="utf-8") == DTS_TEMPLATE, (
        "web/src/field.d.ts was hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_displacement_field")


def test_every_declared_field_is_present_and_the_shapes_agree(bundle):
    b = bundle
    n = b["n_buildings"]
    assert n > 0
    for key in ("x", "y", "r"):
        assert len(b["buildings"][key]) == n, f"buildings.{key} has {len(b['buildings'][key])} of {n}"
    assert len(b["roads"]) == 2, "two default roads (spec §2)"
    assert all(len(r["coords"]) == 2 for r in b["roads"]), "the default roads are straight segments"
    assert b["width"]["floor_m"] == 7.0, (
        "the slider floor must be min_road_width_m: permeability.py:205 RAISES below it")
    assert b["width"]["default_m"] >= b["width"]["floor_m"]
    assert b["width"]["max_m"] > b["width"]["default_m"]
    assert len(b["reference"]) == 6, "six parity fixtures (spec §6)"


def test_the_reference_fixtures_cover_the_cases_that_could_hide_a_bug(bundle):
    """A fixture set that is five variations of the same road proves one thing five times."""
    cases = {c["name"]: c for c in bundle["reference"]}
    assert set(cases) == {"road1", "apart", "coincident", "widest", "in_a_gap", "outside"}, sorted(cases)
    assert cases["outside"]["sum_c"] == 0.0, (
        "the outside-the-block fixture must be EXACTLY zero -- it is the only fixture that pins the "
        "clip at d = r rather than a tolerance")
    # Overlap is free, and the honest form of that is an EQUALITY, not an inequality: a road drawn
    # twice IS one road, because each road is buffered on its own and only then unioned. Measured:
    # both 32.0260. Any implementation that charges per-road instead of per-union breaks this
    # immediately, where "coincident < apart" would still pass.
    assert cases["coincident"]["sum_c"] == cases["road1"]["sum_c"], (
        "two coincident roads must cost EXACTLY what one costs")
    assert cases["apart"]["sum_c"] > cases["road1"]["sum_c"], "adding a disjoint road adds cost"
    # Width, isolated: same road, 20 m against 7 m. Measured 68.1581 against 32.0260.
    assert cases["widest"]["sum_c"] > cases["road1"]["sum_c"]
    # Position, isolated: same width, near-identical length (144.3 m against 143.7 m), through the
    # field's widest gap. Measured 21.8465 against 32.0260. NOT zero -- see _cases' docstring.
    assert 0.0 < cases["in_a_gap"]["sum_c"] < cases["road1"]["sum_c"]


def test_the_bundle_still_matches_what_python_computes_now(bundle):
    """The bundle is committed, so it can go stale against the code that made it. Recompute one
    fixture from the pinned block and compare."""
    from reblock.budget import building_radii, displacement
    from scripts._example_block import load_example_block
    from scripts.gen_displacement_field import roads_from_case
    block, _ = load_example_block(None)
    radii = building_radii(block.building_points)
    case = next(c for c in bundle["reference"] if c["name"] == "apart")
    recomputed = displacement(block.building_points, radii,
                              roads_from_case(block, case, tuple(bundle["origin"])))
    assert abs(recomputed - case["sum_c"]) < 1e-6, (
        f"the committed bundle says {case['sum_c']}, the code now computes {recomputed}; "
        "regenerate: pixi run python -m scripts.gen_displacement_field")


def test_coordinates_are_relative_to_the_origin_and_not_significant_figure_rounded(bundle):
    """The coordinate-precision trap: 6 significant figures on a ~6,240,000 UTM northing quantises
    to 10 m, which dissolves the parcel geometry."""
    b = bundle
    assert len(b["origin"]) == 2
    assert abs(b["origin"][1]) > 1e6, "the origin should be the real UTM offset"
    ys = [y for ring in b["parcels"] for _, y in ring]
    assert max(abs(y) for y in ys) < 1e4, "coordinates are not relative to origin"
    # Centimetre precision means at least some coordinate has a non-zero second decimal.
    assert any(round(y, 2) != round(y, 1) for y in ys), (
        "every coordinate is decimetre-round: these look significant-figure rounded, not _c'd")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_displacement_field_bundle.py -v`
Expected: FAIL — `FileNotFoundError` on the bundle and `ModuleNotFoundError` on the generator.

- [ ] **Step 3: Extract the shared quantisers**

Create `scripts/_bundle_io.py` holding `SIGFIGS = 6`, `sigfig(x)`, `cm(x)`, `line_coords(geom, ox, oy)` — bodies and docstrings moved verbatim from `gen_web_bundle.py:58-92`, renamed off the underscore since they are now a shared surface. Then in `gen_web_bundle.py`: delete `_r`, `_c`, `_line_coords` and `SIGFIGS`, import the new names, and update every call site. **No aliases** — `_r = sigfig` would be exactly the compatibility shim the directives forbid.

Run `pixi run pytest tests/test_web_bundle.py -v` and `pixi run python -m scripts.gen_web_bundle`; the regenerated `bundle.json` must be **byte-identical** to the committed one (`git diff --stat` shows nothing). If it differs, the extraction changed behaviour and that is a bug in the extraction, not a new baseline.

- [ ] **Step 4: Write the generator**

`scripts/gen_displacement_field.py`:

```python
"""Bake the Displacement page's field figure and its widget bundle.

One PNG (the widget's BOOT state -- road 1 alone at 7 m, since road 2 defaults off) plus a
self-contained JSON payload and its generated .d.ts.

Self-contained rather than reading examples/perm-graph/bundle.json, even though both widgets sit on
the same block and that bundle already holds the parcels, boundary, streets and origin: sharing one
payload couples two widgets so that retuning one silently changes the other. The generator CODE is
shared instead (scripts/_bundle_io.py, scripts/_default_road.py, scripts/_example_block.py), which
makes data drift impossible while leaving each widget its own artifact.

Run:  pixi run python -m scripts.gen_displacement_field
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from geopandas import GeoDataFrame
from matplotlib.colors import to_hex
from shapely.affinity import translate
from shapely.geometry import LineString

from reblock.budget import building_radii, displacement
from reblock.contracts import Block
from reblock.render import (
    _BOUNDARY_COLOR,
    _CONTEXT_OUTLINE,
    _DISPLACED_PT,
    _ROAD_COLOR,
    render_field,
    save_render,
)
from scripts._bundle_io import cm, line_coords, sigfig
from scripts._default_road import default_roads
from scripts._example_block import load_example_block

OUT = Path("examples/displacement-field")
DTS = Path("web/src/field.d.ts")

WIDTH_FLOOR_M = 7.0    # == PermeabilityParams.min_road_width_m; :205-209 RAISES below it
WIDTH_MAX_M = 20.0
WIDTH_STEP_M = 0.5

# What the widget draws with, taken from the constants render_field itself uses, so the PNG and the
# canvas cannot drift: a reader with JS off and a reader with JS on must see the same figure. The
# last two have no PNG equivalent -- they are the web figure's own affordances.
ENCODING = {
    "parcel_color": _CONTEXT_OUTLINE,
    "parcel_lw": 0.4,
    "boundary_color": _BOUNDARY_COLOR,
    "boundary_lw": 1.3,
    "street_lw": 1.0,
    "road_color": _ROAD_COLOR,
    "road_alpha": 0.25,
    "disk_color": _DISPLACED_PT,
    "disk_outline_lw": 0.5,
    "handle_radius_px": 7.0,
    "pad": 0.04,
}


def roads_from_case(block: Block, case: ReferenceCase,
                    origin: tuple[float, float]) -> GeoDataFrame:
    """Rebuild a reference fixture's road set from the bundle's own numbers.

    Exists so tests/test_displacement_field_bundle.py can recompute a fixture against live code
    without re-deriving how a case is defined -- the bundle is the single description of it.

    A fixture's coordinates are ORIGIN-RELATIVE, like every other coordinate in the bundle, so the
    origin has to come back in here.

    `ReferenceCase` is a `TypedDict`, not a frozen dataclass: this value arrives through
    `json.loads`, and the directives' own carve-out is exactly that -- dict-ness forced by an
    external interface. A dataclass here would mean every caller converting at the boundary for a
    checker benefit a TypedDict already gives.
    """
    ox, oy = origin
    return GeoDataFrame(
        {"width_m": [float(r["width_m"]) for r in case["roads"]]},
        geometry=[LineString([(x + ox, y + oy) for x, y in r["coords"]]) for r in case["roads"]],
        crs=block.crs)


def _cases(block: Block, roads: GeoDataFrame, ox: float, oy: float) -> list[dict[str, object]]:
    """The five parity fixtures. Chosen so no two of them could pass for the same reason:

    road1      -- road 1 alone at the floor width. The BASELINE, and the PNG's own number.
    apart      -- both default roads, corridors disjoint
    coincident -- road 2 moved onto road 1: costs EXACTLY what road1 costs
    widest     -- road 1 alone at WIDTH_MAX_M: isolates width
    in_a_gap   -- road 1's direction through the field's widest gap: isolates position
    outside    -- road 1 translated clear of the block: exactly zero

    Each isolates ONE of the page's claims against `road1`, which is why the baseline is in the set:
    without it, "in_a_gap is cheaper than apart" compares one road against two and demonstrates
    nothing about gaps. Measured on the pinned block -- road1 32.0260, apart 47.8488, coincident
    32.0260, widest 68.1581, in_a_gap 21.8465, outside 0.0.

    `in_a_gap` is NOT free, and do not chase a zero: the widest gap here has radius 6.95 m against a
    2.19 m median, so a 7 m road down its middle still leaves its two neighbours at d = 3.45 m
    against r = 6.95, i.e. c ~ 0.5 each. A chord across a block cannot stay outside every disk.
    `outside` is the fixture that pins the clip at d = r.
    """
    pts = block.building_points
    radii = building_radii(pts)
    bp = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    r1, r2 = roads.geometry.iloc[0], roads.geometry.iloc[1]
    diag = float(np.hypot(*(np.asarray(block.parcels.total_bounds[2:])
                            - np.asarray(block.parcels.total_bounds[:2]))))

    # The widest gap in the field IS the largest nearest-neighbour distance: that is what a large
    # radius means. Put the road through the midpoint of that pair, along road 1's direction.
    widest = int(np.argmax(radii))
    others = np.delete(np.arange(len(bp)), widest)
    partner = int(others[np.argmin(np.hypot(*(bp[others] - bp[widest]).T))])
    gap_mid = (bp[widest] + bp[partner]) / 2.0
    r1_dir = np.asarray(r1.coords[-1]) - np.asarray(r1.coords[0])
    r1_dir = r1_dir / np.hypot(*r1_dir)
    in_gap = LineString([gap_mid - r1_dir * diag, gap_mid + r1_dir * diag]).intersection(
        block.parcels.union_all())

    named: list[tuple[str, GeoDataFrame]] = [
        ("road1", _set(block, [r1], WIDTH_FLOOR_M)),
        ("apart", roads),
        ("coincident", _set(block, [r1, r1], WIDTH_FLOOR_M)),
        ("widest", _set(block, [r1], WIDTH_MAX_M)),
        ("in_a_gap", _set(block, [max(
            in_gap.geoms if in_gap.geom_type.startswith("Multi") else [in_gap],
            key=lambda g: g.length)], WIDTH_FLOOR_M)),
        ("outside", _set(block, [translate(r1, xoff=2.0 * diag, yoff=2.0 * diag)], WIDTH_FLOOR_M)),
    ]

    out = []
    for name, rs in named:
        total = displacement(pts, radii, rs)
        if name == "outside" and total != 0.0:
            raise AssertionError(
                f"the 'outside' fixture displaces {total}, so it is not outside the block -- it "
                f"would prove nothing about the clip at d = r. Fix the translation, do not bake it.")
        out.append({"name": name,
                    "roads": [{"coords": [[cm(x - ox), cm(y - oy)] for x, y in g.coords],
                               "width_m": float(w)}
                              for g, w in zip(rs.geometry, rs["width_m"], strict=True)],
                    "sum_c": sigfig(total),
                    "fraction": sigfig(total / len(pts))})
    return out


def _set(block: Block, geoms: list[LineString], width_m: float) -> GeoDataFrame:
    """A road set at one width. Every fixture is built through here so no fixture can accidentally
    carry a width the slider could not produce."""
    if not (WIDTH_FLOOR_M <= width_m <= WIDTH_MAX_M):
        raise ValueError(f"{width_m} m is outside the slider's own range")
    return GeoDataFrame({"width_m": [float(width_m)] * len(geoms)},
                        geometry=list(geoms), crs=block.crs)
```

- [ ] **Step 5: Finish the generator body**

`main()`:

1. `block, _ = load_example_block(None)`; `radii = building_radii(block.building_points)`
2. `roads = default_roads(block, WIDTH_FLOOR_M)`
3. `ox, oy = float(block.parcels.total_bounds[0]), float(block.parcels.total_bounds[1])` — the origin, same convention `gen_web_bundle` uses
4. PNG: `save_render(render_field(block, roads.iloc[[0]], radii), OUT / "field.png")` — **road 1 only**, the boot state
5. bundle: buildings (`cm` for x/y relative to origin, `sigfig` for r), `parcels` and `streets` via `line_coords`, `boundary` from `block.parcels.union_all().exterior`, `roads`, `width`, `ENCODING`, `reference` from `_cases`, `n_buildings`, `block_id` from `block.identity`
6. `(OUT / "field.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")`; `DTS.write_text(DTS_TEMPLATE, encoding="utf-8")`
7. print the five fixtures' `sum_c` so the caption numbers are visible in the run log

For `in_a_gap`, do not hand-pick a line: take the building pair with the largest nearest-neighbour distance (`radii.argmax()`), and run road 1's direction through the midpoint of that building and its nearest neighbour. Deterministic, and it is genuinely in a gap because that is what a large NN distance *is*.

For `outside`, translate road 1 by `2 × the block's diagonal` along its normal. Assert in the generator that the result is `0.0` exactly — if it is not, the fixture is not outside and the bake should fail rather than bake a fixture that proves nothing.

`DTS_TEMPLATE` follows `gen_web_bundle.py:107`'s shape: a header naming the generator and the regeneration command, then `export interface FieldBundle { … }` with a nested `Encoding`, `Width`, `Road` and `ReferenceCase`.

- [ ] **Step 6: Run the bake, then the tests**

```bash
pixi run python -m scripts.gen_displacement_field
pixi run pytest tests/test_displacement_field_bundle.py tests/test_web_bundle.py -v
pixi run typecheck && pixi run lint
```
Expected: PASS. `examples/displacement-field/field.json` under ~350 KB.

- [ ] **Step 7: Pixel parity between the figure and the bundle**

Spec §6.3. The widget draws from `ENCODING`; the PNG draws from the `render.py` constants
`ENCODING` was built out of. Nothing yet stops those diverging, and a divergence is invisible —
both figures still look like figures. Add to `tests/test_displacement_field_bundle.py`:

```python
def test_the_baked_colours_are_actually_in_the_committed_png(bundle):
    """D1's pattern: the widget's colours and the fallback image's colours must be the same
    colours, not two lists kept in step by hand. A reader with JS off and a reader with JS on are
    looking at the same figure or the page is lying to one of them."""
    from PIL import Image
    px = set(Image.open(PNG).convert("RGB").getdata())
    for key in ("disk_color", "road_color", "boundary_color"):
        want = bundle["encoding"][key]
        rgb = tuple(int(want[i:i + 2], 16) for i in (1, 3, 5))
        assert any(sum(abs(a - b) for a, b in zip(rgb, got)) <= 12 for got in px), (
            f"encoding.{key} = {want} appears nowhere in {PNG}: the widget and its own fallback "
            f"image are drawing different colours")
```

A tolerance of 12 summed over three channels absorbs the corridor's alpha 0.25 and the disks'
`alpha = c`, both of which composite the constant against white before it reaches a pixel. Confirm
the tolerance is doing real work: widen one constant by a single hex step and check the test still
passes, then change it to a clearly different hue and check it fails. If a *single hex step* fails,
the tolerance is too tight and the test will break on an unrelated matplotlib upgrade.

- [ ] **Step 8: Fault-inject**

Hand-edit one character of `web/src/field.d.ts` → the `.d.ts` test must go red. Bump one `sum_c` in the committed JSON → the recompute test must go red. Swap `cm` for `sigfig` on a coordinate → the coordinate-precision test must go red. Restore all three.

- [ ] **Step 9: Commit**

```bash
git add scripts/_bundle_io.py scripts/gen_web_bundle.py scripts/gen_displacement_field.py \
        examples/displacement-field web/src/field.d.ts tests/test_displacement_field_bundle.py
git commit -m "feat: bake the displacement field -- one PNG, one bundle, five parity fixtures"
```

---

### Task 4: The metric in TypeScript, and the parity that pins it

**Files:**
- Create: `web/src/model/displacement.ts`
- Create: `web/test/displacement-model.test.ts`

**Interfaces:**
- Consumes: `web/src/field.d.ts`'s `FieldBundle`, `ReferenceCase`
- Produces: `corridorDistance(px, py, segs) -> Float64Array`, `sumC(radii, d) -> number`, `flatten(roads) -> Segment[]`

DOM-free by design: the one piece of arithmetic that must agree with Python is testable with no fake DOM at all, and a test that needs a fake DOM to check a number is a test that can fail for a reason unrelated to the number.

- [ ] **Step 1: Write the failing test**

`web/test/displacement-model.test.ts`:

```ts
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { FieldBundle } from "../src/field.js";
import { corridorDistance, flatten, sumC } from "../src/model/displacement.js";

const bundle = JSON.parse(
  readFileSync("../examples/displacement-field/field.json", "utf8")) as FieldBundle;

// 1e-3 relative. The residual is SHAPELY's: its buffer is an inscribed polygon, so it reports
// slightly larger distances and slightly smaller c than the exact closed form this module
// implements. Measured worst case over the eight methods on this block is 4.4e-04
// (tests/test_displacement_closed_form.py pins both the magnitude and the direction).
const TOL = 1e-3;

test("every baked fixture's sum_c is reproduced from its own coordinates", () => {
  const { x, y, r } = bundle.buildings;
  assert.equal(bundle.reference.length, 6);
  for (const c of bundle.reference) {
    const got = sumC(r, corridorDistance(x, y, flatten(c.roads)));
    const rel = Math.abs(got - c.sum_c) / Math.max(c.sum_c, 1);
    assert.ok(rel < TOL, `${c.name}: TS ${got} vs Python ${c.sum_c} (rel ${rel})`);
  }
});

test("the outside-the-block fixture is exactly zero, not merely close", () => {
  const { x, y, r } = bundle.buildings;
  const outside = bundle.reference.find((c) => c.name === "outside")!;
  assert.strictEqual(sumC(r, corridorDistance(x, y, flatten(outside.roads))), 0,
    "c must clip to exactly 0 at d = r -- a tolerance here would hide a soft tail");
});

test("a road drawn twice costs exactly what one costs", () => {
  // The honest form of "overlap is free". Each road is buffered on its OWN width and only then
  // unioned, so two coincident roads occupy one corridor and are charged once -- an equality, not
  // a discount. A TypeScript port that summed per-road distances instead of minimising over
  // segments would pass a `coincident < apart` check and fail this one.
  const { x, y, r } = bundle.buildings;
  const cost = (name: string): number => {
    const c = bundle.reference.find((k) => k.name === name)!;
    return sumC(r, corridorDistance(x, y, flatten(c.roads)));
  };
  assert.equal(cost("coincident"), cost("road1"));
  assert.ok(cost("apart") > cost("road1"), "a disjoint second road must add cost");
});

test("a zero-length road is its own endpoint rather than a NaN", () => {
  const segs = flatten([{ coords: [[0, 0], [0, 0]], width_m: 7 }]);
  const d = corridorDistance([10], [0], segs);
  assert.ok(Number.isFinite(d[0]!), `degenerate road produced ${d[0]}`);
  assert.equal(d[0], 10 - 3.5);
});

test("no roads means no cost, not an empty-array minimum of Infinity leaking into sumC", () => {
  assert.strictEqual(sumC(bundle.buildings.r, corridorDistance(
    bundle.buildings.x, bundle.buildings.y, [])), 0);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd web && npm test` (or `pixi run web-test`)
Expected: FAIL — `Cannot find module '../src/model/displacement.js'`.

- [ ] **Step 3: Implement**

`web/src/model/displacement.ts`:

```ts
import type { Road } from "../field.js";

/** One road segment as the metric needs it: two endpoints and that road's own half-width. */
export interface Segment { x0: number; y0: number; x1: number; y1: number; hw: number }

/** Flatten roads to segments. Mirrors `scripts/_default_road.segments`, so a parity failure is a
 * failure of the FORMULA and never of two different flattenings. */
export function flatten(roads: readonly Road[]): Segment[] {
  const out: Segment[] = [];
  for (const road of roads) {
    const hw = road.width_m / 2;
    for (let i = 1; i < road.coords.length; i++) {
      const [x0, y0] = road.coords[i - 1]!;
      const [x1, y1] = road.coords[i]!;
      out.push({ x0, y0, x1, y1, hw });
    }
  }
  return out;
}

/** Per-building distance to the road corridor, without ever constructing the corridor.
 *
 *     dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)
 *
 * A buffer IS the set of points within w/2 of the line, and distance to a union is the minimum over
 * its parts -- so this is exact, and it is what lets this widget compute the project's real metric
 * on an arbitrary road position with no Pyodide and no geometry library. The reference
 * implementation is `scripts/_default_road.closed_form_distance`, and
 * `tests/test_displacement_closed_form.py` pins it against `budget.displacement` for all eight
 * methods.
 *
 * With no segments every distance is Infinity, which `sumC` turns into zero cost -- the same answer
 * `budget.displacement` gives for an empty road set.
 */
export function corridorDistance(px: readonly number[], py: readonly number[],
                                 segs: readonly Segment[]): Float64Array {
  const out = new Float64Array(px.length).fill(Infinity);
  for (let i = 0; i < px.length; i++) {
    const x = px[i]!, y = py[i]!;
    let best = Infinity;
    for (const s of segs) {
      const dx = s.x1 - s.x0, dy = s.y1 - s.y0;
      const l2 = dx * dx + dy * dy;
      // A zero-length road is its own endpoint. Without this, t is 0/0 and every distance is NaN --
      // and NaN propagates silently through Math.min to a readout of "NaN homes".
      const t = l2 > 0 ? Math.min(1, Math.max(0, ((x - s.x0) * dx + (y - s.y0) * dy) / l2)) : 0;
      const d = Math.hypot(x - (s.x0 + t * dx), y - (s.y0 + t * dy)) - s.hw;
      if (d < best) best = d;
    }
    out[i] = Math.max(0, best);
  }
  return out;
}

/** `Σ clip(1 - d_i/r_i, 0, 1)`. Mirrors `budget.displacement_from_distance`, including its r == 0
 * case: a coincident-points building counts iff the corridor actually touches it. */
export function sumC(radii: readonly number[], d: Float64Array): number {
  let total = 0;
  for (let i = 0; i < radii.length; i++) {
    const r = radii[i]!, di = d[i]!;
    const c = r > 0 ? 1 - di / r : (di <= 0 ? 1 : 0);
    total += Math.min(1, Math.max(0, c));
  }
  return total;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run web-test`, then `pixi run web-check`, then `pixi run lint`.
Expected: PASS, and `tsc --noEmit` clean under `noUncheckedIndexedAccess`.

- [ ] **Step 5: Fault-inject**

Four injections, each must redden: drop `- s.hw`; change `Math.max(0, best)` to `best`; drop the `l2 > 0` guard (the degenerate test); change `sumC`'s clip upper bound to `2`. Restore each.

- [ ] **Step 6: Commit**

```bash
git add web/src/model/displacement.ts web/test/displacement-model.test.ts
git commit -m "feat: displacement in TypeScript, pinned to Python by five baked fixtures"
```

---

### Task 5: Reflow, one fallback remover, and the riders

**Files:**
- Create: `web/src/dom/resize.ts`, `web/src/dom/fallback.ts`
- Modify: `web/src/widgets/perm-graph.ts`, `web/src/widgets/frontier.ts`
- Modify: `scripts/gen_site_pages.py` (delete `data-block`), `tests/test_gen_site_pages.py`
- Create: `web/test/resize.test.ts`
- Modify: `web/test/svg.test.ts` (width sweep), `web/test/transform.test.ts` (`fitBbox` minimum), `web/test/frontier-boot.test.ts` (fake observer)

**Interfaces:**
- Produces: `observeSize(el, onSize) -> () => void` where `onSize({width, height})` runs on every content-box change with a positive width; `removeFallbackImage(host) -> void`

**What is actually broken** (spec §7 — the backlog's "the real fix is a viewBox" is wrong twice over): `perm-graph.ts` already sets `cv.style.width = "100%"`, `frontier.ts` already measures its container, and both re-measure on `window` resize — so neither overflows at mount. The gap is a **container** resize with no window resize.

- [ ] **Step 1: Write the failing test**

`web/test/resize.test.ts`:

```ts
import assert from "node:assert/strict";
import test from "node:test";

/** A ResizeObserver whose callbacks the test fires by hand. Installed on globalThis before the
 * module under test is imported, the same order the other fake-DOM suites use. */
class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  readonly targets: unknown[] = [];
  constructor(private readonly cb: (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(el: unknown): void { this.targets.push(el); }
  disconnect(): void { FakeResizeObserver.live.splice(FakeResizeObserver.live.indexOf(this), 1); }
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}
(globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;

const { observeSize } = await import("../src/dom/resize.js");

test("a zero-width box does not call back, so nothing draws into a hidden container", () => {
  const seen: number[] = [];
  observeSize({} as HTMLElement, (s) => seen.push(s.width));
  FakeResizeObserver.live.at(-1)!.fire(0, 0);
  assert.deepEqual(seen, [], "drew at zero width -- the fallback image is still the honest picture");
  FakeResizeObserver.live.at(-1)!.fire(320, 200);
  assert.deepEqual(seen, [320], "did not draw once the container became visible");
});

test("every positive resize calls back, because a container can narrow without the window moving", () => {
  const seen: number[] = [];
  observeSize({} as HTMLElement, (s) => seen.push(s.width));
  const obs = FakeResizeObserver.live.at(-1)!;
  obs.fire(700, 400);
  obs.fire(320, 200);
  obs.fire(1200, 700);
  assert.deepEqual(seen, [700, 320, 1200]);
});

test("the disposer stops the callbacks", () => {
  const seen: number[] = [];
  const stop = observeSize({} as HTMLElement, (s) => seen.push(s.width));
  const obs = FakeResizeObserver.live.at(-1)!;
  obs.fire(700, 400);
  stop();
  assert.equal(FakeResizeObserver.live.includes(obs), false, "disconnect() was not called");
});
```

And in `web/test/svg.test.ts`, convert the containment assertion into a sweep:

```ts
// A label that escapes the plot rect only on a narrow screen is invisible to a single-width test,
// and 320 px is a real phone. The observer (dom/resize.ts) is what makes several widths reachable.
for (const width of [320, 700, 1200]) {
  test(`every axis label stays inside the viewport at ${width} px`, () => {
    // ... existing containment body, parameterised on `width`
  });
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run web-test`
Expected: FAIL — `Cannot find module '../src/dom/resize.js'`, and the parameterised containment test fails at 320 px if any label escapes (**if it passes at every width immediately, say so** — it means the gutters were already wide enough and the sweep is a regression guard rather than a bug find).

- [ ] **Step 3: Implement the two modules**

`web/src/dom/resize.ts`:

```ts
/** Re-run `onSize` whenever `el`'s content box changes to a positive width.
 *
 * Replaces `window.addEventListener("resize", ...)` in both widgets. A window listener misses the
 * case that actually breaks a figure: a CONTAINER narrowing with the window untouched -- Material's
 * nav drawer at some breakpoints, a <details> opening, a tab panel switching, print. There,
 * Frontier's absolute-pixel SVG overflows and PermGraph's canvas stretches a stale backing store.
 *
 * Deliberately NOT a viewBox, which was the recorded plan: a viewBox scales text with the box, so
 * Frontier's 11 px axis labels would land at ~5 px on a 320 px screen. Re-laying out at the
 * measured width keeps type at its designed size and re-nices the ticks for the narrower span.
 *
 * A zero width means "not laid out yet" (a hidden container, a collapsed tab), so it is SKIPPED
 * rather than drawn or thrown on. Skipping is only safe because both widgets now remove their
 * fallback <img> after a successful draw: nothing is drawn, so the static figure is still there.
 */
export function observeSize(el: HTMLElement,
                            onSize: (size: { width: number; height: number }) => void): () => void {
  const obs = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect;
      if (width > 0) onSize({ width, height });
    }
  });
  obs.observe(el);
  return () => obs.disconnect();
}
```

`web/src/dom/fallback.ts`:

```ts
/** Remove the static fallback image, and the link mkdocs-glightbox wrapped it in.
 *
 * `mkdocs-glightbox` wraps every figure image in `<a class="glightbox" href="...png">`. Removing
 * only the <img> leaves an empty anchor: invisible, but focusable, and announced by a screen reader
 * as a link with no text -- which cuts against the accessibility rationale for drawing SVG in the
 * first place. PermGraph ships that on the live site today.
 *
 * The anchor goes only when the image was its only element child: an anchor with other content is
 * somebody else's, and removing it would be a different bug.
 */
export function removeFallbackImage(host: HTMLElement): void {
  const img = host.querySelector("img");
  if (!img) return;
  const anchor = img.parentElement;
  img.remove();
  if (anchor && anchor.tagName === "A" && anchor.children.length === 0
      && (anchor.textContent ?? "").trim() === "") {
    anchor.remove();
  }
}
```

- [ ] **Step 4: Retrofit both widgets**

`frontier.ts`: delete `measure`'s throw and the `window` resize listener; call `observeSize(chartHost, (size) => { view = fitAxes(...); render(); })`, and remove the fallback inside the first successful callback via `removeFallbackImage(host)` (guarded by a `drawn` flag so it runs once). `measure` itself goes — the observer supplies the width.

`perm-graph.ts`: same swap, and **move the fallback removal to after the first successful draw**. It currently removes the `<img>` right after inserting the canvas, so a zero-width mount would leave a blank figure with the static image already gone.

Both widgets' `showWidgetError` paths stay exactly as they are.

- [ ] **Step 5: The riders**

- `web/test/transform.test.ts`: strengthen `fitBbox`'s uniformity test to assert the scale equals `Math.min(width / bw, height / bh)` scaled by the pad, not merely that `scaleX === scaleY` — a `Math.max`-for-`Math.min` regression is currently green there.
- `scripts/gen_site_pages.py`: delete `data-block` from every mount point it emits, and the assertions in `tests/test_gen_site_pages.py` that expect it. Every bundle carries `block_id`; a second source of one fact is drift waiting to happen. Confirm with `grep -rn "data-block\|dataset.block" web/ scripts/ tests/ docs/` that nothing reads it.

- [ ] **Step 6: Guard the shipped bundle against going stale**

`docs/js/widgets.js` is committed, and **nothing asserts it matches `web/src`.**
`tests/test_gen_site_pages.py:308` checks only that it *exists*, and
`web/test/widgets-bundle.test.ts` evaluates the committed artifact — so a source change that was
never rebuilt leaves a stale bundle that passes every gate, including the artifact test, which is
looking at the stale file and finding it fine. This task edits both shipped widgets, so it is the
task that would ship that.

Add to `web/test/widgets-bundle.test.ts`: rebuild with esbuild into a temp path and assert the
bytes equal the committed `docs/js/widgets.js`. Fault-inject by editing one character of
`web/src/mount.ts` without rebuilding — it must go red.

If esbuild output turns out not to be byte-reproducible across runs, **report that** rather than
weakening the assertion to a substring check: a "the bundle is current" test that passes on a stale
bundle is worse than no test, because its green tick is what stops anyone looking.

- [ ] **Step 7: Run the gates**

```bash
pixi run web && pixi run web-test && pixi run web-check && pixi run lint \
  && pixi run test-py -k "site_pages or web_bundle"
```

`pixi run web` **first**, and commit the rebuilt `docs/js/widgets.js` with this task. Every task that
touches `web/src/**` does this.

- [ ] **Step 8: Fault-inject**

Remove the `width > 0` guard → the zero-width test reddens. Remove `disconnect()` → the disposer test reddens. Restore the `<img>`-only removal in `fallback.ts` (leave the anchor) → add/confirm an assertion in `frontier-boot.test.ts` that no `<a>` survives, and confirm it reddens. Re-add `data-block` → the generator test must be *green* (it no longer asserts it) but `grep` must find one unread attribute; note this rider has no test guarding its absence and say so.

- [ ] **Step 9: Commit**

```bash
git add web/src/dom/resize.ts web/src/dom/fallback.ts web/src/widgets/ web/test/ docs/js/widgets.js \
        scripts/gen_site_pages.py tests/test_gen_site_pages.py
git commit -m "fix: reflow by re-render, not by viewBox -- one ResizeObserver for both widgets"
```

---

### Task 6: The widget

**Files:**
- Create: `web/src/render/field.ts`, `web/src/widgets/displacement-field.ts`
- Modify: `web/src/mount.ts` (register `displacement-field`)
- Create: `web/test/field-boot.test.ts`
- Modify: `web/test/widgets-bundle.test.ts` (its name-derivation test picks the new widget up for free once the generator emits it — confirm, do not edit, unless it fails)

**Interfaces:**
- Consumes: `model/displacement.{corridorDistance, sumC, flatten}`, `render/canvas.sizeCanvas`, `view/transform.{fitBbox, toWorld}`, `dom/resize.observeSize`, `dom/fallback.removeFallbackImage`, `dom/error.showWidgetError`, `field.d.ts`
- Produces: `drawField(ctx, b, frame, size)` with `frame = { view, roads, c }`; `displacementField: Widget`

**Registration:** in `mount.ts`, beside the other two, **after** `REGISTRY` exists. `displacement-field.ts` must import `Widget` with `import type` only — a runtime import of `mount.js` from a widget is the circular import that made the whole bundle throw on load in piece C.

- [ ] **Step 1: Write the failing test**

`web/test/field-boot.test.ts` — same fake-DOM idiom as `frontier-boot.test.ts` (stub `document` and a `FakeResizeObserver` on `globalThis` **before** importing the widget). Four module-level helpers, written once at the top of the file: `mountPoint()` returns a fake `<figure>` holding an `<img>` inside a glightbox `<a>`; `fireResize(w, h)` fires the live fake observer; `makeState` is the same trivial `StateFactory` `frontier-boot.test.ts` uses; and `bundle` is the committed `field.json`, read once. `handleScreenX`/`handleScreenY` are road 1's first handle projected through the same `fitBbox` the widget builds, computed in the test rather than hardcoded — a hardcoded pixel pair is the defect D1's re-review found as N2.

```ts
test("boots, draws every layer, and reports a number that matches the model", async () => {
  const host = mountPoint();                       // <figure> with an <img> in a glightbox <a>
  await displacementField(host, makeState);
  fireResize(700, 700);
  const canvas = host.find("canvas");
  assert.ok(canvas, "no canvas was inserted");
  assert.equal(canvas.style.width, "100%", "sized with an inline style, per the D1 Critical");
  const readout = host.find("p")!.textContent!;
  const { x, y, r } = bundle.buildings;
  const expected = sumC(r, corridorDistance(x, y, flatten([bundle.roads[0]!])));
  assert.match(readout, new RegExp(expected.toFixed(1).replace(".", "\\.")),
    `readout ${readout!} does not quote the model's own ${expected}`);
});

test("the fallback image and its glightbox anchor go only after a successful draw", async () => {
  const host = mountPoint();
  await displacementField(host, makeState);
  assert.ok(host.find("img"), "the image went before anything was drawn");
  fireResize(700, 700);
  assert.equal(host.find("img"), null);
  assert.equal(host.find("a"), null, "glightbox's empty anchor survived");
});

test("a zero-width container leaves the static figure in place", async () => {
  const host = mountPoint();
  await displacementField(host, makeState);
  fireResize(0, 0);
  assert.ok(host.find("img"), "removed the fallback without drawing anything");
});

test("dragging a handle moves the road and changes the cost", async () => {
  const host = mountPoint();
  await displacementField(host, makeState);
  fireResize(700, 700);
  const before = host.find("p")!.textContent!;
  const cv = host.find("canvas")!;
  cv.dispatch("pointerdown", { offsetX: handleScreenX, offsetY: handleScreenY, pointerId: 1 });
  cv.dispatch("pointermove", { offsetX: handleScreenX + 120, offsetY: handleScreenY + 90, pointerId: 1 });
  cv.dispatch("pointerup", { pointerId: 1 });
  assert.notEqual(host.find("p")!.textContent, before, "the drag changed nothing");
});

test("the width slider cannot go below the pipeline's own floor", async () => {
  const host = mountPoint();
  await displacementField(host, makeState);
  fireResize(700, 700);
  const slider = host.findAll("input").find((i) => i.type === "range")!;
  assert.equal(Number(slider.min), bundle.width.floor_m);
  assert.equal(Number(slider.min), 7,
    "permeability.py:205 RAISES below 7 m -- a narrower road is not a road this project has");
});

test("switching the second road on RAISES the cost; only dragging them together lowers it", async () => {
  // The direction matters and is easy to test backwards. Adding a disjoint road can only add
  // corridor, so the cost must rise. The DROP the page claims comes from merging two corridors,
  // which is what the reader does by dragging -- not from the road existing.
  const host = mountPoint();
  await displacementField(host, makeState);
  fireResize(700, 700);
  const cost = (): number => Number(/([\d.]+) homes/.exec(host.find("p")!.textContent!)![1]);
  const alone = cost();
  const toggle = host.findAll("input").find((i) => i.type === "checkbox")!;
  toggle.checked = true;
  toggle.dispatch("change");
  assert.ok(cost() > alone, `two roads cost ${cost()}, one cost ${alone}`);
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run web-test`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `render/field.ts`**

Spec §3's four layers, in order, mirroring `render_field`: parcel wireframe (`encoding.parcel_color`, `parcel_lw`, never filled) → corridor (one `beginPath()` per width group covering every road in it, then **one** `stroke()`, at `road_alpha`; a `stroke()` per road compounds translucency at every overlap, which would draw the opposite of the claim) → boundary and streets → disks (zero-cost as `disk_outline_lw` outlines, grazed filled at `globalAlpha = cᵢ`) → the drag handles as small filled circles at `handle_radius_px`, drawn last so they are never under a disk.

Read `ctx.lineWidth = width_m * view.scaleX` for the corridor, as `canvas.ts` does, and for the same stated reason: `fitBbox` guarantees `scaleX === scaleY` for a map view.

- [ ] **Step 4: Implement the widget**

```ts
interface FieldState { roads: Road[]; second: boolean }
```

Boot order:
1. `fetch(host.dataset.bundle!)`, `.catch(showWidgetError(host, "DisplacementField", err))` — the D1 contract: a 404 or a renamed field must be visible on the page, not an unhandled rejection behind an intact-looking figure.
2. Insert the canvas (`style.width = "100%"`, `style.aspectRatio = "1 / 1"`) and controls **before** any `<figcaption>`, so reading order stays picture-then-caption.
3. Controls: the width slider (`min = b.width.floor_m`, `max = b.width.max_m`, `step = b.width.step_m`, `value = b.width.default_m`), a checkbox for the second road, and a `<p>` readout. Native elements only — keyboard- and screen-reader-reachable, as both existing widgets are.
4. `bbox` from the building coordinates **unioned with the parcel rings**, not from the buildings alone. `PermGraph` fits to node centroids while drawing parcels and streets, and piece C recorded that the 4 % pad absorbs it *by luck* on this block — one vertex 0.4 px outside a 600 px canvas. Do not inherit the luck.
5. `observeSize(cv, (s) => { size = sizeCanvas(cv); view = fitBbox(bbox, size.width, size.height); render(); if (!drawn) { drawn = true; removeFallbackImage(host); } })`
6. `render()` recomputes `c = corridorDistance(...)` over the active roads, calls `drawField`, and writes the readout: `Σcᵢ` to one decimal and the fraction as a percentage of `n_buildings`. **Both numbers, every frame** — the page defines displacement as `Σcᵢ` and reports the fraction, so quoting one and not the other makes the widget disagree with the prose above it.
7. Pointer handlers on the canvas: `pointerdown` picks the nearest handle within `handle_radius_px * 2` in **screen** space (and takes nothing if none is that close, so a press on empty canvas is not a silent 100-metre jump); `pointermove` while held maps through `toWorld` and `state.set`; `pointerup`/`pointercancel` release. `setPointerCapture` so a drag that leaves the canvas still tracks.

- [ ] **Step 5: Run the gates**

```bash
pixi run web && pixi run web-test && pixi run web-check && pixi run lint
```
The last one rebuilds `docs/js/widgets.js`; commit it, as D1 did.

- [ ] **Step 6: Fault-inject**

Delete the registration in `mount.ts` → `widgets-bundle.test.ts` must redden (this is the guard D1 built for exactly this; if it stays green, the guard does not cover a *third* widget and that is a finding). Move `removeFallbackImage` before the first draw → the zero-width test reddens. Drop `setPointerCapture` → note whether any test notices; if none does, say so rather than adding a test that asserts an API call.

- [ ] **Step 7: Commit**

```bash
git add web/src/render/field.ts web/src/widgets/displacement-field.ts web/src/mount.ts \
        web/test/field-boot.test.ts docs/js/widgets.js
git commit -m "feat: DisplacementField -- drag a road, watch what it costs"
```

---

### Task 7: The page, and the sentence that was never true

**Files:**
- Modify: `docs/_partials/displacement.md` (marker + the §8 prose correction)
- Modify: `scripts/gen_site_pages.py` (`_displacement_field_figure`, `MARKERS`, asset copy)
- Modify: `tests/test_gen_site_pages.py`
- Modify: `docs/superpowers/backlog.md`
- Modify: `examples/displacement-field/README.md` (create)

- [ ] **Step 1: Write the failing test**

In `tests/test_gen_site_pages.py`:

```python
def test_the_displacement_page_carries_exactly_one_field_widget():
    page = (DOCS / "methodology" / "displacement.md").read_text(encoding="utf-8")
    assert page.count('data-widget="displacement-field"') == 1
    assert 'data-bundle="../assets/displacement-field/field.json"' in page
    assert "<!-- DISPFIELD -->" not in page, "the marker was emitted instead of replaced"


def test_the_caption_quotes_baked_numbers_and_not_typed_ones():
    """Every number on this page comes off disk. The apart/together pair is the whole point of the
    caption -- it is how a reader with JS off gets the overlap-is-free comparison."""
    bundle = json.loads((EXAMPLES / "displacement-field" / "field.json").read_text())
    cases = {c["name"]: c for c in bundle["reference"]}
    page = (DOCS / "methodology" / "displacement.md").read_text(encoding="utf-8")
    for name in ("apart", "coincident"):
        assert f"{cases[name]['sum_c']:.1f}" in page, f"the caption does not quote {name}"


def test_the_page_no_longer_claims_a_parcel_can_lack_a_building():
    """mesh.py:59: parcels are Voronoi cells OF the building points, so the correspondence is
    exactly one point per parcel. The old sentence described a case this pipeline cannot produce."""
    page = (DOCS / "methodology" / "displacement.md").read_text(encoding="utf-8")
    assert "no building standing on it" not in page
    assert "Voronoi" in page, "the corrected section should say what parcels actually are"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: FAIL on all three.

**Task 5 already edited both files this task edits.** It deleted `data-block` from every mount
point `gen_site_pages.py` emits, and the assertions expecting it. Do not reintroduce the attribute
on the new `<figure>`, and do not restore those assertions — check `git log -p -- scripts/gen_site_pages.py`
for what Task 5 did before adding to it.

- [ ] **Step 3: The producer**

`_displacement_field_figure()` in `gen_site_pages.py`, modelled on `_frontier_figure()` (`:382`): copy `field.png` and `field.json` through `_copy_asset` (`:101`), read the five fixtures out of the JSON, and emit the `<figure>` with `data-widget="displacement-field"`, `data-bundle`, the fallback `<img>`, and a `<figcaption>` quoting the *apart* and *coincident* `sum_c` values and the building count. Register `"DISPFIELD": _displacement_field_figure` in `MARKERS` (`:1119`). **stdlib only** — the numbers come from the JSON, never from importing `reblock`.

- [ ] **Step 4: The prose**

In `docs/_partials/displacement.md`, put `<!-- DISPFIELD -->` after the definition blockquote, and rewrite *Parcels are not buildings*. The current text ends "a parcel with no building standing on it costs nothing to cross" — a case that arises only as a degenerate geometry (`mesh.py:51-66` gives such a parcel radius 0), not as the vacant lot the sentence implies. Replace with the distinction that is actually true and is more interesting: parcels are **Voronoi cells of the building points**, so there is one cell per building by construction, and displacement is charged per building against **its own radius `rᵢ = NN/2`** rather than per parcel against parcel *area* — a road crossing one large sparse parcel is charged by its distance to the single building in it, not by how much land it consumes.

Add one sentence to the intro telling the reader what the widget does *not* say: it reports cost only, so a cheap road is not thereby a good one — permeability is the other half, and it is not computed here.

- [ ] **Step 5: `examples/displacement-field/README.md`**

Short, and generated if that is cheap — `scripts/gen_example_readme.py` already generates the `multiblock_*` variant READMEs, and the backlog records hand-written example READMEs as a live drift class (`examples/nairobi/README.md` claims 89 blocks where every `meta.json` says 43). If generating it is more than a small change, write it by hand and **say so in the report** so the drift exposure is on the record rather than discovered later.

- [ ] **Step 6: Run everything**

```bash
pixi run python -m scripts.gen_site_pages
pixi run check
~/.cache/rattler/cache/cached-envs-v0/4937c48afb8986c1/bin/mkdocs build --strict --site-dir /tmp/d2-site
```

The mkdocs run is **not optional**. D1's backlog entry records that three agents and the controller all repeated "mkdocs is importable in no environment here" and all three were wrong; running it is how D1 found the glightbox anchor at all. Note also that `deploy-site.yml` runs `mkdocs build --strict` only on push to `main`, never on a PR — so a `--strict` failure breaks the deploy, not the pull request. Then inspect the rendered HTML: the `<figure>` must keep its `data-*` attributes, its `<img>` and its `<figcaption>`, with no stray braces.

- [ ] **Step 7: Fault-inject**

Delete the `MARKERS` entry → the unknown-marker test and the widget-count test must both redden. Change one caption number by hand → the baked-numbers test reddens. Restore.

- [ ] **Step 8: Backlog**

Record in `docs/superpowers/backlog.md`, under the piece-D entry: D2 shipped and what it closed (the reflow deferral, the glightbox anchor, `data-block`, the `fitBbox` test, the `.d.ts` template guard); that **`viewBox` was rejected rather than deferred**, with the 11 px → 5 px reason, so nobody re-proposes it; the closed-form finding and its consequence for piece F; and the Displacement page's corrected sentence. Anything deferred here goes in with *why*, not just *that*.

- [ ] **Step 9: Commit**

```bash
git add docs/ scripts/gen_site_pages.py tests/test_gen_site_pages.py examples/displacement-field/
git commit -m "feat: the Displacement page gets a figure, a widget, and one true sentence"
```

---

## Task summary

| task | deliverable | gate |
|---|---|---|
| 1 | `render_field` + `field_contributions` | 3 figure tests, fault-injected |
| 2 | the road rule + the closed form pinned in Python | 8-method identity, direction, determinism |
| 3 | the bake: PNG, `field.json`, `field.d.ts`, 5 fixtures | schema, `.d.ts` equality, staleness, precision |
| 4 | `model/displacement.ts` | Python↔TS parity at 1e-3, exact 0 outside |
| 5 | `dom/resize.ts`, `dom/fallback.ts`, retrofit, riders | zero-width skip, width sweep, disposer |
| 6 | `render/field.ts` + the widget + registration | boot, drag, floor, fallback ordering |
| 7 | the page, the prose, the backlog | one widget, baked captions, `mkdocs --strict` |

**Ordering constraints.** 4 needs 3's `field.json` (its fixtures are the test data). 6 needs 4 and 5. 7 needs 3 and 6. 1 and 2 are independent of each other; 3 needs both. Nothing else is ordered.

**The one thing most likely to go wrong.** Task 5 changes both shipped widgets. `PermGraph` is live on the public site, and moving its fallback removal changes the order of operations in a boot path that currently works. If `perm-graph.ts` regresses, it regresses in a way no Python test can see — the same "the page still looks fine while the widget is silently dead" shape this branch's predecessor spent seven defects eliminating. Review that retrofit as a change to a live widget, not as a refactor.
