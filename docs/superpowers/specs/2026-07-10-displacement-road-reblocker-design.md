# Displacement-road reblocker — Design

**Status:** owner-approved (2026-07-10) · **Date:** 2026-07-10

Lay *truly straight* roads and let the greedy trade navigability benefit against the buildings the
road displaces — "the price of a straight road through a settlement." Reuses the arterial greedy's
straight-chord geometry and the existing access/directness/efficiency machinery **verbatim**; the
one new primitive is displacement.

## The model (RESOLVED)

- **Straight roads.** The greedy lays exact straight chords (the arterial `mode="aspirational"`
  geometry, not snapped to frontages).
- **Benefit is measured exactly as today, on the full settlement.** Nothing is removed: the Voronoi
  tessellation stays whole, every building point stays in place, and the new road is just a 1-D line
  added to the street set (how roads already work). The buildings a road "displaces" still sit in
  the tessellation **occluding** their neighbours — so the road's measured benefit is a *conservative*
  lower bound (no credit for clearing we don't model). This is what dissolves the removal-cheat: the
  benefit denominator never shrinks, so deletion can never manufacture benefit.
- **Displacement = the cost axis, and nothing else.** A building is displaced when its site (the real
  building point) lies in the road **corridor** = `road.buffer(corridor_m)` (`corridor_m` = road
  half-width + setback, a parameter). Displacement is a pure count; it does not touch the access
  computation.
- **Greedy objective:** commit the straight road with the best **Δbenefit / marginal-buildings-
  displaced** (benefit = directness by default, the arterial objective). Marginal = sites in the new
  road's corridor not already in a committed corridor (overlapping corridors don't double-charge).
- **Cost-benefit curve:** x = cumulative buildings displaced, y = benefit. "Navigability gained vs
  homes cleared."

Deferred (out of scope): actually removing/relocating displaced parcels; building footprints (we
have points, not polygons); constant-curvature arcs (straight-only v1).

## 1. `Block` gains `building_points`

The method operates on a `Block`; to count displacement it needs the real building sites, which are
currently consumed during Voronoi construction and discarded. Add them:

```python
@dataclass(frozen=True)
class Block:
    ...
    building_points: GeoDataFrame = field(default_factory=_empty_points)   # the real sites; may be empty
```

- **`KblockSource`** populates it — it already reads the building points per block for Voronoi; keep
  them as `block.building_points` (geometry = Points, region UTM).
- **`region_block`** (region.py) = the union of member `building_points`.
- **`ShapefileSource`** leaves it empty (a parcel shapefile has no point cloud — honest, total).

A genuine new field with an empty default (not a compat shim); existing `Block(...)` constructors
that omit it get empty. Not part of `Block.identity` (derived from the same source, like `parcels`).

## 2. Displacement primitive (`budget.py`)

```python
def displacement_count(building_points: GeoDataFrame, roads: GeoDataFrame, corridor_m: float) -> int:
    """Buildings whose site lies in the road corridor (union of `roads.buffer(corridor_m)`).
    0 if there are no points or no roads."""
```
Marginal displacement of a candidate road (greedy denominator) = `displacement_count(pts, committed ∪
{road}, m) - displacement_count(pts, committed, m)` — sites newly hit, so overlapping corridors are
counted once.

## 3. `cost` axis on `GreedyArterialReblocker`

Add two params (orthogonal to `mode`/`objective`):
```python
cost: str = "length"          # "length" (Δbenefit/metre, today) | "displacement" (Δbenefit/building)
corridor_m: float = 3.0       # road half-width + setback; the displacement corridor
```
- `identity` → `("greedy_arterial", mode, objective, cost)` (new cache key; `corridor_m` too if it
  changes results — include it).
- `_greedy_arterials`: when `cost="displacement"`, the per-candidate denominator is the road's
  **marginal displacement** (via `block.building_points`, `corridor_m`) instead of `real.length`.
- **Zero-displacement road** with positive benefit → strictly good; rank it above any road that
  displaces (take the free navigability first) rather than dividing by zero.
- Guard/document: `cost="displacement"` is meaningful only with `mode="aspirational"` (straight roads)
  on a block with a non-empty `building_points` (kblock). With empty points every road displaces 0, so
  it degenerates to "take all beneficial straight roads."

## 4. Cost-benefit curve with a displacement x-axis (`budget.py`)

```python
def cost_benefit_curve(block, roads, *, benefit_fn=access_benefit,
                       cost="length", corridor_m=3.0) -> Curve:
```
Same road-ordered sweep; when `cost="displacement"`, the x-accumulator is **cumulative buildings
displaced** (sites in the union of corridors of the roads added so far) instead of road density
(m/ha). `auc`/`efficiency_directness_curves` reuse it. This is the headline output: directness (or
access) benefit vs cumulative buildings displaced.

## 5. Config + compare

- `conf/method/greedy_arterial_displacement.yaml`: `mode=aspirational`, `objective=directness`,
  `cost=displacement`, `corridor_m=3.0`.
- A `compare_config` entry including it, so the displacement method shows up. The displacement
  **curve** (`cost="displacement"`) is produced for it (and any method — buildable roads land near
  x≈0 since frontage roads displace almost nothing, which is itself the informative contrast).

## 6. Render (in-scope, builds on the building-point overlay)

In the after-heatmap, mark the **displaced** building points (sites in the corridor) distinctly (e.g.
a red ring/×) on top of the normal building-point overlay — the cost made visible next to the straight
road. Gated by the existing render path; a block without `building_points` just shows none.

## 7. Testing

- **`Block.building_points`:** kblock block has Points (count = buildings in block), region UTM;
  `region_block` unions members; `ShapefileSource` empty.
- **`displacement_count`:** a straight road through a known point grid counts exactly the sites in its
  corridor; sparse points off-corridor aren't counted; marginal de-dups on overlapping corridors.
- **Greedy `cost="displacement"`:** deterministic; given two equal-benefit candidate roads, prefers the
  lower-displacement one; a zero-displacement beneficial road is taken; `identity` includes `cost`.
- **`cost_benefit_curve(cost="displacement")`:** x is cumulative buildings displaced, monotone
  non-decreasing; degenerates sensibly (x≡0) when `building_points` is empty.
- **Render:** the after-heatmap smoke-renders with displaced points marked; a no-points block renders
  without error.

## 8. Out of scope (follow-ons)

- Removal/relocation of displaced parcels + access recomputed on survivors (the richer model,
  deliberately deferred — retention is the defensible, cheap v1).
- True building footprints (polygons); constant-curvature arcs; per-building relocation cost.
