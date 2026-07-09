# Cost-benefit framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare reblocking methods across the whole budget range — incrementally add each method's roads (drainage order), score access at ~20 budgets → a cost-benefit curve; AUC = a 0–1 efficiency score. Emit an aggregate AUC table + sample curve plots.

**Architecture:** `reblock.budget` (pure curve machinery: `access_burden`, `road_drainage`, `cost_benefit_curve`, `auc`) + `reblock.compare` (Hydra sweep over screened-blocks × methods) + emitters (AUC table + overlaid curves). Benefit = fraction of Σ_parcels depth² removed; cost = road density (m/ha). Spec: `docs/superpowers/specs/2026-07-09-cost-benefit-framework-design.md`.

**Tech Stack:** Python 3.12, networkx, shapely/geopandas, matplotlib, Hydra, pixi, pytest, `mypy --strict`, ruff.

## Global Constraints

- `pixi run check` stays green — `ruff check` + `mypy --strict src tests scripts/crossblock_probe.py` + pytest. Suite is currently **150 tests**.
- **`parcel_access_layers` change is additive** — a new optional `adj` param (precomputed adjacency); default `None` recomputes exactly as today. No behavior change; existing callers/tests unaffected.
- **The curve is correct even if a prefix isn't fully connected** — `parcel_access_layers` → `street_connectivity` drops floating (non-street-connected) roads, so a not-yet-connected road in a prefix simply grants no access until its trunk is added. Benefit is monotonic non-decreasing (supersets → more connected roads).
- **Uniform `road_drainage`** — one function over any method's road segments; the comparison is apples-to-apples.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
  ```

---

### Task 1: `reblock.budget` — curve machinery + `parcel_access_layers(adj=)`

**Files:**
- Modify: `src/reblock/derive/access.py` (add optional `adj` param)
- Create: `src/reblock/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Produces: `access_burden(depths)->float`; `road_drainage(block, roads)->list[int]`; `Curve(cost, benefit)`; `cost_benefit_curve(block, roads, n_points=20)->Curve`; `auc(curve, cost_cap)->float`.
- Consumes: `parcel_access_layers` (+ new `adj`), `parcel_adjacency`, `STREET_TOL`.

- [ ] **Step 1: Add the optional `adj` param to `parcel_access_layers`**

In `src/reblock/derive/access.py`, change the signature + the adjacency line so a precomputed adjacency is reused when given:

```python
def parcel_access_layers(
    block: Block, roads: GeoDataFrame | None, *, tol: float = STREET_TOL,
    adj: list[set[int]] | None = None,
) -> pd.Series:
```
and replace `adj = parcel_adjacency(geoms, tol)` with:
```python
    adj = adj if adj is not None else parcel_adjacency(geoms, tol)
```
(Everything else unchanged. `adj` is keyed by row position over `block.parcels.geometry`, so a caller precomputing `parcel_adjacency(list(block.parcels.geometry), tol)` for the same block passes a compatible list.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_budget.py`:

```python
from typing import cast

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.budget import Curve, access_burden, auc, cost_benefit_curve, road_drainage
from reblock.contracts import Block
from reblock.methods.dijkstra import DijkstraReblocker

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(n * n))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_access_burden_is_sum_of_squared_depths() -> None:
    assert access_burden(pd.Series([1, 2, 3])) == 1 + 4 + 9


def test_road_drainage_trunks_exceed_leaves() -> None:
    # dijkstra's roads on a 5x5 grid: a segment near the street carries more parcels than a leaf.
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    drain = road_drainage(block, roads)
    assert len(drain) == len(roads) and max(drain) > min(drain) and max(drain) >= 2


def test_cost_benefit_curve_is_monotonic_and_reaches_full() -> None:
    block = _grid_block(5)
    roads = DijkstraReblocker().propose(block).roads
    curve = cost_benefit_curve(block, roads, n_points=10)
    assert curve.cost[0] == 0.0 and curve.benefit[0] == 0.0
    assert curve.benefit == sorted(curve.benefit)          # monotonic non-decreasing
    assert curve.benefit[-1] > 0.5                         # flattens the block substantially


def test_auc_rewards_reaching_benefit_at_lower_cost() -> None:
    cheap = Curve(cost=[0.0, 1.0], benefit=[0.0, 1.0])     # full benefit by cost 1
    dear = Curve(cost=[0.0, 4.0], benefit=[0.0, 1.0])      # full benefit only by cost 4
    assert auc(cheap, cost_cap=4.0) > auc(dear, cost_cap=4.0)
    assert 0.0 <= auc(dear, cost_cap=4.0) <= 1.0
```

- [ ] **Step 3: Run to verify failure**

Run: `pixi run pytest tests/test_budget.py -v`
Expected: FAIL — `No module named 'reblock.budget'`.

- [ ] **Step 4: Implement `src/reblock/budget.py`**

```python
"""Cost-benefit curves for reblocking methods: add a method's roads incrementally in
drainage order, score access at each budget, trace benefit (fraction of Sigma depth^2
removed) vs cost (road density, m/ha). AUC = a 0-1 efficiency score. See the design spec.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, parcel_access_layers
from reblock.derive.adjacency import parcel_adjacency


def _rnd(c: tuple[float, float]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


def access_burden(depths: pd.Series) -> float:
    """Sigma depth^2 -- q=2 severity-weighted access burden (kblock parcels = one building each)."""
    return float((depths.astype("float64") ** 2).sum())


def road_drainage(block: Block, roads: GeoDataFrame, *, tol: float = STREET_TOL) -> list[int]:
    """Per-road parcel count: build a graph from the road segments, route each parcel to the
    street through it, and count how many parcels use each segment. Uniform across methods."""
    n = len(roads)
    if n == 0:
        return []
    g: nx.Graph = nx.Graph()
    edge_row: dict[frozenset[tuple[float, float]], int] = {}
    for i, geom in enumerate(roads.geometry):
        cs = list(geom.coords)
        for a, b in zip(cs, cs[1:], strict=False):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                g.add_edge(na, nb, weight=Point(na).distance(Point(nb)))
                edge_row[frozenset((na, nb))] = i
    street = unary_union(list(block.streets.geometry))
    snodes = {node for node in g.nodes if Point(node).distance(street) <= tol}
    if not snodes:
        return [0] * n
    dist, paths = nx.multi_source_dijkstra(g, sorted(snodes))
    nodes = list(g.nodes)
    tree = STRtree([Point(node) for node in nodes])
    counts: dict[int, int] = defaultdict(int)
    for geom in block.parcels.geometry:
        reach = [nodes[j] for j in tree.query(geom, predicate="dwithin", distance=tol)
                 if nodes[j] in dist]
        if not reach:
            continue
        entry = min(reach, key=lambda node: (dist[node], node))
        for a, b in zip(paths[entry], paths[entry][1:], strict=False):
            row = edge_row.get(frozenset((a, b)))
            if row is not None:
                counts[row] += 1
    return [counts.get(i, 0) for i in range(n)]


@dataclass(frozen=True)
class Curve:
    cost: list[float]     # cumulative road density, m/ha
    benefit: list[float]  # fraction of Sigma depth^2 removed, in [0, 1]


def cost_benefit_curve(block: Block, roads: GeoDataFrame, *, n_points: int = 20,
                       tol: float = STREET_TOL) -> Curve:
    """Order roads by drainage descending, then at n_points cumulative-length budgets score
    benefit = fraction of Sigma depth^2 removed vs the no-roads baseline. Adjacency is built
    once and reused across prefixes."""
    adj = parcel_adjacency(list(block.parcels.geometry), tol)
    base = access_burden(parcel_access_layers(block, None, tol=tol, adj=adj))
    cost, benefit = [0.0], [0.0]
    if len(roads) == 0 or base == 0.0:
        return Curve(cost, benefit)
    drain = road_drainage(block, roads, tol=tol)
    order = sorted(range(len(roads)), key=lambda i: (-drain[i], i))
    ordered = roads.iloc[order].reset_index(drop=True)
    lengths = ordered.geometry.length.to_numpy()
    cum = lengths.cumsum()
    total = float(cum[-1])
    area_ha = block.boundary.area / 1e4
    seen = 0
    for k in range(1, n_points + 1):
        m = int((cum <= (k / n_points) * total + 1e-9).sum())
        if m <= seen:
            continue
        seen = m
        b = 1.0 - access_burden(parcel_access_layers(block, ordered.iloc[:m], tol=tol, adj=adj)) / base
        cost.append(float(cum[m - 1]) / area_ha)
        benefit.append(b)
    return Curve(cost, benefit)


def auc(curve: Curve, cost_cap: float) -> float:
    """Normalized area under benefit-vs-cost over [0, cost_cap] (curve held at its terminal
    benefit beyond its own max cost) -> 0-1 efficiency; higher = more access per meter."""
    if cost_cap <= 0.0 or len(curve.cost) < 2:
        return 0.0
    cs, bs = list(curve.cost), list(curve.benefit)
    if cs[-1] < cost_cap:                       # extend the plateau to the common cap
        cs, bs = cs + [cost_cap], bs + [bs[-1]]
    area = 0.0
    for (c0, b0), (c1, b1) in zip(zip(cs, bs, strict=False), zip(cs[1:], bs[1:], strict=False)):
        if c1 <= cost_cap:
            area += 0.5 * (b0 + b1) * (c1 - c0)
    return area / cost_cap
```

- [ ] **Step 5: Run tests + full check**

Run: `pixi run pytest tests/test_budget.py -v` then `pixi run check`
Expected: PASS. Drainage trunks > leaves; the curve is monotonic and flattens the grid; AUC rewards cheaper access. `parcel_access_layers`'s new `adj` param leaves all existing tests green. ~154 tests.

- [ ] **Step 6: Commit**

```bash
git add src/reblock/derive/access.py src/reblock/budget.py tests/test_budget.py
git commit -m "$(cat <<'EOF'
feat: reblock.budget -- cost-benefit curve machinery (drainage order, AUC)

access_burden (Sigma depth^2), road_drainage (uniform per-road parcel counts),
cost_benefit_curve (drainage-order prefixes -> benefit-fraction vs road density,
adjacency reused), auc (0-1 efficiency). parcel_access_layers gains an optional
precomputed-adj param so the ~20 curve evals share one adjacency build (additive).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

### Task 2: `reblock.compare` entrypoint + emitters (AUC table + sample curves)

**Files:**
- Create: `src/reblock/compare.py`
- Modify: `src/reblock/emit.py` (add `compare_report`)
- Create: `conf/compare_config.yaml`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `reblock.budget.{cost_benefit_curve, auc}`; `derivations.propose`; `Source`/`Screen`/`Method`; the render helpers.
- Produces: `MethodCurve(method, block_id, curve, auc)`; `compare(cfg) -> list[MethodCurve]`; `compare_report(results, out_dir) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare.py` — a compose+run over the committed DJI fixture (dijkstra + peel; topology omitted for speed), asserting the table + a curve PNG land in the run dir and dijkstra's mean AUC is reported:

```python
import subprocess
import sys
from pathlib import Path


def test_compare_writes_table_and_curves(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "reblock.compare",
         "data=dji", "eval=kcomplexity", "max_blocks=1",
         "methods=[dijkstra,peel]", f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "auc_table.csv").read_text()
    assert "dijkstra" in table and "peel" in table
    assert list(tmp_path.glob("curve_*.png"))
```

Also a focused emitter unit test:

```python
def test_compare_report_writes(tmp_path: Path) -> None:
    from reblock.budget import Curve
    from reblock.compare import MethodCurve, compare_report
    results = [
        MethodCurve("dijkstra", "b1", Curve([0.0, 1.0], [0.0, 0.9]), 0.8),
        MethodCurve("peel", "b1", Curve([0.0, 2.0], [0.0, 0.9]), 0.5),
    ]
    compare_report(results, tmp_path)
    assert (tmp_path / "auc_table.csv").exists()
    assert (tmp_path / "curve_b1.png").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/test_compare.py -v`
Expected: FAIL — `No module named 'reblock.compare'`.

- [ ] **Step 3: Implement `compare_report` in `emit.py`**

Add to `src/reblock/emit.py` (imports `plt`, `save_render`, `Path` already present):

```python
def compare_report(results: list[MethodCurve], out_dir: Path) -> None:
    """Aggregate AUC table (mean efficiency per method) + overlaid cost-benefit curves
    per block. `results` is the flat (method x block) list from reblock.compare."""
    import csv
    from statistics import mean
    out_dir.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[float]] = {}
    by_block: dict[str, list[MethodCurve]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r.auc)
        by_block.setdefault(r.block_id, []).append(r)
    with (out_dir / "auc_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_auc", "n_blocks"])
        for m, aucs in sorted(by_method.items(), key=lambda kv: -mean(kv[1])):
            w.writerow([m, f"{mean(aucs):.4f}", len(aucs)])
    for block_id, curves in by_block.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for mc in curves:
            ax.plot(mc.curve.cost, mc.curve.benefit, marker="o", label=f"{mc.method} (AUC {mc.auc:.2f})")
        ax.set_xlabel("road density (m/ha)")
        ax.set_ylabel("fraction of access-burden removed")
        ax.set_title(f"cost-benefit: {block_id}")
        ax.legend()
        save_render(fig, out_dir / f"curve_{block_id}.png")
        plt.close(fig)
```

Add `from reblock.compare import MethodCurve` under `TYPE_CHECKING` (or accept `list` structurally) to avoid a circular import — `compare` imports `compare_report` from `emit`, so type the param via `TYPE_CHECKING`:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reblock.compare import MethodCurve
```
and annotate `results: "list[MethodCurve]"`.

- [ ] **Step 4: Implement `src/reblock/compare.py`**

```python
"""Hydra entrypoint: sweep cost_benefit_curve over screened blocks x a list of methods,
emit the aggregate AUC table + per-block curve plots. Config only at the edge (like run.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.budget import Curve, auc, cost_benefit_curve
from reblock.contracts import Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import compare_report
from reblock.pipeline import sample

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodCurve:
    method: str
    block_id: str
    curve: Curve
    auc: float


def compare(cfg: DictConfig) -> list[MethodCurve]:
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    methods = [cast(Method, instantiate(cfg.all_methods[name])) for name in cfg.methods]
    selection = screen.select(source)
    picked = sample(selection, cfg.max_blocks)
    if picked is not None:
        source.block_ids = picked  # type: ignore[attr-defined]
        blocks = list(source.region().blocks)
    else:
        from itertools import islice
        blocks = list(islice(source.region().blocks, cfg.max_blocks))

    # one curve per (block, method); a per-block common cost cap = the max full road density.
    raw: list[tuple[str, str, Curve]] = []
    for block in blocks:
        for method in methods:
            roads = propose(method, block).roads
            raw.append((_name(method), block.block_id, cost_benefit_curve(block, roads)))
    results: list[MethodCurve] = []
    for block_id in {b for _, b, _ in raw}:
        block_curves = [(m, c) for m, b, c in raw if b == block_id]
        cap = max((c.cost[-1] for _, c in block_curves if c.cost), default=0.0)
        for m, c in block_curves:
            results.append(MethodCurve(m, block_id, c, auc(c, cap)))
    return results


def _name(method: Method) -> str:
    ident = method.identity
    return str(ident[0]) if isinstance(ident, tuple) and ident else type(method).__name__


@hydra.main(version_base=None, config_path="../../conf", config_name="compare_config")
def main(cfg: DictConfig) -> None:
    results = compare(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    compare_report(results, out_dir)
    for r in sorted(results, key=lambda r: -r.auc):
        log.info("%s %s AUC=%.3f", r.block_id, r.method, r.auc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `conf/compare_config.yaml`**

```yaml
defaults:
  - data: dji
  - screen: identity
  - eval: kcomplexity
  - _self_

# The methods available, then the names to actually run. Overriding the names via
# CLI is clean -- `methods=[dijkstra,peel]` (topology is slow, minutes/block).
all_methods:
  dijkstra: {_target_: reblock.methods.dijkstra.DijkstraReblocker}
  peel: {_target_: reblock.methods.peel.PeelReblocker}
  topology: {_target_: reblock.methods.topology.TopologyMethod}
methods: [dijkstra, peel, topology]

shapefile: ???
region_id: phule
assumed_crs: null
max_blocks: 5
block_ids: null

hydra:
  job:
    chdir: false
```

- [ ] **Step 6: Run tests + full check**

Run: `pixi run check`
Expected: PASS. `python -m reblock.compare data=dji methods=[dijkstra,peel] max_blocks=1` writes `auc_table.csv` (dijkstra + peel rows) + a `curve_<block>.png`; the emitter unit test writes both; no circular-import error. ~156 tests.

- [ ] **Step 7: Commit**

```bash
git add src/reblock/compare.py src/reblock/emit.py conf/compare_config.yaml tests/test_compare.py
git commit -m "$(cat <<'EOF'
feat: reblock.compare -- sweep methods x blocks, emit AUC table + curves

Hydra entrypoint: screen -> pick blocks -> for each (block, method) trace a
cost_benefit_curve, score AUC vs a per-block common cost cap. compare_report writes
auc_table.csv (mean efficiency per method) + overlaid cost-benefit curve PNGs per
block. conf/compare_config.yaml (methods list; topology omittable for speed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0125mCca6BQaTXiLZPMyFa8x
EOF
)"
```

---

## Self-Review

**Spec coverage:** `access_burden`/`road_drainage`/`cost_benefit_curve`/`auc` (Task 1); the `parcel_access_layers(adj=)` optimization (Task 1); the `reblock.compare` sweep + `compare_report` AUC-table + per-block curve emitters + config (Task 2). Benefit = fraction Σdepth² removed; cost = road density; drainage order; AUC 0–1 over a per-block common cost cap. Aggregate table + sample curves both delivered.

**Placeholder scan:** complete code in every step. The `TYPE_CHECKING` import of `MethodCurve` in `emit.py` (Task 2 Step 3) is a real circular-import resolution, not a placeholder.

**Type consistency:** `Curve(cost: list[float], benefit: list[float])` is produced by `cost_benefit_curve` and consumed by `auc`/`compare_report`; `road_drainage -> list[int]` aligns positionally with `roads`; `MethodCurve(method, block_id, curve, auc)` is produced by `compare` and consumed by `compare_report`; `parcel_access_layers(..., adj=)` matches the precomputed `parcel_adjacency(list(block.parcels.geometry), tol)`. `compare` reuses `pipeline.sample` + `derivations.propose` (derive-cached).

**Correctness guard:** benefit monotonicity + full-flatten are tested on the real DijkstraReblocker output; AUC ordering is tested on synthetic curves; the entrypoint's end-to-end test proves the sweep + emitters wire up on the committed DJI fixture without network.
