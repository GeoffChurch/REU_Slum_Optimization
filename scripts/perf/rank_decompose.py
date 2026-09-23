"""WHY does the first-order ranking fail? Numerator or denominator?

`first_order_rank.py` says the estimate puts the exact winner around rank 600-900 out of ~1,300 --
no better than chance. But that run confounds two independent approximations, and they have
completely different fixes:

  NUMERATOR   est = sum over parcels the CHORD fronts of (d^2 - 1)
              vs the exact benefit, which is the peel over the SNAPPED road and includes the
              ripple to neighbours whose depth also falls.
  DENOMINATOR ranked on the chord's LENGTH
              vs the exact cost, `displacement` of the SNAPPED road -- expected buildings displaced.
              A long chord over empty ground is cheap to build and I rank it as expensive.

If the numerator is sound and only the cost proxy is wrong, tier 2 survives with a better cheap
denominator (a bulk building-count query, one more `dwithin`). If the numerator itself does not
track the exact benefit, tier 2 is dead as specified and no denominator rescues it.

So: score every candidate BOTH ways in the same step and correlate the pieces separately. Run
serial (`workers=1`) so `eval_candidate` can be wrapped to stash each candidate's exact `(raw,
denom)` -- a fork pool would compute those in children and throw them away.

Spearman rather than Pearson throughout: only the ORDER matters to a shortlist, and the exact gain
has infinities (a beneficial zero-displacement road) that a linear correlation cannot digest.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import shapely
from scipy.stats import spearmanr
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

import reblock.methods.arterial.engines as engines
import reblock.methods.arterial.scoring as scoring
from reblock.contracts import Block
from reblock.data.pools import evenly_spaced
from reblock.derive.access import (
    STREET_TOL,
    ParcelAdjacency,
    parcel_access_layers,
    past_every_parcel,
)
from reblock.methods.arterial import (
    Access,
    Displacement,
    ExactEngine,
    GreedyArterialReblocker,
    SnapToBoundary,
)
from reblock.methods.arterial.primitives import _candidate_chords, _planarize
from reblock.methods.arterial.scoring import _best_candidate, eval_candidate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from scripts._donor_pool import donor_pool
from scripts.perf.first_order_rank import first_order_gain

N_BLOCKS = 6
MAX_ROADS = 4
RADIUS = STREET_TOL
OUT = Path("scripts/perf/rank_decompose.json")

_EXACT: list[tuple[float, float, float]] = []       # (raw benefit, exact denom, snapped length)
_CHORDS: list[LineString] = []
_ROWS: list[dict[str, float]] = []

_BLOCK: Block | None = None
_ADJACENCY: ParcelAdjacency | None = None
_TREE: STRtree | None = None
_BTREE: STRtree | None = None
_COMMITTED: list[LineString] = []
_HALF_W = 3.0


def _eval_hook(chord: LineString) -> tuple[float, BaseGeometry | None]:
    """`eval_candidate`, plus the (benefit, cost) split it discards. Recomputed here from the
    step's own gain and cost rather than plumbed out of `eval_candidate`, so the shipped scorer is
    untouched and the gain it returns is the real one."""
    st = scoring._STEP_STATE
    gain, real = eval_candidate(chord)
    assert st is not None
    if real is None or real.length == 0:
        _EXACT.append((0.0, 0.0, 0.0))
        return gain, real
    assert isinstance(real, LineString)             # what `eval_candidate`'s realizers produce
    raw = st.gain.of(real)
    denom = st.cost.of(real)
    _EXACT.append((raw, denom, float(real.length)))
    return gain, real


def _chords_hook(anchors: list[tuple[float, float]],
                 targets: list[tuple[float, float]]) -> list[LineString]:
    global _CHORDS
    _CHORDS = _candidate_chords(anchors, targets)
    _EXACT.clear()
    return _CHORDS


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or len(set(a[ok].tolist())) < 2 or len(set(b[ok].tolist())) < 2:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def _best_hook(results: Iterable[tuple[float, BaseGeometry | None]]
               ) -> tuple[float, BaseGeometry | None]:
    res = list(results)
    gain, real = _best_candidate(res)
    blk = _BLOCK
    if (real is None or blk is None or _TREE is None or _ADJACENCY is None
            or len(_EXACT) != len(_CHORDS)):
        return gain, real

    base = _planarize(list(_COMMITTED), blk.crs, 2.0 * _HALF_W)
    # The adjacency set beside `_BLOCK`, for the same block -- both are written together
    # below, and the adjacency carries its block, so the peel cannot read another one's.
    depths = parcel_access_layers(_ADJACENCY, base if len(base) else None,
                                  unreached=past_every_parcel)
    order = depths.loc[blk.parcels["parcel_id"]].to_numpy(dtype=float)
    weights = order ** 2 - 1.0

    est = first_order_gain(_CHORDS, weights, _TREE, RADIUS)
    arr = np.asarray(_CHORDS, dtype=object)
    lengths = shapely.length(arr)
    # cheap displacement proxy: how many buildings lie within the chord's corridor. One bulk
    # `dwithin`, the same shape as the gain query -- affordable at region scale, unlike the real
    # `displacement`, which needs a distance from every building to the corridor polygon.
    assert _BTREE is not None
    src, _tgt = _BTREE.query(arr, predicate="dwithin", distance=_HALF_W)
    n_bldg = np.bincount(src, minlength=len(_CHORDS)).astype(float)

    ex = np.array(_EXACT)
    raw_x, denom_x, snap_len = ex[:, 0], ex[:, 1], ex[:, 2]
    live = snap_len > 0
    gains = np.array([g for g, _ in res])

    row = {
        "n_cand": float(len(_CHORDS)), "n_live": float(live.sum()),
        # numerator: does the local estimate track the exact benefit?
        "rho_num": _rho(est[live], raw_x[live]),
        # denominator: do the cheap costs track the exact displacement?
        "rho_den_length": _rho(lengths[live], denom_x[live]),
        "rho_den_nbldg": _rho(n_bldg[live], denom_x[live]),
        "rho_den_snaplen": _rho(snap_len[live], denom_x[live]),
        # chord vs snapped length -- how much does snapping distort the geometry at all?
        "rho_len_snaplen": _rho(lengths[live], snap_len[live]),
        # end to end, three cheap rankings against the exact gain
        "rho_est_len": _rho((est / np.maximum(lengths, 1e-12))[live], gains[live]),
        "rho_est_nbldg": _rho((est / np.maximum(n_bldg, 1.0))[live], gains[live]),
        "rho_est_raw": _rho(est[live], gains[live]),
        # the ceiling: rank by the EXACT benefit over the cheap cost. If this is also poor, the
        # cost proxy is the binding problem, not the estimate.
        "rho_exactnum_len": _rho((raw_x / np.maximum(lengths, 1e-12))[live], gains[live]),
        # THE TIER-3 GATE. Tier 3 (ALT/landmark depths) improves only the NUMERATOR, so its ceiling
        # is a perfect numerator over the best cheap denominator. If this sits at the same place as
        # `rho_est_nbldg`, a perfect numerator buys nothing over the first-order one and tier 3 is
        # dead before it is written -- the residual error is all denominator.
        "rho_exactnum_nbldg": _rho((raw_x / np.maximum(n_bldg, 1.0))[live], gains[live]),
        # and the reverse control: exact COST with the cheap numerator, isolating the denominator's
        # own contribution the same way.
        "rho_est_exactden": _rho((est / np.maximum(denom_x, 1e-12))[live], gains[live]),
        "frac_zero_denom": float((denom_x[live] <= 0).mean()),
    }
    _ROWS.append(row)
    assert isinstance(real, LineString)             # as `_greedy_arterials` asserts on commit
    _COMMITTED.append(real)
    print(f"      step {len(_COMMITTED)}: {len(_CHORDS):>5,} cand  "
          f"rho(num) {row['rho_num']:+.2f}  rho(len~disp) {row['rho_den_length']:+.2f}  "
          f"rho(nbldg~disp) {row['rho_den_nbldg']:+.2f}  "
          f"end-to-end est/len {row['rho_est_len']:+.2f}  "
          f"exactnum/len {row['rho_exactnum_len']:+.2f}", flush=True)
    return gain, real


def main() -> None:
    # Replaced in `engines`, the namespace its loop looks them up in -- it imports the first two
    # without re-exporting them, which is all the ignores concede.
    engines._candidate_chords = _chords_hook           # type: ignore[attr-defined]
    engines._best_candidate = _best_hook               # type: ignore[attr-defined]
    engines.eval_candidate = _eval_hook
    pools = donor_pool("donor_pool=capetown").pools()
    blocks = pools.blocks
    counts = [float(len(b.parcels)) for b in blocks]
    sel = [i for i in pools.recipients if len(blocks[i].parcels) <= 90]

    global _BLOCK, _ADJACENCY, _TREE, _BTREE
    by_block: dict[str, list[dict[str, float]]] = {}
    for i in evenly_spaced(sorted(sel), counts, N_BLOCKS):
        b = blocks[i]
        _ROWS.clear()
        _COMMITTED.clear()
        _BLOCK = b
        _ADJACENCY = ParcelAdjacency.of(b, STREET_TOL)
        _TREE = STRtree(list(b.parcels.geometry))
        _BTREE = STRtree(list(b.building_geometries.geometry))
        print(f"  {b.block_id}  ({len(b.parcels)} parcels)", flush=True)
        # workers=1 -> the serial path, so `_eval_hook`'s stash survives (a fork pool would
        # compute it in children and discard it)
        GreedyArterialReblocker(realizer=SnapToBoundary(lam=2.0), objective=Access(),
                                cost=Displacement(), workers=1, max_roads=MAX_ROADS, n_anchors=32,
                                top_k=8,
                                road_width_m=DEFAULT_ROAD_WIDTH_M, engine=ExactEngine(),
                                max_anchors=0).propose(b)
        by_block[b.block_id] = list(_ROWS)
    OUT.write_text(json.dumps(by_block, indent=1))

    rows = [r for v in by_block.values() for r in v]
    if not rows:
        print("no steps recorded -- the hooks did not fire")
        return

    def med(key: str) -> float:
        vals = np.array([r[key] for r in rows])
        return float(np.nanmedian(vals))

    print(f"\n{'=' * 78}\nWHERE THE RANKING BREAKS -- {len(rows)} steps over {len(by_block)} "
          "blocks (Spearman, median over steps)\n")
    print("  NUMERATOR   first-order estimate  ~ exact benefit      "
          f"{med('rho_num'):+.3f}")
    print("  DENOMINATOR chord length          ~ exact displacement "
          f"{med('rho_den_length'):+.3f}")
    print("              buildings in corridor ~ exact displacement "
          f"{med('rho_den_nbldg'):+.3f}")
    print("              SNAPPED length        ~ exact displacement "
          f"{med('rho_den_snaplen'):+.3f}")
    print("  GEOMETRY    chord length          ~ snapped length     "
          f"{med('rho_len_snaplen'):+.3f}")
    print("\n  END TO END, cheap ranking ~ exact gain")
    print(f"    est / chord length                                  {med('rho_est_len'):+.3f}")
    print(f"    est / buildings in corridor                         {med('rho_est_nbldg'):+.3f}")
    print(f"    est alone (no cost)                                 {med('rho_est_raw'):+.3f}")
    print(f"    CEILING: EXACT benefit / chord length               "
          f"{med('rho_exactnum_len'):+.3f}")
    print("\n  THE TIER-3 GATE -- tier 3 improves only the numerator, so this is its ceiling")
    print(f"    EXACT benefit / buildings in corridor               "
          f"{med('rho_exactnum_nbldg'):+.3f}")
    print(f"    est / EXACT displacement (the reverse control)      "
          f"{med('rho_est_exactden'):+.3f}")
    gap = med("rho_exactnum_nbldg") - med("rho_est_nbldg")
    print(f"\n    a PERFECT numerator buys {gap:+.3f} over the first-order one.")
    print(f"    a PERFECT denominator buys "
          f"{med('rho_est_exactden') - med('rho_est_nbldg'):+.3f}.")
    print(f"\n  candidates with zero exact displacement (infinite gain): "
          f"{med('frac_zero_denom'):.1%} of live candidates")


if __name__ == "__main__":
    main()
