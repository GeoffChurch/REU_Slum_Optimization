"""An addition to item 4 (not asked for): a feasible clearing at the full budget inside the top
tiles, to tighten the truncation bound. Each stored clearing's picks inside the tiles, then the
tiles' other buildings by the screening layer's own per-building signal -- the isotropic
sensitivity density under the footprint, (S_xx + S_yy) of the L 100 inside-avg macro adjoint
(the first-order gain of raising the local conductance evenly) -- added in that order while
they fit the budget (relax.cut_to_budget's rule). No optimizer runs; the clearings are scored by
region_score_gpu.py like the truncations.

    ... uv run python research/roadless/fft_homog/region_topup.py <restrict tag>
reads OUT/restrict_tiles_<tag>.json, writes OUT/topup_clearings_<tag>.json.
"""
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import numpy as np  # noqa: E402
import region_common as rc  # noqa: E402
import ts  # noqa: E402
from region_macro import MacroBase  # noqa: E402
from region_tiles import BUDGET, TILE  # noqa: E402

if __name__ == "__main__":
    tag = sys.argv[1]
    fab = rc.load_fabric()
    tf = ts.load_field(str(rc.OUT / "fields_L100_s25.npz"))
    mac = MacroBase(ts.fill(tf, None, True), tf.s, 4, fab, adjoint=True)
    iso = mac.S[..., 0] + mac.S[..., 1]
    nx = fab.o.shape[1]
    er, ec = (fab.sc // nx) // mac.k, (fab.sc % nx) // mac.k
    dens = (np.bincount(fab.sb, weights=iso[er, ec] * fab.sn, minlength=len(fab.w))
            / np.maximum(np.bincount(fab.sb, weights=fab.sn, minlength=len(fab.w)), 1))
    cost = fab.area / fab.area.sum()
    tiles = rc.full_tiles(fab.inside, TILE)
    tid = {t: i for i, t in enumerate(tiles)}
    br, bc = np.floor(fab.py + 0.5).astype(int), np.floor(fab.px + 0.5).astype(int)
    btile = np.array([tid.get((r - r % TILE, c - c % TILE), -1)
                      for r, c in zip(br, bc, strict=True)])
    chosen = json.loads((rc.OUT / f"restrict_tiles_{tag}.json").read_text())
    out = {}
    for key, tl in chosen.items():
        sel = np.zeros(len(tiles) + 1, dtype=bool)
        sel[np.asarray(tl, dtype=np.int64)] = True
        cand = np.flatnonzero(sel[btile])                     # btile -1 -> sel[-1] False
        order = cand[np.argsort(-dens[cand], kind="stable")]
        for name, ids in rc.stored_clearings().items():
            inside = ids[sel[btile[ids]]]
            r = np.zeros(len(cost), dtype=bool)
            r[inside] = True
            left = BUDGET - float(cost[r].sum()) + 1e-12
            for j in order:
                if not r[j] and cost[j] <= left:
                    r[j] = True
                    left -= cost[j]
            out[f"{name}_{key}_topup"] = np.flatnonzero(r).tolist()
            print(f"{name} {key}: {len(inside)} kept + {int(r.sum()) - len(inside)} added, "
                  f"D {cost[r].sum():.4f}", flush=True)
    (rc.OUT / f"topup_clearings_{tag}.json").write_text(json.dumps(out))
