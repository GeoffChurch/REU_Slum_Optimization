"""Whether the cut test (ears.cuts, owner 2026-11-22: rank only ears build can build) agrees with
build's refusal rule (ears.offer: an ear is built only where its face's faces of growth.FACE_M2
and more rise). A run of a world and paths plan (atlas/crossings' worlds) is built as ears.build
does, and at each state, beside the ears Paths offers:

  kept: each ear it offers (each of the best `pairs` sources' best ranked target): whether build
    refuses it;
  dropped: for the same sources and the next best, up to DROPPED of them, the best target
    the cut test drops (its ring the source's), its route drawn and its corridor built as an
    offered ear's: whether build would have accepted it.

Each tested ear's row: the pieces the cut test predicts (ears.cuts, gross of other holes), its
path's length, and the faces build sees (whether they rise, and the smaller new piece's area), so
the allowance (ears.CUT_ALLOW) can be judged offline at any value.

    C=research/roadless/atlas/routers/cutrule.py       # from the repo root, PYTHONPATH=.:src
    python $C run <world> <plan>     # rows to rows/cutrule/<world>__<plan>.parquet
    python $C table                  # results/cutrule.csv: the error rates
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[0] / "crossings"))
import crossings as cx  # noqa: E402
import ears  # noqa: E402
import growth as gr  # noqa: E402

ROWS = HERE / "rows" / "cutrule"
DROPPED = 16           # dropped: sources tested per state (the best by their best dropped target)
ALLOWS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)   # table: the allowances judged


def split_of(e: ears.EarWorld, faces: ears.Face, f: int, tris: Any) -> tuple[bool, float]:
    """Whether building `tris` raises the faces of FACE_M2 and more (ears.offer's rule), and the
    smaller of the new faces' areas in face f's place (0 if none)."""
    g = e.g
    mark = int(g.step.max()) + 1
    g.step[tris] = mark
    nf = e.faces()
    g.step[tris] = -1
    was = faces.lab == f
    new = np.unique(nf.lab[was & (nf.lab >= 0)])
    areas = np.sort(nf.area[new])[::-1]
    return bool(nf.n > faces.n), float(areas[1]) if len(areas) > 1 else 0.0


@dataclass(kw_only=True)
class Audit:
    """Paths (`inner`) whose states are audited (module docstring); rows appended to `rows`."""
    inner: ears.Paths
    rows: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def pairs(self) -> int:
        return self.inner.pairs

    def ears(self, e: ears.EarWorld, faces: ears.Face, gain: Any, *, cost: gr.Cost,
             scale: float) -> list[ears.Ear]:
        out = self.inner.ears(e, faces, gain, cost=cost, scale=scale)
        step = len({r["step"] for r in self.rows})
        p = self.inner
        ter, src, pf, bnd, ground, arcs = p.routing(e, faces, gain)
        ra = e.ra
        for q in out:
            if q.spur:
                continue
            a, t = int(q.path[0]), int(q.path[-1])
            reach = next(iter(p.router.reaches(ter, np.array([a]))))
            P, Q = ears.cuts(arcs, int(pf[a]), a, np.array([t]), reach.area[[t]])
            ok, small = split_of(e, faces, q.face, q.tris)
            self.rows.append(dict(step=step, kind="kept", P=P[0], Q=Q[0],
                                  L=float(reach.length()[t]), split=ok, small=small))
        # the dropped: per source its best target the cut test drops
        fld = p.attract.field(e, P0=e.g.P0)
        lex = ter.lex()
        best: list[tuple[tuple[int, float], int, int, Any, float, float, float]] = []
        for reach in p.router.reaches(ter, src):
            a = reach.src
            d = reach.cost
            own = bnd & (arcs.ring == arcs.ring[a])
            crow = np.hypot(*(ra.xy - ra.xy[a]).T)
            tg = np.flatnonzero(p.ends.targets(own, ground, crow, ear_m=p.ear_m)
                                & np.isfinite(d) & (pf == pf[a]) & bnd)
            if not len(tg):
                continue
            ln = reach.length()[tg]
            X, Y = ears.cuts(arcs, int(pf[a]), int(a), tg, reach.area[tg])
            drop = ~(np.minimum(X, Y) - ears.CUT_ALLOW * gr.PATCH_R * ln >= gr.FACE_M2)
            if not drop.any():
                continue
            tg, ln, X, Y = tg[drop], ln[drop], X[drop], Y[drop]
            cr = fld.credits(int(a), tg, reach)
            tier, r = ears.rank(reach.gain[tg] + scale * cr, d[tg], ln if lex else None)
            k = int(np.lexsort((-r, -tier))[0])
            best.append(((int(tier[k]), float(r[k])), int(a), int(tg[k]), reach,
                         float(X[k]), float(Y[k]), float(ln[k])))
            best.sort(key=lambda z: (-z[0][0], -z[0][1]))
            del best[DROPPED:]
        for _, a, t, reach, X_, Y_, L in best:
            rt = reach.route(t)
            tris = e.corridor(rt.line, spur=False)
            if not len(tris):
                continue
            ok, small = split_of(e, faces, int(pf[a]), tris)
            self.rows.append(dict(step=step, kind="dropped", P=X_, Q=Y_, L=L, split=ok,
                                  small=small))
        return out


def run(world: str, spec: str) -> None:
    t0 = time.time()
    w = cx.world_of(world)
    plan = ears.Plan.of(spec)
    assert isinstance(plan.candidates, ears.Paths)
    aud = Audit(inner=plan.candidates, rows=[])
    b = w.block()
    metric = ears.metrics.metric_of(ears.metrics.RUNTIME, device="cpu")
    e = ears.EarWorld(gr.growth_of(b, metric, contrast=plan.contrast))
    out = HERE / "rows" / "cutrule_build" / w.name.replace(":", "_") / plan.name
    ears.build(e, replace(plan, candidates=aud), out=out, log=lambda s: print(s, flush=True))
    ROWS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(aud.rows).assign(world=w.name, plan=plan.name)
    df.to_parquet(ROWS / f"{w.name.replace(':', '_')}__{plan.name}.parquet")
    print(f"done: {len(df)} ears tested ({time.time() - t0:.0f}s)", flush=True)


def table() -> None:
    df = pd.concat([pd.read_parquet(p) for p in sorted(ROWS.glob("*.parquet"))],
                   ignore_index=True)
    df["b"] = df.plan.str.extract(r"\.b([0-9.]+|inf)\.")[0]
    rows = []
    for allow in ALLOWS:
        keep = (np.minimum(df.P, df.Q) - allow * gr.PATCH_R * df.L >= gr.FACE_M2).to_numpy()
        for (b, kind), sub in df.groupby(["b", "kind"]):
            k = keep[sub.index.to_numpy()]
            rows.append(dict(allow=allow, b=b, sample=kind, n=len(sub),
                             kept=int(k.sum()), kept_refused=int((k & ~sub.split).sum()),
                             dropped=int((~k).sum()),
                             dropped_accepted=int((~k & sub.split).sum())))
    t = pd.DataFrame(rows)
    t["kept_refused_frac"] = t.kept_refused / t.kept.clip(lower=1)
    t["dropped_accepted_frac"] = t.dropped_accepted / t.dropped.clip(lower=1)
    (HERE / "results").mkdir(exist_ok=True)
    t.to_csv(HERE / "results" / "cutrule.csv", index=False)
    print(t.to_string(index=False))


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "run":
        run(argv[1], argv[2])
    elif argv == ["table"]:
        table()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
