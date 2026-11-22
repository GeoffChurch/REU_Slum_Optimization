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
ALLOWS = tuple(np.round(np.arange(0.8, 1.45, 0.05), 2))   # table: the allowances judged


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


def pinches(ra: ears.Raster, pf: Any, f: int, line: Any) -> int:
    """Diagonal steps of the centreline (sampled every PIX / 8) between two of face f's pixels
    whose two common neighbours are both out of it: where the raster's network touches itself
    at a corner and the path crosses it."""
    seg = np.diff(line, axis=0)
    n = np.maximum(np.ceil(np.hypot(*seg.T) / (ears.PIX / 8)).astype(np.int64), 1)
    pts = np.vstack([line[i] + seg[i] * (np.arange(k) / k)[:, None] for i, k in enumerate(n)]
                    + [line[-1:]])
    px = ears._pix_of(ra, pts)
    px = px[np.r_[True, px[1:] != px[:-1]]]
    ny = ra.shape[1]
    i, j = px // ny, px % ny
    di, dj = np.diff(i), np.diff(j)
    diag = np.flatnonzero((np.abs(di) == 1) & (np.abs(dj) == 1))
    o1 = (i[diag] + di[diag]) * ny + j[diag]
    o2 = i[diag] * ny + j[diag] + dj[diag]
    return int(((pf[o1] != f) & (pf[o2] != f)).sum())


def sides(ra: ears.Raster, pf: Any, line: Any) -> tuple[float, float]:
    """The face area the corridor (PATCH_R a side of the centreline) covers left and right of
    it, by samples every PIX / 4 along it and PIX / 4 across it (offline: the oracle)."""
    seg = np.diff(line, axis=0)
    out = np.zeros(2)
    for p, d in zip(line[:-1], seg, strict=True):
        L = float(np.hypot(*d))
        if L == 0:
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]])
        n = max(int(np.ceil(L / (ears.PIX / 4))), 1)
        ts = (np.arange(n) + 0.5) / n
        ofs = (np.arange(int(gr.PATCH_R / (ears.PIX / 4))) + 0.5) * (ears.PIX / 4)
        for k, sg in enumerate((1.0, -1.0)):
            pts = (p + ts[:, None] * d)[:, None, :] + sg * ofs[None, :, None] * nrm
            px = ears._pix_of(ra, pts.reshape(-1, 2))
            out[k] += float((pf[px] >= 0).mean()) * gr.PATCH_R * L
    return float(out[0]), float(out[1])


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
            cut = ears.cuts(arcs, int(pf[a]), a, np.array([t]), reach.area[[t]])
            ok, small = split_of(e, faces, q.face, q.tris)
            sl, sr = sides(ra, pf, q.line)
            self.rows.append(dict(step=step, kind="kept", P=cut.P[0], Q=cut.Q[0],
                                  left=bool(cut.left[0]), E=reach.cor[t, 0], S=reach.cor[t, 1],
                                  L=float(reach.length()[t]), split=ok, small=small,
                                  pinch=pinches(ra, pf, q.face, q.line), El=sl, Er=sr))
        # the dropped: per source its best target the cut test drops
        fld = p.attract.field(e, P0=e.g.P0)
        lex = ter.lex()
        best: list[tuple[tuple[int, float], int, int, Any, Any]] = []
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
            cut = ears.cuts(arcs, int(pf[a]), int(a), tg, reach.area[tg])
            X, Y = cut.net(reach.cor[tg])
            drop = ~(np.minimum(X, Y) >= gr.FACE_M2)
            if not drop.any():
                continue
            tg, ln = tg[drop], ln[drop]
            cr = fld.credits(int(a), tg, reach)
            tier, r = ears.rank(reach.gain[tg] + scale * cr, d[tg], ln if lex else None)
            k = int(np.lexsort((-r, -tier))[0])
            t = int(tg[k])
            row = dict(P=float(cut.P[drop][k]), Q=float(cut.Q[drop][k]),
                       left=bool(cut.left[drop][k]), E=float(reach.cor[t, 0]),
                       S=float(reach.cor[t, 1]), L=float(ln[k]))
            best.append(((int(tier[k]), float(r[k])), int(a), t, reach, row))
            best.sort(key=lambda z: (-z[0][0], -z[0][1]))
            del best[DROPPED:]
        for _, a, t, reach, row in best:
            rt = reach.route(t)
            tris = e.corridor(rt.line, spur=False)
            if not len(tris):
                continue
            ok, small = split_of(e, faces, int(pf[a]), tris)
            sl, sr = sides(ra, pf, rt.line)
            self.rows.append(dict(step=step, kind="dropped", **row, split=ok, small=small,
                                  pinch=pinches(ra, pf, int(pf[a]), rt.line), El=sl, Er=sr))
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


def net_min(df: pd.DataFrame, allow: float) -> Any:
    """The smaller piece less `allow` times the corridor's take on its side (ears.Cut.net)."""
    lt, rt = (df.E + df.S) / 2.0, (df.E - df.S) / 2.0
    sp = np.where(df.left, lt, rt)
    sq = np.where(df.left, rt, lt)
    return np.minimum(df.P - allow * sp, df.Q - allow * sq).to_numpy()


def table() -> None:
    df = pd.concat([pd.read_parquet(p) for p in sorted(ROWS.glob("*.parquet"))],
                   ignore_index=True)
    df["b"] = df.plan.str.extract(r"\.b([0-9.]+|inf)\.")[0]
    rows = []
    for allow in ALLOWS:
        keep = net_min(df, allow) >= gr.FACE_M2
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
