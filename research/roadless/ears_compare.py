"""Ears runs side by side at matched D (Idea A, spurs; Idea C, detour saving): per run, L_P at
each target D itself (L_P_at: linear between the states on either side, so runs whose ears step D
differently compare at one D), and at the first state reaching it L_P (and its recomputation from
tri_step), faces, the ears and spurs built, their length, the share of it in dead ends (a spur
whose corridor, ungrown at that state, leaves the face count as it is: it bounds one face only;
a spur an ear later closed, or one that ended near another road, bounds two and is on a loop),
and the worst-off home (the largest home escape time, and its population-weighted 95th
percentile); at the run's end, its wall time.

    uv run python research/roadless/ears_compare.py <block> <D,D,...> <run> [<run> ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ears  # noqa: E402
import growth as gr  # noqa: E402


def label(run: str) -> str:
    c = ears.Plan.of(run.rsplit(".v", 1)[0]).candidates
    if isinstance(c, ears.Paths):
        return f"{c.ends.name}.{c.attract.name}"
    return c.name


def main(bid: str, targets: list[float], runs: list[str]) -> None:
    e: ears.EarWorld | None = None
    rows = []
    for run in runs:
        d = gr.OUT / bid / run
        s = pd.read_parquet(d / "steps.parquet")
        p = pd.read_parquet(d / "picks.parquet")
        ts = pd.read_parquet(d / "tri_step.parquet")["step"].to_numpy()
        if e is None:
            e = ears.world_of(bid, ears.Plan.of(run.rsplit(".v", 1)[0]).contrast.h)
        g = e.g
        pop = g.m.pop
        for D in [*targets, None]:
            if D is None:
                k = len(s) - 1
            else:
                hit = np.flatnonzero(s.D.to_numpy() >= D)
                if not len(hit):
                    continue
                k = int(hit[0])
            on = (ts >= 0) & (ts < k)
            g.step[:] = np.where(on, ts, -1)
            st = g.state(on)
            u, P = g.solve(st)
            uh = g.m.home_u(st, u)
            ok = np.isfinite(uh)
            o = np.argsort(uh[ok])
            cw = np.cumsum(pop[ok][o]) / pop[ok].sum()
            p95 = float(uh[ok][o][np.searchsorted(cw, 0.95)])
            pk = p[p.step < k]
            n = e.faces().n
            dead = 0.0
            for _, q in pk[pk.spur].iterrows():
                mine = ts == q.step
                g.step[mine] = -1
                if e.faces().n == n:
                    dead += float(q.length)
                g.step[mine] = q.step
            km = float(pk.length.sum())
            spur_km = float(pk.length[pk.spur].sum())
            at = float(np.interp(D, s.D, s.L_P)) if D is not None else float(s.L_P[k])
            rows.append(dict(run=label(run), D_target="end" if D is None else D, step=k,
                             L_P_at=at, D=float(s.D[k]), L_P=float(s.L_P[k]),
                             L_P_check=1 - P / g.P0,
                             faces=int(s.faces[k]), ears=k, spurs=int(pk.spur.sum()),
                             km=km / 1e3, spur_share=spur_km / max(km, 1e-9),
                             dead_share=dead / max(km, 1e-9), worst_u=float(np.nanmax(uh)), u95=p95,
                             wall_s=float(s.t.iloc[-1]) if D is None else np.nan))
    df = pd.DataFrame(rows).sort_values(["D_target", "run"], key=lambda c: c.astype(str))
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2].split(",")], sys.argv[3:])
