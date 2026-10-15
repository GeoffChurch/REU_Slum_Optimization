"""The complex rebuilt for every clearing (ComplexModel): the guards' and the calibration's
reference for the runtime's fixed mesh (gap_clear_check `baseline` and `roads`, gap_calib
`baseline`, `clearings` and `roads`), gap_flow's mesh of the standing footprints less a road's
corridor under a width form (`form_of`), the demand fixed at baseline.

This module was the gap complex's stage 2 (NOTES, "The gap complex: passability, enclosure and
flow without the raster"): the complex against the raster's isotropic model and the lifted metric,
the convergence and flow sweeps, the walled channel's laws; those finished studies' code went with
the raster (NOTES, "The complex as the runtime: the raster retired"), their numbers in NOTES and
gap_rows/summary_*.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402
import metrics  # noqa: E402

S_M = 0.125                       # the complex's outline spacing (stage 1's choice)
AREA = common.AreaPopulation()
PQ = common.PriorityDemand(0.5)
RAMP_S = 0.5                      # the ramp's length (m): full conductance 0.75 m from a wall


def form_of(token: str) -> gf.Width:
    """close<w> | shy<w> | ramp<w> (its s RAMP_S): gap_flow's width forms."""
    for prefix, make in (("close", lambda w: gf.Closing(w=w)), ("shy", lambda w: gf.Shy(w=w)),
                         ("ramp", lambda w: gf.Ramp(w=w, s=RAMP_S))):
        if token.startswith(prefix):
            return make(float(token[len(prefix):]))
    raise ValueError(f"unknown width form {token!r}: close<w>, shy<w> or ramp<w>")


class ComplexModel:
    """One block's complex under a width form (module docstring): the baseline's demand, escape
    times and lenses' references; `score` a clearing by the complex rebuilt."""

    def __init__(self, b, fs: gf.FlowSpec, form: gf.Width):
        self.b, self.form = b, form
        self.geo = gc.geometry(b)
        self.spec = metrics.spec_of(S_M)
        self.fs = fs
        n = len(self.geo.polys)
        self.n = n
        self.pop = AREA.weights(self.geo.polys)
        self.cost = self.pop / self.pop.sum()
        t0 = time.time()
        m0 = gf.mesh(self.geo, self.spec, self.fs, form, np.ones(n, dtype=bool), carve=None)
        c0 = gf.cells(m0)
        self.dem = gf.point_demand(self.geo, m0, c0)
        self.walled = self.dem.walled
        self.t_mesh = time.time() - t0
        sc = gf.Scores(m0, c0, self.dem)
        self.P0, self.u0 = sc.u_home(self.pop)
        self.inj_q = PQ.injection(self.pop, self.u0, self.walled)
        self.Pq0, _ = sc.u_home(self.inj_q)
        self.J2_0 = float(np.nansum(self.pop * self.u0 ** 2))
        self.cells0, self.nt0 = c0.n, m0.nt
        self.u_q0 = sc.u_home(self.inj_q)[1]
        self.diag0 = dict(merged=c0.merged, ground_obtuse=c0.ground_obtuse,
                          negative=c0.negative, skipped=m0.skipped, nt=m0.nt, cells=c0.n)
        self.t_base = time.time() - t0

    def score(self, standing: np.ndarray, carve) -> tuple[dict, np.ndarray]:
        """(the lenses of a clearing, its escape times per home)."""
        r, u, _live = self.scored(standing, carve)
        return r, u

    def scored(self, standing: np.ndarray, carve) -> tuple[dict, np.ndarray, np.ndarray]:
        """(the lenses of a clearing, its escape times per home, the homes it reaches: some open
        ground within ring_m of the outline reaching the street, gap_flow.live_homes)."""
        t0 = time.time()
        m = gf.mesh(self.geo, self.spec, self.fs, self.form, standing, carve=carve)
        c = gf.cells(m)
        sc = gf.Scores(m, c, self.dem)
        P, u = sc.u_home(self.pop)
        Pq, _ = sc.u_home(self.inj_q)
        live = gf.live_homes(self.geo, m, c)
        left = self.walled & ~live
        J2 = float(np.nansum(self.pop * u ** 2))
        return dict(P=P, Pq=Pq, L_P=1.0 - P / self.P0, L_J2=1.0 - (J2 / self.J2_0) ** 0.5,
                    **{"L_Pq0.5": 1.0 - Pq / self.Pq0},
                    enclosed=float(self.cost[left].sum()), n_enclosed=int(left.sum()),
                    unknowns=int((~c.ground).sum()), nt=m.nt, dropped=sc.dropped,
                    t=time.time() - t0), u, live

    def free_first(self, held: np.ndarray) -> np.ndarray:
        """gap_complex.free_first at the form's w from `held` (no budget): the held set grown (the
        three forms share stage 1's topology)."""
        fr = gc.free_first(self.geo, self.spec, w=self.form.w, grow=0.0, budget=1.0,
                           cost=self.cost, pop=self.pop, held0=held, cx0=None)
        return fr.held
