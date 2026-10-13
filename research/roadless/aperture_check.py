"""Guards of the grid fix (lifted.StepWidth: SmallestCell, today's, and Aperture; NOTES, "The grid
fix: steps that conduct by their open width"), each with its fault injection (a patch in one
process, no file edited): a guard passes on the code as it is and must fail under its fault.

    S=research/roadless/aperture_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S identity <ids,> [fault]
    uv run python $S open <ids,> [fault]
    uv run python $S channel - [fault]
    uv run python $S cracks - [fault]
    uv run python $S connect <ids,> [fault]
    uv run python $S convex <ids,> [fault]
    uv run python $S gradient <ids,> [fault]

`identity`: with today's rule selected the numbers of width_check's REFERENCE (57, computed at
3f74f5fa before the cells strategy and the step rule existed) reproduce bit for bit
(width_check.identity: the scorer's baseline and placement, a random clearing's operator, P, J_2
and P_{1/2}, the relaxation's value and gradient, the greedy's tension, current_map.flow, the
sightline metric's). Fault `ulp`: SmallestCell's widths one ulp up.

`open`: an open field gives today's operator exactly. On synthetic rectangles wholly inside the
block (every sub-sample open, OPEN_SHAPES), lifted.edges under each Aperture equals SmallestCell's,
family for family, bit for bit; on real blocks with every footprint open (the no-buildings field),
every along edge whose cells are wholly open (inside the block, every sub-sample) has today's
weight bit for bit (the edges at the block's ragged edge differ, as the rule intends: counted).
Fault `sum`: the step by the sum over its tracks, not their mean.

`channel`: a straight open channel (checks.channel's polygon, no walls: the block is the channel)
of width W and length L aligned with axis k conducts m_k W / L in layer k at every lattice angle:
the layer's along edges alone form chains along v_k, each a series of steps, and their parallel
sum (each chain's conductance between its first and last cell, over the chains that run the
channel's length) over m_k W / L is within CHANNEL_TOL of 1, for each rule, at the K = 8 lattice
angles in [0, 180) and W in CHANNEL_W. Fault `max-track`: the step by its best track (any open
track conducts the whole step).

`cracks`: synthetic pairs of cells: two cells whose open sub-samples do not touch conduct nothing
(axis-aligned, diagonal and knight steps, several placements; today's rule joins every one of
them); a crack one sub-sample wide on an axis-aligned step, running from centre to centre,
conducts 1/S of it. Fault `smallest`: the aperture's widths replaced by today's.

`connect`: on blocks, the cells the aperture reaches from the street (lifted.grounded) against
those a 4-connected flood fill of open sub-samples reaches (width.fine_bottleneck's), every
disagreement classified: a cell only the aperture reaches must lie in a pocket of the flood fill
joined to the street's side through a split cell (a cell whose open sub-samples form two parts,
one in the street's flood component and one not, joined by the cell's one node); a cell only the
flood fill reaches is reported by kind (its 2 x 2 centre footprint: an edge sliver no track from
a neighbour's centre reaches; else behind a bent passage, open but with no straight track to a
reached cell). The guard fails on a cell only the aperture reaches that no split cell explains.
Fault `diagonal-leak`: tracks traced through sub-sample centres only (a diagonal step through two
sub-samples touching at a corner counts as open).

`convex`: under the Aperture, P is convex in the clearing fractions (q 1): along CONVEX_LINES
random segments, every along edge's width has second differences at most CONVEX_TOL (concave) and
P's at least -CONVEX_TOL of P. Fault `square`: a track by the square of its smallest factor
(equal at every 0/1 state; not concave).

`gradient`: the relaxation's gradient under the Aperture (relax.Relaxation.value_grad: SIMP q 1
and 3, P and J_2, eps 1e-4, a random fractional clearing) against central differences (step
FD_STEP, solves to FD_RTOL from zero) on GRAD_BUILDINGS buildings, within GRAD_TOL of the largest.
Fault `no-spread`: the openness's vjp without the spread S^2 / n (a footprint sub-sample's openness
per unit of its cell's open fraction).
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402
import width_check  # noqa: E402

RULES = tuple(lifted.steps_of(t) for t in ("ap", "apw", "apws", "apwfs"))
SPLIT_RULES = ("aps", "apws", "apwfs")     # `connect`: the rules whose split cells conduct
                                           # through one part
FACE_RULES = ("apwfs",)                    # ... and whose axis steps read faces, the anchor part
                                           # too: its reach is the flood fill's of what it reads
SEED = 20261013
EPS = 1e-4                # SIMP base's eps world
H = 0.5


def _params(steps: lifted.StepWidth) -> lifted.Params:
    return lifted.Params(3.0, 8, solver=lifted.CpuAMG(), cells=lifted.OpenFraction(), steps=steps)


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _label(rule: lifted.StepWidth) -> str:
    return rule.name or "today's"


# ------------------------------------------------------------------ identity

def identity(ids: list[str]) -> list[str]:
    return width_check.identity(ids)


def _ulp_widths(self, grid, o, kap, p):
    for frac in _SMALLEST_WIDTHS(self, grid, o, kap, p):
        yield np.nextafter(frac, np.inf)


_SMALLEST_WIDTHS = lifted.SmallestCell.widths


# ------------------------------------------------------------------ open field

OPEN_SHAPES = ((5, 7), (9, 6), (12, 12))     # rectangles of cells, every sub-sample open


def synthetic(sub_open: np.ndarray, ground: tuple[tuple[int, int], ...] = ()) -> lifted.Grid:
    """A uniform Grid every sub-sample of which is inside the block, `sub_open` (ny S, nx S) the
    open ones (the rest footprint); `ground` the ground cells."""
    S = 4
    nyS, nxS = sub_open.shape
    ny, nx = nyS // S, nxS // S
    isub = np.ones((ny, nx, S * S), dtype=bool)
    bsub = ~lifted._to_cells(sub_open, S)
    building = bsub.mean(axis=-1) > 0.5
    g = np.zeros((ny, nx), dtype=bool)
    for r, c in ground:
        g[r, c] = True
    X, Y = np.meshgrid(np.arange(nx) * H, np.arange(ny) * H)
    return lifted.Grid(x0=0.0, y0=0.0, h=H, inside=isub.any(axis=-1), building=building,
                       ground=g, dist_b=ndimage.distance_transform_edt(~building) * H,
                       xy=np.stack([X, Y], axis=-1), isub=isub, bsub=bsub, S=S)


def _same_families(f0: list, f1: list) -> bool:
    return len(f0) == len(f1) and all(
        all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        for a, b in zip(f0, f1, strict=True))


def open_field(ids: list[str]) -> list[str]:
    bad = []
    today = _params(lifted.SmallestCell())
    for ny, nx in OPEN_SHAPES:
        g = synthetic(np.ones((ny * 4, nx * 4), dtype=bool))
        f0 = list(lifted.edges(g, g.ff0, today))
        for rule in RULES:
            same = _same_families(f0, list(lifted.edges(g, g.ff0, _params(rule))))
            print(f"open {ny} x {nx} cells, {rule.name}: lifted.edges equal to today's family for "
                  f"family, bit for bit: {same}", flush=True)
            if not same:
                bad.append(f"{ny}x{nx} {rule.name}")
    for bid in ids:
        [b] = common.build_blocks([bid])
        g = lifted.Grid.of(*lifted._geometry(b), H, offset=lifted.OFFSET)
        o = g.isub.mean(axis=-1).ravel()                 # every footprint open
        whole = (g.isub.all(axis=-1)).ravel()
        pat = g.pattern(8, np)
        w0 = list(lifted.SmallestCell().widths(g, o, o, today))
        for rule in RULES:
            diff = n_whole = n_edge = 0
            wr = rule.widths(g, o, o, _params(rule))
            for (a, b_, line, _wk), x0, x1 in zip(pat.along, w0, wr, strict=True):
                inner = whole[a] & whole[b_]
                for li in line:
                    inner &= whole[li]
                n_whole += int(inner.sum())
                diff += int((x0[inner] != x1[inner]).sum())
                n_edge += int((~inner & (x0 != x1)).sum())
            ok = diff == 0
            print(f"{bid} {rule.name}: every footprint open, {n_whole} edges whose cells are "
                  f"wholly open, {diff} differ from today's; {n_edge} edges at the ragged block "
                  f"edge differ -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} {rule.name}")
    return bad


def _sum_means(flat, ax, xp):
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        acc = m if acc is None else acc + m
    return acc if ax.coarse is None else xp.where(ax.coarse, 1.0, acc)


# ------------------------------------------------------------------ channel

CHANNEL_W = (1.0, 2.0, 4.0)
CHANNEL_L = 40.0
CHANNEL_EXACT = 1e-9          # an axis-aligned channel: m_k W exactly (its walls on sub-sample
                              # boundaries or not, every track is either in the open or not)
CHANNEL_MARGIN = 0.2          # an oblique one: the width lost at each wall, at most (m): half a
                              # sub-sample of the wall's rasterization, a track's half-thickness
                              # (its staircase, up to 0.71 of a sub-sample across) and half the
                              # tracks' spacing, about 0.19 m at 45 deg


def _layer_conduction(g: lifted.Grid, k: int, p: lifted.Params, e: np.ndarray,
                      L: float) -> float:
    """Layer k's conductance per unit length along the channel's middle half (clear of the
    rasterized ends): its along edges form chains along v_k, parallel to the walls; each chain's
    edges with both cells in the middle half are a series, conducting span / sum(1 / w) per unit
    length (span: the distance its first to its last cell covers), the chains in parallel. A
    channel of width W conducts m_k W per unit length."""
    o = g.ff0.ravel()
    fam = list(lifted.along_edges(g, o, p))
    _k, a, b, w = fam[k]
    cell = np.flatnonzero(o > 0)
    a, b = cell[a], cell[b]
    s = g.xy.reshape(-1, 2) @ e                           # position along the channel
    mid = (s[a] >= L / 4) & (s[b] <= 3 * L / 4)
    a, b, w = a[mid], b[mid], w[mid]
    n = o.size
    nxt = -np.ones(n, dtype=np.int64)
    nxt[a] = b
    wt = np.zeros(n)
    wt[a] = w
    has_prev = np.zeros(n, dtype=bool)
    has_prev[b] = True
    step = g.h * float(np.hypot(*lifted.axes(p.K)[0][k]))
    total = 0.0
    for start in np.flatnonzero((nxt >= 0) & ~has_prev):
        R, c = 0.0, start
        while nxt[c] >= 0:
            R += 1.0 / wt[c]
            c = nxt[c]
        span = s[c] - s[start]
        if span > L / 2 - 2 * step:                      # runs the middle half's length
            total += span / R
    return total


def channel(_ids: list[str]) -> list[str]:
    """Each rule's layer-k conductance per unit length of the aligned channel over m_k W (W_eff /
    W): the aperture passes when it is 1 at the axis-aligned angles and W_eff lies in [W - 2
    CHANNEL_MARGIN, W] at the oblique ones; today's rule is reported beside it (it meets the
    first, not the second: an oblique step takes the smallest open fraction of four cells)."""
    bad = []
    v, th, m, _gap = lifted.axes(8)
    for k in range(8):
        e = np.array([np.cos(th[k]), np.sin(th[k])])
        nrm = np.array([-e[1], e[0]])
        aligned = int(v[k, 0]) == 0 or int(v[k, 1]) == 0
        for W in CHANNEL_W:
            corners = [-nrm * W / 2, e * CHANNEL_L - nrm * W / 2, e * CHANNEL_L + nrm * W / 2,
                       nrm * W / 2]
            poly = Polygon(corners)
            g = lifted.Grid.of(poly, [], [LineString([corners[1], corners[2]])], H, band_m=1.0)
            want = m[k] * W
            today = _layer_conduction(g, k, _params(lifted.SmallestCell()), e, CHANNEL_L) / want
            got = {r.name: _layer_conduction(g, k, _params(r), e, CHANNEL_L) / want
                   for r in RULES}
            if aligned:
                ok = all(abs(x - 1.0) <= CHANNEL_EXACT for x in got.values())
            else:
                ok = all(W - 2 * CHANNEL_MARGIN <= x * W <= W + CHANNEL_EXACT
                         for x in got.values())
            print(f"axis {k} ({np.degrees(th[k]):.2f} deg) W {W:g}: layer {k}'s conductance per "
                  f"metre over m_k W: today's {today:.4f}, " + ", ".join(
                      f"{n} {x:.4f} (W_eff {x * W:.3f} m)" for n, x in got.items())
                  + f" -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"axis {k} W {W:g}")
    return bad


def _max_track(flat, ax, xp):
    best = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        best = m if best is None else xp.maximum(best, m)
    return best if ax.coarse is None else xp.where(ax.coarse, 1.0, best)


# ------------------------------------------------------------------ cracks

def _pair(dx: int, dy: int, a_open: list[tuple[int, int]], b_open: list[tuple[int, int]],
          line_open: str) -> lifted.Grid:
    """A 3 x 4 block of cells, a at (row 1, col 1 if dx >= 0 else 2), b at a + (dx, dy) (cells),
    the open sub-samples of a and b as (row, col) within the cell; the cells the step crosses
    wholly open (`line_open` "all") or wholly footprint ("none"); every other cell footprint."""
    S = 4
    ny, nx = 4, 5
    sub = np.zeros((ny * S, nx * S), dtype=bool)
    ra, ca = 1, 1 if dx >= 0 else 3
    rb, cb = ra + dy, ca + dx
    for (r, c), cells in (((ra, ca), a_open), ((rb, cb), b_open)):
        for i, j in cells:
            sub[r * S + i, c * S + j] = True
    if line_open == "all":
        for ix, iy in lifted._line_cells(dx, dy):
            sub[(ra + iy) * S:(ra + iy + 1) * S, (ca + ix) * S:(ca + ix + 1) * S] = True
    return synthetic(sub)


def _width_of(g: lifted.Grid, k: int, rule: lifted.StepWidth, dx: int, dy: int) -> float:
    ny, nx = g.inside.shape
    ra, ca = 1, 1 if dx >= 0 else 3
    a = ra * nx + ca
    o = g.ff0.ravel()
    pat = g.pattern(8, np)
    w = list(rule.widths(g, o, o, _params(rule)))[k]
    i = np.flatnonzero(pat.along[k][0] == a)
    return float(w[i[0]]) if len(i) else 0.0


S4 = 4
CRACK_CASES = (
    # (name, axis, a's open sub-samples, b's, line cells, the aperture's width)
    ("axis 0, a crack one sub-sample wide, centre to centre", 0,
     [(1, j) for j in range(S4)], [(1, j) for j in range(S4)], "none", 1 / S4),
    ("axis 0, open rows that do not meet (a's row 0, b's row 3)", 0,
     [(0, j) for j in range(S4)], [(3, j) for j in range(S4)], "none", 0.0),
    ("axis 0, a's open half far from b, b's near a", 0,
     [(i, j) for i in range(S4) for j in (0, 1)], [(i, j) for i in range(S4) for j in (0, 1)],
     "none", 0.0),
    ("axis 4, open columns that do not meet (a's col 0, b's col 3)", 4,
     [(i, 0) for i in range(S4)], [(i, 3) for i in range(S4)], "none", 0.0),
    ("axis 2 (1, 1), a and b open, the cells crossed footprint", 2,
     [(i, j) for i in range(S4) for j in range(S4)], [(i, j) for i in range(S4) for j in range(S4)],
     "none", 0.0),
    ("axis 2 (1, 1), a and b open only at the corners away from each other", 2,
     [(0, 0)], [(3, 3)], "all", 0.0),
    ("axis 1 (2, 1), a and b open, the cells crossed footprint", 1,
     [(i, j) for i in range(S4) for j in range(S4)], [(i, j) for i in range(S4) for j in range(S4)],
     "none", 0.0),
)


def cracks(_ids: list[str]) -> list[str]:
    bad = []
    v, _th, _m, _gap = lifted.axes(8)
    for name, k, a_open, b_open, line, want in CRACK_CASES:
        dx, dy = int(v[k, 0]), int(v[k, 1])
        g = _pair(dx, dy, a_open, b_open, line)
        today = _width_of(g, k, lifted.SmallestCell(), dx, dy)
        got = [_width_of(g, k, r, dx, dy) for r in RULES]
        ok = all(abs(x - want) < 1e-15 for x in got)
        print(f"{name}: today's {today:.4f}, " + ", ".join(
            f"{r.name} {x:.4f}" for r, x in zip(RULES, got, strict=True))
              + f" (want {want:.4f}) -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(name)
    return bad


def _smallest_widths(self, grid, o, kap, p):
    yield from lifted.SmallestCell().widths(grid, o, o if kap is None else kap, p)


# ------------------------------------------------------------------ connect

def flood(g: lifted.Grid, keep: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 4-connected flood fill of open sub-samples from the ground cells'
    (width.fine_bottleneck's connectivity), over the sub-samples `keep` (cells, S*S) only if
    given: (per cell, reached; the fine lattice's component labels; per label, whether it holds a
    ground cell's open sub-sample)."""
    S = g.S
    op = g.isub & ~g.bsub
    if keep is not None:
        op = op & keep.reshape(op.shape).astype(bool)
    opened = lifted._fine(op, S)
    gr = lifted._fine(np.repeat(g.ground[..., None], S * S, axis=-1), S)
    lab, n = ndimage.label(opened)
    ok = np.zeros(n + 1, dtype=bool)
    ok[lab[opened & gr]] = True
    ok[0] = False
    return lifted._cell_sum(ok[lab].astype(float), S) > 0, lab, ok


def split_cells(g: lifted.Grid) -> np.ndarray:
    """(ny, nx): how many 4-connected parts each cell's open sub-samples form (0 if none)."""
    S = g.S
    op = (g.isub & ~g.bsub).reshape(-1, S, S)
    n = np.zeros(len(op), dtype=np.int64)
    for i in np.flatnonzero(op.any(axis=(1, 2))):
        n[i] = ndimage.label(op[i])[1]
    return n.reshape(g.inside.shape)


def classify(g: lifted.Grid, reach: np.ndarray) -> dict:
    """The disagreements of `reach` (the aperture's) with the flood fill, classified (the
    module's docstring, `connect`)."""
    S = g.S
    fl, lab, okc = flood(g)
    parts = split_cells(g)
    labc = lifted._to_cells(lab, S)
    join = np.zeros(g.inside.shape, dtype=bool)
    for r, c in zip(*np.nonzero(parts >= 2), strict=True):
        ls = labc[r, c][labc[r, c] > 0]
        if okc[ls].any() and not okc[ls].all():
            join[r, c] = True
    jl = np.zeros(len(okc), dtype=bool)
    jl[labc[join][labc[join] > 0]] = True
    op_only = reach & ~fl
    explained = np.zeros(g.inside.shape, dtype=bool)
    for r, c in zip(*np.nonzero(op_only), strict=True):
        ls = labc[r, c][labc[r, c] > 0]
        explained[r, c] = bool(jl[ls].any())
    fl_only = fl & ~reach
    centre = g.bsub.reshape(*g.inside.shape, S, S)[:, :, S // 2 - 1:S // 2 + 1,
                                                   S // 2 - 1:S // 2 + 1].all(axis=(2, 3))
    return dict(reach=int(reach.sum()), flood=int(fl.sum()), split=int((parts >= 2).sum()),
                joining=int(join.sum()), op_only=int(op_only.sum()),
                op_only_explained=int((op_only & explained).sum()),
                flood_only=int(fl_only.sum()), sliver=int((fl_only & centre).sum()),
                bent=int((fl_only & ~centre).sum()),
                sliver_open_max=float(g.ff0[fl_only & centre].max()) if (fl_only & centre).any()
                else 0.0)


def connect(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        g = lifted.Grid.of(*lifted._geometry(b), H, offset=lifted.OFFSET)
        for rule in (lifted.SmallestCell(), *RULES):
            reach = lifted.grounded(g, g.ff0, _params(rule))
            d = classify(g, reach)
            if rule.name in FACE_RULES:
                # its axis-aligned steps join exactly the cells whose read sub-samples touch
                # across a face, and its cells' read parts are each one: its reach is the flood
                # fill of the sub-samples it reads (split cells' minor parts closed), exactly
                keep = rule.parts.keep(_params(rule).cells.sub_factor(
                    g, g.ff0, lifted.CpuAMG()), g, np)
                exact = int((reach != flood(g, keep)[0]).sum())
                print(f"{bid} {rule.name}: reach against the flood fill of the sub-samples it "
                      f"reads: {exact} cells differ -> {'ok' if exact == 0 else 'FAILS'}",
                      flush=True)
                if exact:
                    bad.append(f"{bid} {rule.name} exact")
            # a rule whose split cells conduct through one part joins nothing the flood fill does
            # not: no cell only it reaches; one whose node joins the parts, only those explained
            ok = d["op_only"] == (0 if rule.name in SPLIT_RULES else d["op_only_explained"])
            verdict = ("ok" if ok else "FAILS") if rule.name else (
                "not required: the rest are its phantom passages")
            print(f"{bid} {_label(rule)}: reach {d['reach']}, flood fill {d['flood']}; only the "
                  f"operator {d['op_only']} (in a pocket joined through a split cell: "
                  f"{d['op_only_explained']}; split cells {d['split']}, joining street and pocket "
                  f"{d['joining']}); only the flood fill {d['flood_only']} (edge slivers, the "
                  f"centre footprint: {d['sliver']}, open share at most "
                  f"{d['sliver_open_max']:.3f}; "
                  f"behind a bent passage {d['bent']}) -> {verdict}", flush=True)
            if not ok and rule.name:
                bad.append(f"{bid} {rule.name}")
    return bad


def _centre_tracks(dx: int, dy: int, S: int, band: lifted.Band) -> lifted.Tracks:
    """axis_tracks traced through sub-sample centres only: each track the sub-samples whose centre
    its line passes (Bresenham-like, one per row or column crossed), so a diagonal step through
    two sub-samples touching at a corner is open."""
    tr = _TRACKS(dx, dy, S, band)
    out = []
    for t in tr.tracks:
        keep = list(t)
        # keep every other square of a staircase: the ones a line through centres would visit
        if abs(dx) > 0 and abs(dy) > 0:
            keep = keep[::2]
        out.append(tuple(keep))
    return lifted.Tracks(cells=tr.cells, tracks=tuple(out))


_TRACKS = lifted.axis_tracks


# ------------------------------------------------------------------ convex

CONVEX_LINES = 4
CONVEX_POINTS = 7
CONVEX_TOL = 1e-9


def _clearing(b, rule: lifted.StepWidth) -> clear.Clearing:
    return clear.Clearing(b, lifted.UniformMesh(H, offset=lifted.OFFSET), _params(rule),
                          population=common.POPULATIONS["area"], demand=common.PopulationDemand())


def convex(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for rule in RULES:
            c = _clearing(b, rule)
            rel = relax.Relaxation(c, 1.0, q=1.0, rtol=width_check.FD_RTOL, eps=EPS)
            rng = np.random.default_rng(SEED + 2)
            worst_w, worst_P = -np.inf, np.inf
            for _ in range(CONVEX_LINES):
                a, z = rng.uniform(0.0, 1.0, c.n), rng.uniform(0.0, 1.0, c.n)
                ws, P = [], []
                for t in np.linspace(0.0, 1.0, CONVEX_POINTS):
                    x = a + t * (z - a)
                    op = rel.open_of(x)
                    op_eps = (op + EPS * (c.full - op)).ravel()
                    ws.append(np.concatenate(list(rule.widths(c.sc.grid, op_eps, op_eps, c.p))))
                    P.append(width_check._value(rel, x))
                W2 = np.diff(np.array(ws), 2, axis=0)
                P2 = np.diff(np.array(P), 2)
                worst_w = max(worst_w, float(W2.max()))
                worst_P = min(worst_P, float((P2 / np.max(P)).min()))
            ok = worst_w <= CONVEX_TOL and worst_P >= -CONVEX_TOL
            print(f"{bid} {rule.name}: on {CONVEX_LINES} lines, the step widths' largest second "
                  f"difference {worst_w:.1e} (concave: <= {CONVEX_TOL:g}), P's smallest "
                  f"{worst_P:.1e} of P (convex: >= -{CONVEX_TOL:g}) -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} {rule.name}")
    return bad


def _square_means(flat, ax, xp):
    """A track by the square of its smallest factor (equal at every 0/1 state; convex, not
    concave, between). A track by the product of its factors (width_check's `product`) is the
    same kind of fault but leaves the eps world's system too ill-conditioned to solve (a closed
    track's eps^9 -- eps^13: AMG-CG did not converge)."""
    acc = None
    for tr in ax.tracks:
        m = None
        for slot, sub in tr:
            v = flat[ax.base[slot] + sub]
            m = v if m is None else xp.minimum(m, v)
        acc = m * m if acc is None else acc + m * m
    w = acc / len(ax.tracks)
    return w if ax.coarse is None else xp.where(ax.coarse, 1.0, w)


# ------------------------------------------------------------------ gradient

GRAD_BUILDINGS = 6
GRAD_Q = (1.0, 3.0)
FD_STEP = 1e-5            # the widths are piecewise linear (a min per track): small steps
FD_RTOL = 1e-9
GRAD_TOL = 1e-5


def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for rule in RULES:
            c = _clearing(b, rule)
            rng = np.random.default_rng(SEED)
            x0 = rng.uniform(0.1, 0.4, c.n)
            pick = rng.choice(c.n, GRAD_BUILDINGS, replace=False)
            for power in (1.0, 2.0):
                for q in GRAD_Q:
                    rel = relax.Relaxation(c, power, q=q, rtol=FD_RTOL, eps=EPS)
                    rel._u = None
                    _, g = rel.value_grad(x0)
                    fd = []
                    for j in pick:
                        e = np.zeros(c.n)
                        e[j] = FD_STEP
                        fd.append((width_check._value(rel, x0 + e)
                                   - width_check._value(rel, x0 - e)) / (2 * FD_STEP))
                    err = float(np.abs(np.array(fd) - g[pick]).max() / np.abs(g[pick]).max())
                    ok = err <= GRAD_TOL
                    print(f"{bid} {rule.name} J_{power:g} SIMP q {q:g}: gradient against central "
                          f"differences on {GRAD_BUILDINGS} buildings, {err:.1e} of the largest "
                          f"-> {'ok' if ok else 'FAILS'}", flush=True)
                    if not ok:
                        bad.append(f"{bid} {rule.name} p{power:g} q{q:g}")
    return bad


def _no_spread_vjp(grid, A, xp):
    st = lifted._sub_state(grid, xp)
    return (st.cov * A).sum(axis=1)


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "identity": identity, "open": open_field, "channel": channel, "cracks": cracks,
    "connect": connect, "convex": convex, "gradient": gradient}
FAULTS: dict[str, dict[str, Callable]] = {
    "identity": {"ulp": lambda: _patched(lifted.SmallestCell, "widths", _ulp_widths)},
    "open": {"sum": lambda: _patched(lifted, "_track_means", _sum_means)},
    "channel": {"max-track": lambda: _patched(lifted, "_track_means", _max_track)},
    "cracks": {"smallest": lambda: _patched(lifted.Aperture, "widths", _smallest_widths)},
    "connect": {"diagonal-leak": lambda: _patched(lifted, "axis_tracks", _centre_tracks)},
    "convex": {"square": lambda: _patched(lifted, "_track_means", _square_means)},
    "gradient": {"no-spread": lambda: _patched(lifted, "_openness_vjp", _no_spread_vjp)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else a[1].split(","), a[2] if len(a) == 3 else None)
