"""The road-transparency experiment on the 4-block sentinel (BACKLOG 5; the owner approved it
2026-10-12, "Leave out arterial"): do cycle_native, resistance_lp and clearance_looped build
better networks at equal displacement when the buildings the roadless optimum clears are made
cheaper to displace (the PRICE) and to route through (the ROUTE; reblock.transparency)?

An arm is a named, block-independent configuration: a preset of compare_config's all_methods
(common.arms(), every building opaque) and what the arm changes on it -- nothing (the base), one
of the method's own knobs, or a Discount: a set of buildings at a factor f, in the price, the
route or both. The sets: C, the buildings SIMP's default clears at D 0.05
(transparency_sets.SIMP_DEFAULT); C', the cheap preset's (transparency_sets.CHEAP); and the
control, random sets matched to C by footprint area in 20 strata of equal area mass
(transparency_sets.AreaMass), seeds 1 -- 5. PLAN lists every method's arms by their part in the
screen; `arms` writes them, with every value that can change a network, to
transparency_rows/arms.csv (committed). An arm's name keys its network and its rows.

The networks are proposed fresh (road_rescore.propose, no derive cache anywhere), cut to their
canonical prefixes by road_rescore.prefixes and scored by road_rescore.score, all under this
experiment's rows root, transparency_rows/ (item 4's road_rows/ is read, never written):

    S=research/roadless/transparency_run.py    # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S arms                         # write transparency_rows/arms.csv
    uv run python $S dry <id>                     # every arm's method on one block, no proposal
    uv run python $S propose <procs> <timeout h> <ids,>
    uv run python $S check-base <ids,>            # the base arms against item 4's networks
    uv run python $S prefixes <procs> <ids,>
    uv run python $S tars <dir> <ids,>            # a tar per block for score on the cluster
    uv run python $S records <procs> <ids,>       # displacement on C, the draws, the costs
    uv run python $S report                       # the screen, and the summaries

`propose` runs `propose-one <id> <arm>` children, `procs` at a time; an arm marked `reuse_item4`
(the desire-routed cycle_native) takes item 4's network of its preset instead (its code path is
unchanged under Opaque: NOTES). `tars` prints the cluster command that scores lenses (b) and (c).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import road_rescore as rr  # noqa: E402
import transparency_sets as ts  # noqa: E402

from reblock.contracts import Method  # noqa: E402
from reblock.methods.clearance import ClearanceReblocker  # noqa: E402
from reblock.methods.cycle_native import CycleNativeReblocker  # noqa: E402
from reblock.methods.loop_closure import LoopClosureRefiner  # noqa: E402
from reblock.methods.resistance_lp import ResistanceLPReblocker  # noqa: E402
from reblock.transparency import Transparency, TransparentSubstrate, factors_of  # noqa: E402

# ------------------------------------------------------------------ the experiment's configuration
ROWS = HERE / "transparency_rows"            # this experiment's rows root
SCRATCH = ROWS / "scratch"                   # networks, logs, progress (git-ignored)
ITEM4_ROWS = HERE / "road_rows"              # item 4's rows and networks, read only
ITEM4_NETS = ITEM4_ROWS / "scratch" / "networks"
SENTINEL = ("ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712")
MESH = "0.5"                                 # item 4's: h 0.5, uni (road_rescore's parameters)
F_LEVELS = (0.5, 0.1)                        # the main form's factors
F_ONE = 0.1                                  # the ablations', C''s and the random sets' factor
STRATIFICATION = ts.AreaMass(strata=20)      # the control's (owner's ruling 2026-10-12)
SEEDS = (1, 2, 3, 4, 5)
LENS_D = 0.05                                # (c) at area D 0.05: the screen's primary number
GUARD_D = (0.05, 0.10)                       # (a) at count D 0.05 and 0.10: the guard
VETO = 0.1                                   # an arm more than this below base on (c) is dropped
A_LOSS = 0.01                                # the guard's largest median (a) loss
CHEAP_WITHIN = 0.01                          # C' advances within this of C's mean gain
# propose's order only (no result depends on it): a guess at each method's cost per building
COST_GUESS = {"cycle_native": 2.5, "resistance_lp": 2.0, "clearance_looped": 1.0}


# ------------------------------------------------------------------ the sets
class SetChoice(Protocol):
    """Which buildings a discount makes cheaper, read for one block where the arm is built."""

    @property
    def label(self) -> str: ...

    def transparency(self, block_id: str, f: float) -> Transparency: ...

    def record(self) -> dict[str, object]: ...


@dataclass(frozen=True, kw_only=True)
class StoredClearing:
    """The buildings a stored roadless clearing clears: C (SIMP's default) or C' (the cheap
    preset's), transparency_sets.ClearedSet."""

    label: str
    rows: ts.StoredRows

    def transparency(self, block_id: str, f: float) -> ts.ClearedSet:
        return ts.ClearedSet(clearing=ts.Clearing.read(self.rows, block_id), f=f)

    def record(self) -> dict[str, object]:
        return dict(set=self.label, clearing=str(self.rows.rows), stratification="", seed=np.nan)


@dataclass(frozen=True, kw_only=True)
class MatchedRandom:
    """The control: a random set matched to a stored clearing (MatchedRandomSet)."""

    rows: ts.StoredRows
    stratification: ts.Stratification
    seed: int

    @property
    def label(self) -> str:
        return f"rand{self.seed}"

    def transparency(self, block_id: str, f: float) -> ts.MatchedRandomSet:
        return ts.MatchedRandomSet(clearing=ts.Clearing.read(self.rows, block_id),
                                   stratification=self.stratification, seed=self.seed, f=f)

    def record(self) -> dict[str, object]:
        return dict(set=self.label, clearing=str(self.rows.rows),
                    stratification=repr(self.stratification), seed=self.seed)


C = StoredClearing(label="simp", rows=ts.SIMP_DEFAULT)
C_CHEAP = StoredClearing(label="cheap", rows=ts.CHEAP)


# ------------------------------------------------------------------ what an arm changes
class Change(Protocol):
    """What an arm changes on its preset; applied per block (a discount reads the block's set)."""

    @property
    def label(self) -> str: ...

    def apply(self, method: Method, block_id: str) -> Method: ...

    def record(self) -> dict[str, object]: ...


_NO_DISCOUNT: dict[str, object] = dict(set="", clearing="", stratification="", seed=np.nan,
                                       f=np.nan, price=False, route=False)


@dataclass(frozen=True, kw_only=True)
class AsConfigured:
    """The preset as configured, every building opaque."""

    label: str

    def apply(self, method: Method, block_id: str) -> Method:
        return method

    def record(self) -> dict[str, object]:
        return dict(knob="", value=np.nan, **_NO_DISCOUNT)


@dataclass(frozen=True, kw_only=True)
class Shortlist:
    """cycle_native's `shortlist` (8 in the preset): the deepest parcels it tries a cycle to."""

    value: int

    @property
    def label(self) -> str:
        return f"shortlist{self.value}"

    def apply(self, method: Method, block_id: str) -> Method:
        if not isinstance(method, CycleNativeReblocker):
            raise TypeError(f"shortlist is cycle_native's, not {type(method).__name__}'s")
        return dataclasses.replace(method, shortlist=self.value)

    def record(self) -> dict[str, object]:
        return dict(knob="shortlist", value=self.value, **_NO_DISCOUNT)


@dataclass(frozen=True, kw_only=True)
class Chunks:
    """resistance_lp's `chunks` (8 in the preset): its Frank-Wolfe instalments."""

    value: int

    @property
    def label(self) -> str:
        return f"chunks{self.value}"

    def apply(self, method: Method, block_id: str) -> Method:
        if not isinstance(method, ResistanceLPReblocker):
            raise TypeError(f"chunks is resistance_lp's, not {type(method).__name__}'s")
        return dataclasses.replace(method, chunks=self.value)

    def record(self) -> dict[str, object]:
        return dict(knob="chunks", value=self.value, **_NO_DISCOUNT)


@dataclass(frozen=True, kw_only=True)
class TreeRepulsion:
    """clearance_looped's tree's `repulsion` (0 in the preset), the logit of its building term."""

    value: float

    @property
    def label(self) -> str:
        return f"repulsion{self.value:g}"

    def apply(self, method: Method, block_id: str) -> Method:
        if not (isinstance(method, LoopClosureRefiner)
                and isinstance(method.base, ClearanceReblocker)):
            raise TypeError(f"repulsion is clearance_looped's tree's, not {method!r}'s")
        return dataclasses.replace(method,
                                   base=dataclasses.replace(method.base, repulsion=self.value))

    def record(self) -> dict[str, object]:
        return dict(knob="base.repulsion", value=self.value, **_NO_DISCOUNT)


Priced = CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker


@dataclass(frozen=True, kw_only=True)
class Discount:
    """`sets` at factor `f`, in the price (the method's `price`), the route (TransparentSubstrate
    over its substrate) or both; on clearance_looped, its tree's (the loop stage has no building
    term)."""

    sets: SetChoice
    f: float
    price: bool
    route: bool

    def __post_init__(self) -> None:
        if not (self.price or self.route):
            raise ValueError("a discount enters the price, the route or both")
        if not 0.0 < self.f < 1.0:
            raise ValueError(f"a discount's factor lies in (0, 1); got {self.f}")

    @property
    def label(self) -> str:
        where = "-".join(w for w, on in (("price", self.price), ("route", self.route)) if on)
        return f"{self.sets.label}-f{self.f:g}-{where}"

    def _on(self, m: Priced, t: Transparency) -> Priced:
        if self.price:
            m = dataclasses.replace(m, price=t)
        if self.route:
            m = dataclasses.replace(m, substrate=TransparentSubstrate(base=m.substrate,
                                                                      transparency=t))
        return m

    def transparency(self, block_id: str) -> Transparency:
        return self.sets.transparency(block_id, self.f)

    def apply(self, method: Method, block_id: str) -> Method:
        t = self.transparency(block_id)
        if isinstance(method, LoopClosureRefiner):
            if not isinstance(method.base, ClearanceReblocker):
                raise TypeError(f"clearance_looped's tree is {type(method.base).__name__}")
            return dataclasses.replace(method, base=self._on(method.base, t))
        if isinstance(method, CycleNativeReblocker | ResistanceLPReblocker):
            return self._on(method, t)
        raise TypeError(f"{type(method).__name__} has no price")

    def record(self) -> dict[str, object]:
        return dict(knob="", value=np.nan, **self.sets.record(), f=self.f, price=self.price,
                    route=self.route)


# ------------------------------------------------------------------ the arms
@dataclass(frozen=True, kw_only=True)
class Arm:
    """`change` on the preset `preset` of compare_config's all_methods, in method `group`'s part
    of the experiment. `reuse_item4`: the network is item 4's of the same preset (no proposal)."""

    group: str
    preset: str
    change: Change
    reuse_item4: bool

    def __post_init__(self) -> None:
        if self.reuse_item4 and not isinstance(self.change, AsConfigured):
            raise ValueError("only a preset as configured can reuse item 4's network")

    @property
    def name(self) -> str:
        return f"{self.group}+{self.change.label}"

    def method(self, presets: Mapping[str, Method], block_id: str) -> Method:
        return self.change.apply(presets[self.preset], block_id)


@dataclass(frozen=True, kw_only=True)
class MethodArms:
    """One method's arms by their part in the screen."""

    method: str
    base: Arm
    knobs: tuple[Arm, ...]          # its own knobs, opaque
    main: tuple[Arm, ...]           # C in the main form, one per F_LEVELS
    ablations: tuple[Arm, ...]      # C at F_ONE, the main form's parts apart or together
    cheap: Arm                      # C' in the main form at F_ONE
    random: tuple[Arm, ...]         # matched to C, the main form at F_ONE, one per seed

    @property
    def arms(self) -> tuple[Arm, ...]:
        return (self.base, *self.knobs, *self.main, *self.ablations, self.cheap, *self.random)

    def parts(self) -> list[tuple[str, Arm]]:
        return ([("base", self.base)] + [("knob", a) for a in self.knobs]
                + [("main", a) for a in self.main] + [("ablation", a) for a in self.ablations]
                + [("cheap", self.cheap)] + [("random", a) for a in self.random])

    def main_at(self, f: float) -> Arm:
        [arm] = [a for a in self.main if isinstance(a.change, Discount) and a.change.f == f]
        return arm


def _method_arms(method: str, *, knobs: tuple[Arm, ...], price: bool, route: bool,
                 ablations: tuple[tuple[bool, bool], ...]) -> MethodArms:
    """`method`'s arms, the main form discounting in the price (`price`) and the route (`route`),
    each ablation a (price, route) pair at F_ONE."""
    def arm(change: Change) -> Arm:
        return Arm(group=method, preset=method, change=change, reuse_item4=False)

    def disc(sets: SetChoice, f: float, p: bool, r: bool) -> Arm:
        return arm(Discount(sets=sets, f=f, price=p, route=r))

    return MethodArms(
        method=method, base=arm(AsConfigured(label="base")), knobs=knobs,
        main=tuple(disc(C, f, price, route) for f in F_LEVELS),
        ablations=tuple(disc(C, F_ONE, p, r) for p, r in ablations),
        cheap=disc(C_CHEAP, F_ONE, price, route),
        random=tuple(disc(MatchedRandom(rows=ts.SIMP_DEFAULT, stratification=STRATIFICATION,
                                        seed=s), F_ONE, price, route) for s in SEEDS))


def _knob(method: str, change: Change) -> Arm:
    return Arm(group=method, preset=method, change=change, reuse_item4=False)


PLAN = (
    # cycle_native: the main form prices and routes; its knobs the shortlist and the
    # desire-routed preset (item 4's network reused)
    _method_arms("cycle_native", knobs=(
        _knob("cycle_native", Shortlist(value=4)), _knob("cycle_native", Shortlist(value=16)),
        Arm(group="cycle_native", preset="cycle_native_betweenness_contrast",
            change=AsConfigured(label="desire"), reuse_item4=True)),
        price=True, route=True, ablations=((True, False), (False, True))),
    # resistance_lp: the main form prices alone (the pilot: its route lowers it)
    _method_arms("resistance_lp", knobs=(
        _knob("resistance_lp", Chunks(value=4)), _knob("resistance_lp", Chunks(value=16))),
        price=True, route=False, ablations=((False, True), (True, True))),
    # clearance_looped: prices and routes its tree
    _method_arms("clearance_looped", knobs=tuple(
        _knob("clearance_looped", TreeRepulsion(value=v)) for v in (-2.0, 2.0, 4.0)),
        price=True, route=True, ablations=((True, False), (False, True))),
)
ARMS = tuple(a for m in PLAN for a in m.arms)
LINEUP = tuple(a.name for a in ARMS)
if len(set(LINEUP)) != len(LINEUP):
    raise ValueError(f"arm names collide: {LINEUP}")


def arm_of(name: str) -> Arm:
    """The arm a command line names (an unknown name raises)."""
    [arm] = [a for a in ARMS if a.name == name]
    return arm


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in rr.SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {rr.SINGLE}")


# ------------------------------------------------------------------ arms, dry
def arms_table() -> pd.DataFrame:
    rows = []
    for m in PLAN:
        for part, arm in m.parts():
            rows.append(dict(arm=arm.name, method=m.method, part=part, preset=arm.preset,
                             reuse_item4=arm.reuse_item4, **arm.change.record()))
    return pd.DataFrame(rows)


def write_arms() -> None:
    df = arms_table()
    ROWS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROWS / "arms.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    reused = int(df.reuse_item4.sum())
    print(f"{len(df)} arms ({reused} reused), {len(df) - reused} to propose per block; written to "
          f"{ROWS / 'arms.csv'}")


def dry(bid: str) -> None:
    """Every arm's method on one block, built and not run: its label and identity, the base arms'
    against the presets', every identity distinct, each discount's set and factors."""
    presets = common.arms()
    [b] = common.build_blocks([bid])
    seen: dict[object, str] = {}
    for arm in ARMS:
        m = arm.method(presets, bid)
        ident = m.identity
        if ident is None:
            raise AssertionError(f"{arm.name}: an uncacheable identity")
        if ident in seen:
            raise AssertionError(f"{arm.name} and {seen[ident]} share an identity")
        seen[ident] = arm.name
        if isinstance(arm.change, AsConfigured) and ident != presets[arm.preset].identity:
            raise AssertionError(f"{arm.name}: not the preset {arm.preset}")
        line = f"{arm.name:45s} {type(m).__name__}"
        if isinstance(arm.change, Discount):
            t = arm.change.transparency(bid)
            f = factors_of(t, b)
            line += (f"  {int((f < 1).sum())} buildings at f {arm.change.f:g}, tag {t.tag},"
                     f" price {arm.change.price}, route {arm.change.route}")
        print(line, flush=True)
    print(f"{len(ARMS)} arms on {bid}: identities distinct, base arms the presets")


# ------------------------------------------------------------------ propose
def _record_path(bid: str, name: str) -> Path:
    """The network of record: the fresh run's roads (road_rescore.propose writes them beside it as
    <name>.fresh.parquet) or, for a reused arm, item 4's."""
    return rr.net_path(SCRATCH, bid, name, False)


def propose_one(bid: str, name: str) -> None:
    """One (block, arm) network, fresh (road_rescore.propose); the roads become the network of
    record."""
    _check_single_threaded()
    arm = arm_of(name)
    if arm.reuse_item4:
        raise SystemExit(f"{name} reuses item 4's network: nothing to propose")
    method = arm.method(common.arms(), bid)
    rr.propose(ROWS, SCRATCH, bid, name, method, True)
    row = pd.read_parquet(rr.proposal_row(ROWS, bid, name, True)).iloc[0]
    if row.status == "ok":
        os.replace(rr.net_path(SCRATCH, bid, name, True), _record_path(bid, name))


def propose(procs: int, timeout_s: float, ids: list[str]) -> None:
    """Every (block, arm) network without a proposal row, `procs` children at a time, the
    costliest first; a child past `timeout_s` of wall time is killed by PID and recorded. A reused
    arm's network is copied from item 4's."""
    _check_single_threaded()
    for bid in ids:
        for arm in ARMS:
            dest = _record_path(bid, arm.name)
            if arm.reuse_item4 and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ITEM4_NETS / bid / f"{arm.preset}.parquet", dest)
                print(f"{bid} {arm.name}: item 4's {arm.preset} network", flush=True)
    sizes = pd.read_parquet(HERE / "large82_area.parquet").set_index("block")["n"]
    queue = sorted(((bid, arm) for bid in ids for arm in ARMS if not arm.reuse_item4
                    and not rr.proposal_row(ROWS, bid, arm.name, True).exists()),
                   key=lambda j: -float(sizes[j[0]]) * COST_GUESS[j[1].group])
    env = {**os.environ, **rr.SINGLE,
           "PYTHONPATH": os.pathsep.join([str(common.REPO), str(HERE)])}
    (SCRATCH / "logs").mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple] = {}              # pid -> (Popen, bid, arm, start, log handle)
    print(f"{time.strftime('%H:%M:%S')} {len(queue)} jobs, {procs} at a time", flush=True)
    while queue or running:
        while queue and len(running) < procs:
            bid, arm = queue.pop(0)
            fh = open(SCRATCH / "logs" / f"propose_{bid}__{arm.name}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, "propose-one", bid, arm.name],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=str(common.REPO))
            running[p.pid] = (p, bid, arm, time.perf_counter(), fh)
            print(f"{time.strftime('%H:%M:%S')} start pid {p.pid} {bid} {arm.name}", flush=True)
        time.sleep(5.0)
        for pid, (p, bid, arm, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is None and time.perf_counter() - t0 > timeout_s:
                p.kill()                        # by PID, never a pattern
                p.wait()
                prog = SCRATCH / "progress" / f"{bid}__{arm.name}__fresh.json"
                known = json.loads(prog.read_text()) if prog.exists() else {}
                dest = rr.proposal_row(ROWS, bid, arm.name, True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([dict(block=bid, n=int(sizes[bid]), method=arm.name, fresh=True,
                                   status=f"timeout after {timeout_s:.0f} s wall",
                                   wall_s=known.get("wall_s", np.nan),
                                   cpu_s=known.get("cpu_s", np.nan),
                                   peak_rss_gb=known.get("peak_rss_gb", np.nan))]).to_parquet(dest)
                rc = "killed"
            if rc is None:
                continue
            fh.close()
            del running[pid]
            print(f"{time.strftime('%H:%M:%S')} done pid {pid} {bid} {arm.name}: rc {rc}, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)


def _wkb(roads) -> list[bytes]:
    import shapely
    return [bytes(w) for w in shapely.to_wkb(np.asarray(roads.geometry))]


def check_base(ids: list[str]) -> None:
    """Every opaque preset arm (the bases and the reused desire-routed network) against item 4's
    network of its preset: the same WKB row for row and the same widths. Raises on any
    difference: the comparison rests on it."""
    import geopandas as gpd
    out = []
    for bid in ids:
        for arm in ARMS:
            if not isinstance(arm.change, AsConfigured):
                continue
            mine = gpd.read_parquet(_record_path(bid, arm.name))
            theirs = gpd.read_parquet(ITEM4_NETS / bid / f"{arm.preset}.parquet")
            same = bool(len(mine) == len(theirs) and _wkb(mine) == _wkb(theirs)
                        and np.array_equal(mine["width_m"].to_numpy(),
                                           theirs["width_m"].to_numpy()))
            out.append(dict(block=bid, arm=arm.name, reused=arm.reuse_item4, roads=len(mine),
                            item4_roads=len(theirs), same=same))
            print(f"{bid} {arm.name:30s} {len(mine):5d} roads, item 4 {len(theirs):5d}: "
                  f"{'identical' if same else 'DIFFERENT'}", flush=True)
    df = pd.DataFrame(out)
    if not df.same.all():
        raise SystemExit(f"{int((~df.same).sum())} base networks differ from item 4's")
    print(f"all {len(df)} identical to item 4's networks")


# ------------------------------------------------------------------ prefixes, tars
def prefixes(procs: int, ids: list[str]) -> None:
    rr.prefixes(ROWS, LINEUP, SCRATCH, procs, ids)


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(common.REPO))


def tars(out: Path, ids: list[str]) -> None:
    """Per block, a tar of what road_rescore.score reads that git does not carry, at its repo
    paths: every arm's prefix rows and ordered network, and the stored clearings it rescores."""
    out.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        members = [rr.prefix_dir(ROWS) / f"{bid}__{name}.parquet" for name in LINEUP]
        members += [rr.ordered_path(SCRATCH, bid, name) for name in LINEUP]
        members += [p for rel_dir, *_ in rr.STORED.values()
                    if (p := HERE / rel_dir / f"{bid}_p{rr.CLEARING_POWER:g}.parquet").exists()]
        absent = [str(m) for m in members if not m.exists()]
        if absent:
            raise SystemExit(f"{bid}: {len(absent)} inputs absent, e.g. {absent[0]}")
        with tarfile.open(out / f"{bid}.tar", "w") as tf:
            for m in members:
                tf.add(m, arcname=_rel(m))
        print(f"{bid}: {len(members)} files, {(out / f'{bid}.tar').stat().st_size / 2**20:.1f} MB",
              flush=True)
    print("\nscore (lenses (b), (c)) on the cluster, one task a block:\n"
          f"  uv run python research/roadless/cluster.py submit blocks <run> --time 3:00:00 "
          f"--unpack {out}/{{id}}.tar {','.join(ids)} -- research/roadless/road_rescore.py score "
          f"{_rel(ROWS)} {','.join(LINEUP)} {_rel(SCRATCH)} {MESH} count,area {{id}}")


# ------------------------------------------------------------------ records
_BLOCKS: dict = {}       # the jobs' blocks, built once before the pool forks


def _prefix_rows(bid: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(rr.prefix_dir(ROWS) / f"{bid}__{name}.parquet")


def _record_one(job: tuple[str, str]) -> dict[str, object]:
    """Per (block, arm): the whole network's true displacement under both weightings, the share
    of the true (area-weighted) displacement of its (c) prefix at LENS_D that falls on C, on C'
    and on the arm's own set, and a random set's draw."""
    import geopandas as gpd

    from reblock.budget import road_corridor
    bid, name = job
    arm, b = arm_of(name), _BLOCKS[bid]
    pre = _prefix_rows(bid, name)
    full = pre[pre.full.astype(bool) & (pre["pop"] == "area")].iloc[0]
    m_c = int(pre[(pre["pop"] == "area") & np.isclose(pre.budget, LENS_D)].m.iloc[0])
    ordered = gpd.read_parquet(rr.ordered_path(SCRATCH, bid, name))
    polys = np.asarray(b.buildings.outlines)
    w = common.POPULATIONS["area"].weights(polys)
    cw = b.buildings.displacement(road_corridor(ordered.iloc[:m_c])) * w
    on = {"simp": ts.Clearing.read(ts.SIMP_DEFAULT, bid).positions(b),
          "cheap": ts.Clearing.read(ts.CHEAP, bid).positions(b)}
    d_c = float(cw.sum() / w.sum())
    row: dict[str, object] = dict(
        block=bid, arm=name, n=len(polys), roads=int(full.n_roads), road_km=full.road_m / 1000,
        D_count_whole=float(full.D_count), D_area_whole=float(full.D_area),
        m_c=m_c, D_area_c=d_c, D_area_c_row=float(
            pre[(pre["pop"] == "area") & np.isclose(pre.budget, LENS_D)].D_area.iloc[0]),
        share_simp=float(cw[on["simp"]].sum() / cw.sum()),
        share_cheap=float(cw[on["cheap"]].sum() / cw.sum()),
        set_size=np.nan, share_own=np.nan, area_ratio=np.nan, jaccard=np.nan, tag="")
    if isinstance(arm.change, Discount):
        t = arm.change.transparency(bid)
        own = np.flatnonzero(factors_of(t, b) < 1.0)
        row.update(set_size=len(own), share_own=float(cw[own].sum() / cw.sum()), tag=t.tag)
        if isinstance(t, ts.MatchedRandomSet):
            d = t.draw(b)
            if not np.array_equal(d.members, own):
                raise AssertionError(f"{bid} {name}: the draw is not the discounted set")
            row.update(area_ratio=d.area_ratio, jaccard=d.jaccard)
    return row


def _cost(bid: str, arm: Arm) -> dict[str, object]:
    """What proposing the arm's network cost: its fresh run, or item 4's for a reused one."""
    if arm.reuse_item4:
        pt = rr.proposals_table(ITEM4_ROWS)
        r = pt[(pt.block == bid) & (pt.method == arm.preset)].iloc[0]
        return dict(status=f"item 4: {r.cost_status}", wall_s=r.cost_wall_s, cpu_s=r.cost_cpu_s,
                    peak_rss_gb=r.cost_peak_rss_gb)
    r = pd.read_parquet(rr.proposal_row(ROWS, bid, arm.name, True)).iloc[0]
    return dict(status=r.status, wall_s=r.wall_s, cpu_s=r.cpu_s, peak_rss_gb=r.peak_rss_gb)


def records(procs: int, ids: list[str]) -> None:
    _check_single_threaded()
    import multiprocessing
    _BLOCKS.update({b.block_id: b for b in common.build_blocks(ids)})
    jobs = [(bid, name) for bid in ids for name in LINEUP]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=8) as pool:
        got = list(pool.imap(_record_one, jobs))
    df = pd.DataFrame(got)
    cost = pd.DataFrame([dict(block=bid, arm=arm.name, **_cost(bid, arm))
                         for bid in ids for arm in ARMS])
    df = df.merge(cost, on=["block", "arm"])
    off = (df.D_area_c - df.D_area_c_row).abs().max()
    if off > 1e-12:
        raise AssertionError(f"the (c) prefix's displacement differs from its row by {off:.1e}")
    df.to_csv(ROWS / "summary_records.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df.drop(columns=["tag", "D_area_c_row"]).to_string(index=False))


# ------------------------------------------------------------------ report: the screen
def _per_block(t: pd.DataFrame, name: str, col: str) -> pd.Series:
    s = t[t.method == name].set_index("block")[col]
    return s.reindex(list(SENTINEL))


def _short(bid: str) -> str:
    return bid.rsplit("_", 1)[-1]


@dataclass(frozen=True, kw_only=True)
class Reference:
    """One method's reference points on (c) at LENS_D, per block: its base, the gap from base to
    SIMP's default, its random sets' mean and sd, and its best own-knob point (of base and the
    knob arms, by mean (c))."""

    base: pd.Series
    gap: pd.Series
    rand_mean: pd.Series
    rand_sd: pd.Series
    best_knob: Arm
    best_mean: float

    @classmethod
    def of(cls, m: MethodArms, t5: pd.DataFrame, simp: pd.Series) -> Reference:
        base = _per_block(t5, m.base.name, "c")
        rand = pd.DataFrame({a.name: _per_block(t5, a.name, "c") for a in m.random})
        best = max((m.base, *m.knobs), key=lambda a: float(_per_block(t5, a.name, "c").mean()))
        return cls(base=base, gap=simp - base, rand_mean=rand.mean(axis=1),
                   rand_sd=rand.std(axis=1, ddof=1), best_knob=best,
                   best_mean=float(_per_block(t5, best.name, "c").mean()))


def _stats(lens: Mapping[float, pd.DataFrame], m: MethodArms, ref: Reference, arm: Arm,
           part: str) -> dict[str, object]:
    """An arm's numbers in the screen (NOTES, "Transparency for the road methods")."""
    t5 = lens[LENS_D]
    c = _per_block(t5, arm.name, "c")
    gain = c - ref.base
    a_loss = {d: float((_per_block(lens[d], m.base.name, "a")
                        - _per_block(lens[d], arm.name, "a")).median()) for d in GUARD_D}
    r: dict[str, object] = dict(
        method=m.method, arm=arm.name, part=part,
        **{f"c_{_short(b)}": float(c[b]) for b in SENTINEL},
        mean_c=float(c.mean()), mean_gain=float(gain.mean()), min_gain=float(gain.min()),
        veto=bool((gain < -VETO).any()), above_random=int((c > ref.rand_mean).sum()),
        **{f"a_loss_D{d:g}": a_loss[d] for d in GUARD_D},
        a_ok=all(v <= A_LOSS for v in a_loss.values()),
        **{f"gap_{_short(b)}": float(gain[b] / ref.gap[b]) for b in SENTINEL},
        mean_gap_closed=float((gain / ref.gap).mean()),
        vs_best_knob=float(c.mean()) - ref.best_mean,
        reach_c=bool(_per_block(t5, arm.name, "reach_area").all()),
        reach_a=bool(all(_per_block(lens[d], arm.name, "reach_count").all() for d in GUARD_D)))
    r["passes"] = bool(not r["veto"] and float(gain.mean()) > 0 and int(r["above_random"]) >= 3
                       and r["a_ok"])
    return r


def screen(lens: Mapping[float, pd.DataFrame], m: MethodArms,
           ref: Reference) -> tuple[Arm, dict[str, dict[str, object]]]:
    """The screen on one method: the main form (its f with the larger mean gain) and every arm's
    numbers, `advances` among them."""
    st = {arm.name: _stats(lens, m, ref, arm, part) for part, arm in m.parts()}

    def gain(a: Arm) -> float:
        return float(st[a.name]["mean_gain"])

    main = max(m.main, key=gain)
    main_go = bool(st[main.name]["passes"])
    for part, arm in m.parts():
        r = st[arm.name]
        if part == "main":
            adv = arm is main and main_go
        elif part == "ablation":
            adv = bool(r["passes"]) and gain(arm) > gain(main)
        elif part == "cheap":
            adv = main_go and not r["veto"] and gain(arm) >= gain(m.main_at(F_ONE)) - CHEAP_WITHIN
        elif part == "random":
            adv = main_go                       # the advancing main form's control
        else:
            adv = False
        r.update(advances=adv, chosen_main=arm is main, best_knob=arm is ref.best_knob)
    return main, st


def report() -> None:
    """The screen as fixed in NOTES before the runs, and the tables behind it; writes
    transparency_rows/summary_lens.csv (per block and arm: every lens) and summary_screen.csv
    (per arm: the screen's numbers)."""
    pd.set_option("display.width", 250)
    lens = {d: rr.lens_table(ROWS, LINEUP, MESH, d) for d in GUARD_D}
    t5 = lens[LENS_D]
    missing = sorted(set(LINEUP) - set(t5.method))
    if missing or t5.groupby("method").size().ne(len(SENTINEL)).any():
        raise SystemExit(f"not every arm is scored on every block: missing {missing}")
    if any((t.enc_c > 0).any() or (t.enc_b > 0).any() for t in lens.values()):
        raise SystemExit("an arm leaves population enclosed: rank in B's order, not by (c) alone")
    rec = pd.read_csv(ROWS / "summary_records.csv")
    # item 4: SIMP's default on the same scorer and mesh (its rows), and the base networks
    item4 = {d: pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, f"lens_D{d:g}")) for d in GUARD_D}
    cl4 = pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, "clearings"))
    simp = (cl4[(cl4.mesh == "h0.5") & (cl4.clearing == "simp_default")].set_index("block")
            .perm.reindex(list(SENTINEL)))
    mine = rr._read_all(sorted(rr.score_dir(ROWS, MESH, "area").glob("*__clearings.parquet")))
    both = mine.merge(cl4[["block", "mesh", "clearing", "perm"]],
                      on=["block", "mesh", "clearing"], suffixes=("", "_item4"))
    print(f"SCORER CHECK: the stored clearings rescored here against item 4's rows, "
          f"{len(both)} rows ({', '.join(sorted(set(both.clearing)))}): max |diff| "
          f"{(both.perm - both.perm_item4).abs().max():.1e}")
    for m in PLAN:
        for arm in (m.base, *[a for a in m.knobs if a.reuse_item4]):
            for d in GUARD_D:
                ref4 = item4[d][item4[d].method == arm.preset].set_index("block")
                diffs = [float((_per_block(lens[d], arm.name, col)
                                - ref4[col].reindex(list(SENTINEL))).abs().max())
                         for col in ("a", "b", "c")]
                print(f"  {arm.name} at D {d:g} against item 4's {arm.preset}: max |diff| (a) "
                      f"{diffs[0]:.1e}, (b) {diffs[1]:.1e}, (c) {diffs[2]:.1e}")
    screen_rows = []
    for m in PLAN:
        ref = Reference.of(m, t5, simp)
        main, st = screen(lens, m, ref)
        screen_rows += list(st.values())
        short = {a.name: a.change.label for a in m.arms}
        print(f"\n{m.method.upper()}: (c) at area D {LENS_D:g} per block, its mean gain over "
              f"base, the gap to SIMP's default closed (mean over blocks), the (a) loss against "
              f"base (median over blocks) at count D 0.05 | 0.10, the blocks above the random "
              f"sets' mean; * vetoed, > advances")
        hdr = "".join(f"{_short(b):>9s}" for b in SENTINEL)
        print(f"  {'arm':28s}{hdr}     gain     gap  a-loss .05 |   .10  >rand")
        for _part, arm in m.parts():
            r = st[arm.name]
            mark = ("*" if r["veto"] else "") + (">" if r["advances"] else "")
            cells = "".join(f"{r['c_' + _short(b)]:9.4f}" for b in SENTINEL)
            print(f"  {mark:2s}{short[arm.name]:26s}{cells} {r['mean_gain']:+8.4f} "
                  f"{r['mean_gap_closed']:+7.3f}  {r['a_loss_D0.05']:+7.4f} | "
                  f"{r['a_loss_D0.1']:+7.4f}  {r['above_random']}")
        print(f"  {'random mean':28s}" + "".join(f"{v:9.4f}" for v in ref.rand_mean)
              + "   sd " + " ".join(f"{v:.4f}" for v in ref.rand_sd))
        print(f"  {'SIMP default':28s}" + "".join(f"{v:9.4f}" for v in simp))
        go = [short[n] for n, r in st.items() if r["advances"]]
        print(f"  main form: {short[main.name]} (mean gain {st[main.name]['mean_gain']:+.4f}); "
              f"advances: {', '.join(go) or 'none'}; best own-knob point "
              f"{short[ref.best_knob.name]} (mean (c) {ref.best_mean:.4f})")
    pd.DataFrame(screen_rows).to_csv(ROWS / "summary_screen.csv", index=False)
    keep = ["block", "method", "n", "m_count", "a", "b", "enc_b", "m_area", "c", "enc_c",
            "Dc_count", "Da_area", "road_m_count", "road_m_area", "reach_count", "reach_area"]
    out = pd.concat([lens[d][keep].assign(budget=d) for d in GUARD_D], ignore_index=True)
    out.rename(columns={"method": "arm"}).merge(
        rec[["block", "arm", "share_simp", "share_own", "area_ratio", "jaccard"]],
        on=["block", "arm"]).to_csv(ROWS / "summary_lens.csv", index=False)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "arms":
        write_arms()
    elif cmd == "dry":
        dry(a[1])
    elif cmd == "propose":
        propose(int(a[1]), float(a[2]) * 3600, a[3].split(","))
    elif cmd == "propose-one":
        try:
            propose_one(a[1], a[2])
        except Exception:
            traceback.print_exc()
            raise
    elif cmd == "check-base":
        check_base(a[1].split(","))
    elif cmd == "prefixes":
        prefixes(int(a[1]), a[2].split(","))
    elif cmd == "tars":
        tars(Path(a[1]), a[2].split(","))
    elif cmd == "records":
        records(int(a[1]), a[2].split(","))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
