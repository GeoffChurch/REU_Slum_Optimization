"""The road-transparency experiment (BACKLOG 5; the owner approved it 2026-10-12, "Leave out
arterial"): do cycle_native, resistance_lp and clearance_looped build better networks at equal
displacement when the buildings the roadless optimum clears are made cheaper to displace (the
PRICE) and to route through (the ROUTE; reblock.transparency)?

An arm is a named, block-independent configuration: a preset of compare_config's all_methods
(common.arms(), every building opaque) and what the arm changes on it -- nothing (the base), one
of the method's own knobs, or a Discount: a set of buildings at a factor f, in the price, the
route or both. The sets: C, the buildings SIMP's default clears at D 0.05
(transparency_sets.SIMP_DEFAULT); C', the cheap preset's (transparency_sets.CHEAP); and the
control, random sets matched to C by footprint area in 20 strata of equal area mass
(transparency_sets.AreaMass), seeds 1 -- 5. An arm's name keys its network and its rows.

A Stage is one step of the experiment: its blocks, its arms, the rows root they go under, what
its scoring solves, and its Verdict, the rule fixed in NOTES before its results:
  sentinel  the 4-block sentinel, 41 arms (PLAN), every lens scored; the Screen (transparency_rows/)
  tuning    the 13 tuning blocks (simp_compare.TUNING), 37 arms (TUNING_PLAN), (c) scored at area
            D 0.05 and 0.10 only, (a) from the prefixes; the Choice (transparency_rows/tuning/)
  held      the 46 held-out blocks (HELD), scored as the 13; its arms are data, the committed
            transparency_rows/held/plan.csv (held_plan: per method its claim and its own knobs,
            each an arm of the 13, base and the claim's random sets with them); the Claim
            (transparency_rows/held/)
`arms` writes a stage's arms, with every value that can change a network, to <rows>/arms.csv
(committed). The networks are proposed fresh (road_rescore.propose, no derive cache anywhere),
cut to their canonical prefixes by road_rescore.prefixes and scored by road_rescore.score, under
the stage's rows root (item 4's road_rows/ is read, never written):

    S=research/roadless/transparency_run.py    # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S <stage> arms                     # write <rows>/arms.csv
    uv run python $S <stage> dry <id>                 # every arm's method on one block, unrun
    uv run python $S <stage> propose <procs> <timeout h> <ids,>    # here, `procs` at a time
    uv run python $S <stage> arm <id> <arm>           # one cluster task: network, then prefixes
    uv run python $S <stage> recompute <id> <arm>     # a reused arm's preset, fresh (a check)
    uv run python $S <stage> cpu-tars <dir> <ids,>    # a tar per block for the arm tasks
    uv run python $S <stage> pulled <ids,> <arms,|all>        # their networks and rows came back
    uv run python $S <stage> check-base <ids,> <arms,|all>    # opaque arms against item 4's
    uv run python $S <stage> same-as <stage'> <ids,>  # the arms both have, network for network
    uv run python $S <stage> same-lens <stage'> <ids,>    # ... and where the same, their lenses
    uv run python $S <stage> prefixes <procs> <ids,>
    uv run python $S <stage> tars <dir> <ids,>        # a tar per block for score on the cluster
    uv run python $S <stage> records <procs> <ids,>   # displacement on C, the draws, the costs
    uv run python $S <stage> report                   # the stage's verdict, and the summaries

`propose` runs `propose-one <id> <arm>` children here; on the cluster's CPU nodes `arm` is one
task of cluster.py's `cpu` kind (`cpu-tars` prints its command). An arm marked `reuse_item4` (the
desire-routed cycle_native) takes item 4's network of its preset instead (its code path is
unchanged under Opaque: NOTES); `recompute` proposes that preset afresh beside it, for
`check-base`. `tars` prints the cluster command that scores the stage's lenses.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import road_rescore as rr  # noqa: E402
import simp_compare  # noqa: E402
import transparency_sets as ts  # noqa: E402

from reblock.contracts import Method  # noqa: E402
from reblock.methods.clearance import ClearanceReblocker  # noqa: E402
from reblock.methods.cycle_native import CycleNativeReblocker  # noqa: E402
from reblock.methods.loop_closure import LoopClosureRefiner  # noqa: E402
from reblock.methods.resistance_lp import ResistanceLPReblocker  # noqa: E402
from reblock.transparency import Transparency, TransparentSubstrate, factors_of  # noqa: E402

# ------------------------------------------------------------------ the experiment's configuration
ITEM4_ROWS = HERE / "road_rows"              # item 4's rows and networks, read only
ITEM4_NETS = ITEM4_ROWS / "scratch" / "networks"
SENTINEL = ("ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712")
# The held-out blocks: item 4's 60 large blocks that fit at h 0.5 (the blocks of its committed lens
# rows, road_rows/summary_*_lens_D0.05.csv; road_rescore.py `project`'s gb_uniform under 48 GB)
# less the 13 tuning blocks (simp_compare.TUNING) and 14401, which has no stored clearing at h 0.5
# (neither SIMP's default nor the cheap preset's). transparency_check.py `held` derives them again.
HELD = ("ZAF.9.1.4_1_6312", "ZAF.9.3.1_1_17608", "ZAF.9.3.1_1_18739", "ZAF.9.3.1_1_18910",
        "ZAF.9.3.1_1_18947", "ZAF.9.3.1_1_18985", "ZAF.9.3.1_1_19161", "ZAF.9.3.1_1_19810",
        "ZAF.9.3.1_1_20023", "ZAF.9.3.1_1_20080", "ZAF.9.3.1_1_20423", "ZAF.9.3.1_1_20952",
        "ZAF.9.3.1_1_21252", "ZAF.9.3.1_1_21777", "ZAF.9.3.1_1_22267", "ZAF.9.3.1_1_22336",
        "ZAF.9.3.1_1_22591", "ZAF.9.3.1_1_22633", "ZAF.9.3.1_1_22640", "ZAF.9.3.1_1_22659",
        "ZAF.9.3.1_1_22673", "ZAF.9.3.1_1_23735", "ZAF.9.3.1_1_24240", "ZAF.9.3.1_1_30796",
        "ZAF.9.3.1_1_38254", "ZAF.9.3.1_1_38366", "ZAF.9.3.1_1_38528", "ZAF.9.3.1_1_38988",
        "ZAF.9.3.1_1_39240", "ZAF.9.3.1_1_40144", "ZAF.9.3.1_1_41515", "ZAF.9.3.1_1_41673",
        "ZAF.9.3.1_1_41806", "ZAF.9.3.1_1_43547", "ZAF.9.3.1_1_43708", "ZAF.9.3.1_1_44602",
        "ZAF.9.3.1_1_44775", "ZAF.9.3.1_1_45876", "ZAF.9.3.1_1_5706", "ZAF.9.3.1_1_6310",
        "ZAF.9.3.1_1_6316", "ZAF.9.3.1_1_7662", "ZAF.9.3.1_1_8152", "ZAF.9.3.1_1_8480",
        "ZAF.9.5.4_1_8235", "ZAF.9.5.4_1_9710")
HELD_ROWS = HERE / "transparency_rows" / "held"
HELD_PLAN = HELD_ROWS / "plan.csv"           # the held-out stage's arms, as data (committed)
MESH = "0.5"                                 # item 4's: h 0.5, uni (road_rescore's parameters)
F_LEVELS = (0.5, 0.1)                        # the sentinel's main form's factors
F_ONE = 0.1                                  # the ablations', C''s and the random sets' factor;
                                             # every discount's on the 13
STRATIFICATION = ts.AreaMass(strata=20)      # the control's (owner's ruling 2026-10-12)
SEEDS = (1, 2, 3, 4, 5)
LENS_D = 0.05                                # (c) at area D 0.05: the primary number
GUARD_D = (0.05, 0.10)                       # (a) at count D 0.05 and 0.10: the guard
VETO = 0.1                                   # an arm more than this below base on (c) is dropped
A_LOSS = 0.01                                # the guard's largest median (a) loss
CHEAP_WITHIN = 0.01                          # the sentinel: C' advances within this of C's gain
BOOT_DRAWS = 10_000                          # the 13: the Bayesian bootstrap's draws ...
BOOT_SEED = 20261012                         # ... and its seed (one weight matrix for every arm)
ENCLOSED_TOL = 1e-12                         # B (i) shares equal within this (sums of weights)
# propose's order only (no result depends on it): a guess at each method's cost per building
COST_GUESS = {"cycle_native": 2.5, "resistance_lp": 2.0, "clearance_looped": 1.0}


# ------------------------------------------------------------------ the sets
class SetChoice(Protocol):
    """Which buildings a discount makes cheaper, read for one block where the arm is built."""

    @property
    def label(self) -> str: ...

    @property
    def rows(self) -> ts.StoredRows: ...     # the stored clearing it is read from

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

    @property
    def form(self) -> tuple[bool, bool]:
        """Where it enters: (price, route)."""
        return (self.price, self.route)

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


def _discount(arm: Arm) -> Discount:
    if not isinstance(arm.change, Discount):
        raise TypeError(f"{arm.name} discounts nothing")
    return arm.change


@dataclass(frozen=True, kw_only=True)
class MethodArms:
    """One method's arms on the sentinel, by their part in the screen."""

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
        [arm] = [a for a in self.main if _discount(a).f == f]
        return arm


def _arm(method: str, change: Change) -> Arm:
    return Arm(group=method, preset=method, change=change, reuse_item4=False)


def _disc(method: str, sets: SetChoice, f: float, price: bool, route: bool) -> Arm:
    return _arm(method, Discount(sets=sets, f=f, price=price, route=route))


def _random(method: str, price: bool, route: bool) -> tuple[Arm, ...]:
    """The control in one form: random sets matched to C at F_ONE, one per seed."""
    return tuple(_disc(method, MatchedRandom(rows=ts.SIMP_DEFAULT, stratification=STRATIFICATION,
                                             seed=s), F_ONE, price, route) for s in SEEDS)


def _method_arms(method: str, *, knobs: tuple[Arm, ...], price: bool, route: bool,
                 ablations: tuple[tuple[bool, bool], ...]) -> MethodArms:
    """`method`'s arms on the sentinel, the main form discounting in the price (`price`) and the
    route (`route`), each ablation a (price, route) pair at F_ONE."""
    return MethodArms(
        method=method, base=_arm(method, AsConfigured(label="base")), knobs=knobs,
        main=tuple(_disc(method, C, f, price, route) for f in F_LEVELS),
        ablations=tuple(_disc(method, C, F_ONE, p, r) for p, r in ablations),
        cheap=_disc(method, C_CHEAP, F_ONE, price, route), random=_random(method, price, route))


# Each method's own knobs, opaque: cycle_native's shortlist (8 in the preset) and the
# desire-routed preset (item 4's network reused); resistance_lp's chunks (8); clearance_looped's
# tree's repulsion (0).
KNOBS = {
    "cycle_native": (_arm("cycle_native", Shortlist(value=4)),
                     _arm("cycle_native", Shortlist(value=16)),
                     Arm(group="cycle_native", preset="cycle_native_betweenness_contrast",
                         change=AsConfigured(label="desire"), reuse_item4=True)),
    "resistance_lp": (_arm("resistance_lp", Chunks(value=4)),
                      _arm("resistance_lp", Chunks(value=16))),
    "clearance_looped": tuple(_arm("clearance_looped", TreeRepulsion(value=v))
                              for v in (-2.0, 2.0, 4.0)),
}

PLAN = (
    # cycle_native: the main form prices and routes
    _method_arms("cycle_native", knobs=KNOBS["cycle_native"], price=True, route=True,
                 ablations=((True, False), (False, True))),
    # resistance_lp: the main form prices alone (the pilot: its route lowers it)
    _method_arms("resistance_lp", knobs=KNOBS["resistance_lp"], price=True, route=False,
                 ablations=((False, True), (True, True))),
    # clearance_looped: prices and routes its tree
    _method_arms("clearance_looped", knobs=KNOBS["clearance_looped"], price=True, route=True,
                 ablations=((True, False), (False, True))),
)


@dataclass(frozen=True, kw_only=True)
class ChoiceArms:
    """One method's arms on the 13 tuning blocks, by their part in the choice rule."""

    method: str
    base: Arm
    knobs: tuple[Arm, ...]          # its own knobs, opaque
    candidates: tuple[Arm, ...]     # C and C' at F_ONE: the arms the rule chooses among
    random: tuple[Arm, ...]         # matched to C at F_ONE, seeds SEEDS, in each candidate's form

    def __post_init__(self) -> None:
        for cand in self.candidates:
            self.own_random(cand)       # every candidate has its control

    @property
    def arms(self) -> tuple[Arm, ...]:
        return (self.base, *self.knobs, *self.candidates, *self.random)

    def parts(self) -> list[tuple[str, Arm]]:
        return ([("base", self.base)] + [("knob", a) for a in self.knobs]
                + [("candidate", a) for a in self.candidates]
                + [("random", a) for a in self.random])

    def own_random(self, cand: Arm) -> tuple[Arm, ...]:
        """The random sets in `cand`'s form (where its discount enters), one per seed."""
        own = tuple(r for r in self.random if _discount(r).form == _discount(cand).form)
        if len(own) != len(SEEDS):
            raise ValueError(f"{cand.name}: {len(own)} random sets in its form, not {len(SEEDS)}")
        return own


def _choice_arms(method: str, *, candidates: tuple[tuple[SetChoice, bool, bool], ...]
                 ) -> ChoiceArms:
    """`method`'s arms on the 13: its candidates, each a (set, price, route) at F_ONE, and the
    random sets in each of their forms."""
    forms = sorted({(p, r) for _, p, r in candidates})
    return ChoiceArms(
        method=method, base=_arm(method, AsConfigured(label="base")), knobs=KNOBS[method],
        candidates=tuple(_disc(method, s, F_ONE, p, r) for s, p, r in candidates),
        random=tuple(a for p, r in forms for a in _random(method, p, r)))


# The 13's arms (task brief, 2026-10-12, after the sentinel's screen): cycle_native's main form
# by the controller's ruling (it missed the sentinel's (a) guard by 0.00015); resistance_lp's
# main form (price), its price and route, and C' priced, with the control in both forms;
# clearance_looped's main form (price and route) and C'.
TUNING_PLAN = (
    _choice_arms("cycle_native", candidates=((C, True, True),)),
    _choice_arms("resistance_lp", candidates=((C, True, False), (C, True, True),
                                              (C_CHEAP, True, False))),
    _choice_arms("clearance_looped", candidates=((C, True, True), (C_CHEAP, True, True))),
)


@dataclass(frozen=True, kw_only=True)
class ClaimArms:
    """One method's arms on the held-out blocks, by their part in its claim: base, its own knobs
    (opaque), the claim (a candidate of the 13) and the claim's random sets (the five seeds in its
    form)."""

    method: str
    base: Arm
    knobs: tuple[Arm, ...]
    claim: Arm
    random: tuple[Arm, ...]

    def __post_init__(self) -> None:
        if not self.knobs:
            raise ValueError(f"{self.method}: a claim is held against at least one own knob")

    @property
    def arms(self) -> tuple[Arm, ...]:
        return (self.base, *self.knobs, self.claim, *self.random)

    def parts(self) -> list[tuple[str, Arm]]:
        return ([("base", self.base)] + [("knob", a) for a in self.knobs]
                + [("claim", self.claim)] + [("random", a) for a in self.random])


PLAN_PARTS = ("claim", "knob")                # a plan file's parts; base and random come with


def held_plan(path: Path) -> tuple[ClaimArms, ...]:
    """The held-out stage's arms from `path`, a CSV of method, part, arm: per method one `claim`,
    a candidate of its arms on the 13 (TUNING_PLAN), and one or more `knob`s, its own knobs there;
    its base and the claim's random sets (its form's, ChoiceArms.own_random) come with them.
    Checked here, where the file is read: an unknown method, part or arm, a method without its one
    claim, or an arm named twice raises."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["method", "part", "arm"] or (df == "").any().any():
        raise SystemExit(f"{path}: the columns are method, part, arm, every cell filled")
    tuning = {m.method: m for m in TUNING_PLAN}
    off = sorted(set(df.method) - set(tuning)) + sorted(set(df.part) - set(PLAN_PARTS))
    if off or df.arm.duplicated().any():
        raise SystemExit(f"{path}: unknown methods or parts {off}, or an arm named twice")
    out = []
    for method in dict.fromkeys(df.method):          # in the file's order
        m, g = tuning[method], df[df.method == method]
        by_name = {a.name: a for a in m.arms}
        unknown = sorted(set(g.arm) - set(by_name))
        if unknown:
            raise SystemExit(f"{path}: {unknown} are not {method}'s arms on the 13")
        claims = [by_name[n] for n in g[g.part == "claim"].arm]
        knobs = tuple(by_name[n] for n in g[g.part == "knob"].arm)
        if len(claims) != 1 or claims[0] not in m.candidates or not all(k in m.knobs
                                                                         for k in knobs):
            raise SystemExit(f"{path}: {method} needs one claim among its candidates on the 13 "
                             "and knobs among its knobs there")
        [claim] = claims
        out.append(ClaimArms(method=method, base=m.base, knobs=knobs, claim=claim,
                             random=m.own_random(claim)))
    return tuple(out)


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in rr.SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {rr.SINGLE}")


# ------------------------------------------------------------------ stages
class Verdict(Protocol):
    """How a stage's results are judged: the rule fixed in NOTES before them."""

    @property
    def arms(self) -> tuple[Arm, ...]: ...

    def report(self, stage: Stage) -> None: ...


@dataclass(frozen=True, kw_only=True)
class Stage:
    """One step of the experiment: `blocks`, the verdict's arms, the rows root `rows` (its
    networks, logs and progress in <rows>/scratch, git-ignored), and what its scoring solves
    (`pops` and `scored`, road_rescore.score's)."""

    name: str
    rows: Path
    blocks: tuple[str, ...]
    verdict: Verdict
    pops: tuple[str, ...]
    scored: rr.Scored

    def __post_init__(self) -> None:
        names = self.lineup
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: arm names collide: {names}")

    @property
    def arms(self) -> tuple[Arm, ...]:
        return self.verdict.arms

    @property
    def lineup(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms)

    @property
    def scratch(self) -> Path:
        return self.rows / "scratch"

    def arm_of(self, name: str) -> Arm:
        """The arm a command line names (an unknown name raises)."""
        found = [a for a in self.arms if a.name == name]
        if len(found) != 1:
            raise SystemExit(f"{self.name} has no arm {name!r}")
        return found[0]

    def ids_of(self, arg: str) -> list[str]:
        """Block ids named on the command line, each one of the stage's."""
        ids = arg.split(",")
        off = sorted(set(ids) - set(self.blocks))
        if off or len(set(ids)) != len(ids):
            raise SystemExit(f"{self.name}'s blocks are {self.blocks}, each once: {arg!r}")
        return ids

    def arms_of(self, arg: str) -> list[Arm]:
        """Arms named on the command line, comma-separated, or `all`."""
        return list(self.arms) if arg == "all" else [self.arm_of(n) for n in arg.split(",")]

    def record_path(self, bid: str, name: str) -> Path:
        """The network of record: the fresh run's roads (road_rescore.propose writes them as
        <name>.fresh.parquet and propose_one renames them) or, for a reused arm, item 4's."""
        return rr.net_path(self.scratch, bid, name, False)


# ------------------------------------------------------------------ arms, dry
def arms_table(stage: Stage) -> pd.DataFrame:
    rows = []
    for m in _plan(stage):
        for part, arm in m.parts():
            rows.append(dict(arm=arm.name, method=m.method, part=part, preset=arm.preset,
                             reuse_item4=arm.reuse_item4, **arm.change.record()))
    return pd.DataFrame(rows)


def _plan(stage: Stage
          ) -> tuple[MethodArms, ...] | tuple[ChoiceArms, ...] | tuple[ClaimArms, ...]:
    v = stage.verdict
    if not isinstance(v, Screen | Choice | Claim):
        raise TypeError(f"{stage.name}'s verdict has no plan: {v!r}")
    return v.plan


def write_arms(stage: Stage) -> None:
    df = arms_table(stage)
    stage.rows.mkdir(parents=True, exist_ok=True)
    df.to_csv(stage.rows / "arms.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    reused = int(df.reuse_item4.sum())
    print(f"{len(df)} arms ({reused} reused), {len(df) - reused} to propose per block; written to "
          f"{stage.rows / 'arms.csv'}")


def dry(stage: Stage, bid: str) -> None:
    """Every arm's method on one block, built and not run: its label and identity, the base arms'
    against the presets', every identity distinct, each discount's set and factors."""
    presets = common.arms()
    [b] = common.build_blocks([bid])
    seen: dict[object, str] = {}
    for arm in stage.arms:
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
    print(f"{len(stage.arms)} arms on {bid}: identities distinct, base arms the presets")


# ------------------------------------------------------------------ propose
def propose_one(stage: Stage, bid: str, name: str) -> None:
    """One (block, arm) network, fresh (road_rescore.propose); the roads become the network of
    record."""
    _check_single_threaded()
    arm = stage.arm_of(name)
    if arm.reuse_item4:
        raise SystemExit(f"{name} reuses item 4's network: nothing to propose")
    method = arm.method(common.arms(), bid)
    rr.propose(stage.rows, stage.scratch, bid, name, method, True)
    row = pd.read_parquet(rr.proposal_row(stage.rows, bid, name, True)).iloc[0]
    if row.status == "ok":
        os.replace(rr.net_path(stage.scratch, bid, name, True), stage.record_path(bid, name))


def _take_item4(stage: Stage, bid: str, arm: Arm) -> None:
    """A reused arm's network of record: item 4's network of its preset, copied whole."""
    dest = stage.record_path(bid, arm.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(ITEM4_NETS / bid / f"{arm.preset}.parquet", tmp)
    os.replace(tmp, dest)


def propose(stage: Stage, procs: int, timeout_s: float, ids: list[str]) -> None:
    """Every (block, arm) network without a proposal row, `procs` children at a time, the
    costliest first; a child past `timeout_s` of wall time is killed by PID and recorded. A reused
    arm's network is copied from item 4's."""
    _check_single_threaded()
    for bid in ids:
        for arm in stage.arms:
            if arm.reuse_item4 and not stage.record_path(bid, arm.name).exists():
                _take_item4(stage, bid, arm)
                print(f"{bid} {arm.name}: item 4's {arm.preset} network", flush=True)
    sizes = pd.read_parquet(HERE / "large82_area.parquet").set_index("block")["n"]
    queue = sorted(((bid, arm) for bid in ids for arm in stage.arms if not arm.reuse_item4
                    and not rr.proposal_row(stage.rows, bid, arm.name, True).exists()),
                   key=lambda j: -float(sizes[j[0]]) * COST_GUESS[j[1].group])
    env = {**os.environ, **rr.SINGLE,
           "PYTHONPATH": os.pathsep.join([str(common.REPO), str(HERE)])}
    (stage.scratch / "logs").mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple] = {}              # pid -> (Popen, bid, arm, start, log handle)
    print(f"{time.strftime('%H:%M:%S')} {len(queue)} jobs, {procs} at a time", flush=True)
    while queue or running:
        while queue and len(running) < procs:
            bid, arm = queue.pop(0)
            fh = open(stage.scratch / "logs" / f"propose_{bid}__{arm.name}.log", "w")
            p = subprocess.Popen([sys.executable, __file__, stage.name, "propose-one", bid,
                                  arm.name], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 cwd=str(common.REPO))
            running[p.pid] = (p, bid, arm, time.perf_counter(), fh)
            print(f"{time.strftime('%H:%M:%S')} start pid {p.pid} {bid} {arm.name}", flush=True)
        time.sleep(5.0)
        for pid, (p, bid, arm, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is None and time.perf_counter() - t0 > timeout_s:
                p.kill()                        # by PID, never a pattern
                p.wait()
                prog = stage.scratch / "progress" / f"{bid}__{arm.name}__fresh.json"
                known = json.loads(prog.read_text()) if prog.exists() else {}
                dest = rr.proposal_row(stage.rows, bid, arm.name, True)
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


def run_arm(stage: Stage, bid: str, name: str) -> None:
    """One (block, arm) as a cluster task (cluster.py's `cpu` kind): its network -- proposed
    fresh, or for a reused arm item 4's, from the block's tar -- then its prefixes. Exits non-zero
    when the proposal leaves no network (its row, which comes back, says why)."""
    _check_single_threaded()
    arm = stage.arm_of(name)
    if arm.reuse_item4:
        _take_item4(stage, bid, arm)
    else:
        propose_one(stage, bid, name)
    if not stage.record_path(bid, name).exists():
        raise SystemExit(f"{bid} {name}: no network (its proposal row says why)")
    rr.prefixes(stage.rows, (name,), stage.scratch, 1, [bid])


def recompute(stage: Stage, bid: str, name: str) -> None:
    """A reused arm's preset proposed afresh (road_rescore.propose, fresh: <arm>.fresh.parquet
    beside the network of record, and its proposal row), for check-base to set against item 4's
    network: the reuse holds while the two are the same."""
    _check_single_threaded()
    arm = stage.arm_of(name)
    if not arm.reuse_item4:
        raise SystemExit(f"{name} is proposed, not reused: nothing to recompute")
    rr.propose(stage.rows, stage.scratch, bid, name, arm.method(common.arms(), bid), True)
    row = pd.read_parquet(rr.proposal_row(stage.rows, bid, name, True)).iloc[0]
    if row.status != "ok":
        raise SystemExit(f"{bid} {name}: {row.status}")


def _wkb(roads) -> list[bytes]:
    import shapely
    return [bytes(w) for w in shapely.to_wkb(np.asarray(roads.geometry))]


def _same(a_path: Path, b_path: Path) -> tuple[bool, int, int]:
    """Two networks road for road: the same WKB row for row and the same widths."""
    import geopandas as gpd
    a, b = gpd.read_parquet(a_path), gpd.read_parquet(b_path)
    same = bool(len(a) == len(b) and _wkb(a) == _wkb(b)
                and np.array_equal(a["width_m"].to_numpy(), b["width_m"].to_numpy()))
    return same, len(a), len(b)


def check_base(stage: Stage, ids: list[str], arms: list[Arm]) -> None:
    """`arms`, opaque preset arms, against item 4's network of each one's preset: a proposed base,
    a reused network, and a reused arm's fresh recomputation where `recompute` made one. The same
    WKB row for row and the same widths; raises on any difference: the comparison rests on it."""
    out = []
    for bid in ids:
        for arm in arms:
            if not isinstance(arm.change, AsConfigured):
                raise SystemExit(f"{arm.name} is not a preset as configured")
            theirs = ITEM4_NETS / bid / f"{arm.preset}.parquet"
            mine = [("", stage.record_path(bid, arm.name))]
            fresh = rr.net_path(stage.scratch, bid, arm.name, True)
            if arm.reuse_item4 and fresh.exists():
                mine.append((" (recomputed)", fresh))
            for what, path in mine:
                same, n_mine, n_theirs = _same(path, theirs)
                out.append(dict(block=bid, arm=arm.name + what, reused=arm.reuse_item4,
                                roads=n_mine, item4_roads=n_theirs, same=same))
                print(f"{bid} {arm.name + what:42s} {n_mine:5d} roads, item 4 {n_theirs:5d}: "
                      f"{'identical' if same else 'DIFFERENT'}", flush=True)
    df = pd.DataFrame(out)
    if not df.same.all():
        raise SystemExit(f"{int((~df.same).sum())} networks differ from item 4's")
    print(f"all {len(df)} identical to item 4's networks")


def same_as(stage: Stage, other: Stage, ids: list[str]) -> None:
    """On blocks of both stages, every arm both have (by name): this stage's network against the
    other's, road for road. Prints each; raises on any difference."""
    both = [a.name for a in stage.arms if a.name in set(other.lineup)]
    out = []
    for bid in ids:
        if bid not in other.blocks:
            raise SystemExit(f"{bid} is not one of {other.name}'s blocks")
        for name in both:
            same, n_mine, n_theirs = _same(stage.record_path(bid, name),
                                           other.record_path(bid, name))
            out.append(same)
            print(f"{bid} {name:42s} {n_mine:5d} roads, {other.name} {n_theirs:5d}: "
                  f"{'identical' if same else 'DIFFERENT'}", flush=True)
    print(f"{sum(out)} of {len(out)} networks identical to {other.name}'s ({len(both)} arms in "
          f"both, {len(ids)} blocks)")
    if not all(out):
        raise SystemExit(f"{len(out) - sum(out)} networks differ from {other.name}'s")


def same_lens(stage: Stage, other: Stage, ids: list[str]) -> None:
    """On blocks of both stages, every arm both have whose networks are the same road for road:
    (a) and (c), with B (i), at each of GUARD_D, this stage's against the other's (each stage's
    own scoring, reduced_lens's columns). Prints the largest differences."""
    both = [a.name for a in stage.arms if a.name in set(other.lineup)]
    same = [(bid, n) for bid in ids for n in both
            if _same(stage.record_path(bid, n), other.record_path(bid, n))[0]]
    for d in GUARD_D:
        mine, theirs = reduced_lens(stage, d), reduced_lens(other, d)
        keys = ["block", "method"]
        pairs = pd.DataFrame(same, columns=keys)
        t = pairs.merge(mine, on=keys).merge(theirs, on=keys, suffixes=("", "_other"))
        diffs = {col: float((t[col] - t[col + "_other"]).abs().max()) for col in ("a", "c", "enc_c")}
        print(f"D {d:g}: {len(t)} (block, arm) pairs with the same network on {t.block.nunique()} "
              f"blocks, max |diff| against {other.name}: "
              + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()), flush=True)


def pulled(stage: Stage, ids: list[str], arms: list[Arm]) -> None:
    """Every (block, arm) of `ids` x `arms` has its network of record, ordered network and prefix
    rows here, and a proposed arm its proposal row with status ok: what the cluster's tasks write,
    back through their run's pull. Raises listing what is missing."""
    missing, bad = [], []
    for bid in ids:
        for arm in arms:
            want = [stage.record_path(bid, arm.name),
                    rr.ordered_path(stage.scratch, bid, arm.name),
                    rr.prefix_dir(stage.rows) / f"{bid}__{arm.name}.parquet"]
            if not arm.reuse_item4:
                want.append(rr.proposal_row(stage.rows, bid, arm.name, True))
            missing += [str(p) for p in want if not p.exists()]
            row = rr.proposal_row(stage.rows, bid, arm.name, True)
            if not arm.reuse_item4 and row.exists():
                status = str(pd.read_parquet(row).status.iloc[0])
                if status != "ok":
                    bad.append(f"{bid} {arm.name}: {status}")
    print(f"{len(ids)} blocks x {len(arms)} arms: {len(missing)} files missing, "
          f"{len(bad)} proposals not ok")
    if missing or bad:
        raise SystemExit("\n".join(missing[:20] + bad[:20]))


# ------------------------------------------------------------------ prefixes, tars
def prefixes(stage: Stage, procs: int, ids: list[str]) -> None:
    rr.prefixes(stage.rows, stage.lineup, stage.scratch, procs, ids)


def _rel(p: Path) -> str:
    return str(p.resolve().relative_to(common.REPO))


def _tar(out: Path, bid: str, members: list[Path]) -> None:
    absent = [str(m) for m in members if not m.exists()]
    if absent:
        raise SystemExit(f"{bid}: {len(absent)} inputs absent, e.g. {absent[0]}")
    with tarfile.open(out / f"{bid}.tar", "w") as tf:
        for m in members:
            tf.add(m, arcname=_rel(m))
    print(f"{bid}: {len(members)} files, {(out / f'{bid}.tar').stat().st_size / 2**20:.1f} MB",
          flush=True)


def cpu_tars(stage: Stage, out: Path, ids: list[str]) -> None:
    """Per block, what `arm` reads that git does not carry, at its repo paths: the stored
    clearings the stage's discounts read, and item 4's network of each reused arm's preset."""
    out.mkdir(parents=True, exist_ok=True)
    stored = sorted({str(_discount(a).sets.rows.rows) for a in stage.arms
                     if isinstance(a.change, Discount)})
    for bid in ids:
        members = [HERE / rows / f"{bid}_p2.parquet" for rows in stored]
        members += [ITEM4_NETS / bid / f"{a.preset}.parquet" for a in stage.arms if a.reuse_item4]
        _tar(out, bid, members)
    print("\nthe arms on a CPU node (cluster.py's cpu kind; check the node's load first), one "
          "task per (block, arm):\n"
          f"  uv run python research/roadless/cluster.py submit cpu <run> --node <node> --cpus 1 "
          f"--mem <G> --gpus <at once> --time <T> --items <arms,> --unpack {out}/{{id}}.tar "
          f"<ids,> -- research/roadless/transparency_run.py {stage.name} arm {{id}} {{item}}")


def tars(stage: Stage, out: Path, ids: list[str]) -> None:
    """Per block, a tar of what road_rescore.score reads that git does not carry, at its repo
    paths: every arm's prefix rows and ordered network, and the stored clearings it rescores."""
    out.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        members = [rr.prefix_dir(stage.rows) / f"{bid}__{name}.parquet" for name in stage.lineup]
        members += [rr.ordered_path(stage.scratch, bid, name) for name in stage.lineup]
        members += [p for rel_dir, *_ in rr.STORED.values()
                    if (p := HERE / rel_dir / f"{bid}_p{rr.CLEARING_POWER:g}.parquet").exists()]
        _tar(out, bid, members)
    scored = ",".join([f"{b:g}" for b in stage.scored.budgets]
                      + (["whole"] if stage.scored.whole else []))
    print(f"\nscore ({', '.join(stage.pops)}) on the cluster, one task a block:\n"
          f"  uv run python research/roadless/cluster.py submit blocks <run> --time 3:00:00 "
          f"--unpack {out}/{{id}}.tar {','.join(ids)} -- research/roadless/road_rescore.py score "
          f"{_rel(stage.rows)} {','.join(stage.lineup)} {_rel(stage.scratch)} {MESH} "
          f"{','.join(stage.pops)} {scored} {{id}}")


# ------------------------------------------------------------------ records
_BLOCKS: dict = {}       # the jobs' blocks, built once before the pool forks


class RecordJob(NamedTuple):
    stage: Stage
    bid: str
    name: str


def _prefix_rows(stage: Stage, bid: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(rr.prefix_dir(stage.rows) / f"{bid}__{name}.parquet")


def _record_one(job: RecordJob) -> dict[str, object]:
    """Per (block, arm): the whole network's true displacement under both weightings, the share
    of the true (area-weighted) displacement of its (c) prefix at LENS_D that falls on C, on C'
    and on the arm's own set, and a random set's draw."""
    import geopandas as gpd

    from reblock.budget import road_corridor
    stage, bid, name = job
    arm, b = stage.arm_of(name), _BLOCKS[bid]
    pre = _prefix_rows(stage, bid, name)
    full = pre[pre.full.astype(bool) & (pre["pop"] == "area")].iloc[0]
    m_c = int(pre[(pre["pop"] == "area") & np.isclose(pre.budget, LENS_D)].m.iloc[0])
    ordered = gpd.read_parquet(rr.ordered_path(stage.scratch, bid, name))
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


def _cost(stage: Stage, bid: str, arm: Arm) -> dict[str, object]:
    """What proposing the arm's network cost: its fresh run, or item 4's for a reused one."""
    if arm.reuse_item4:
        pt = rr.proposals_table(ITEM4_ROWS)
        r = pt[(pt.block == bid) & (pt.method == arm.preset)].iloc[0]
        return dict(status=f"item 4: {r.cost_status}", wall_s=r.cost_wall_s, cpu_s=r.cost_cpu_s,
                    peak_rss_gb=r.cost_peak_rss_gb)
    r = pd.read_parquet(rr.proposal_row(stage.rows, bid, arm.name, True)).iloc[0]
    return dict(status=r.status, wall_s=r.wall_s, cpu_s=r.cpu_s, peak_rss_gb=r.peak_rss_gb)


def records(stage: Stage, procs: int, ids: list[str]) -> None:
    _check_single_threaded()
    import multiprocessing
    _BLOCKS.update({b.block_id: b for b in common.build_blocks(ids)})
    jobs = [RecordJob(stage, bid, name) for bid in ids for name in stage.lineup]
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=8) as pool:
        got = list(pool.imap(_record_one, jobs))
    df = pd.DataFrame(got)
    cost = pd.DataFrame([dict(block=bid, arm=arm.name, **_cost(stage, bid, arm))
                         for bid in ids for arm in stage.arms])
    df = df.merge(cost, on=["block", "arm"])
    off = (df.D_area_c - df.D_area_c_row).abs().max()
    if off > 1e-12:
        raise AssertionError(f"the (c) prefix's displacement differs from its row by {off:.1e}")
    df.to_csv(stage.rows / "summary_records.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 1000)
    print(df.drop(columns=["tag", "D_area_c_row"]).to_string(index=False))


# ------------------------------------------------------------------ the sentinel's screen
def _per_block(t: pd.DataFrame, blocks: tuple[str, ...], name: str, col: str) -> pd.Series:
    s = t[t.method == name].set_index("block")[col]
    return s.reindex(list(blocks))


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
    def of(cls, m: MethodArms, blocks: tuple[str, ...], t5: pd.DataFrame,
           simp: pd.Series) -> Reference:
        base = _per_block(t5, blocks, m.base.name, "c")
        rand = pd.DataFrame({a.name: _per_block(t5, blocks, a.name, "c") for a in m.random})
        best = max((m.base, *m.knobs),
                   key=lambda a: float(_per_block(t5, blocks, a.name, "c").mean()))
        return cls(base=base, gap=simp - base, rand_mean=rand.mean(axis=1),
                   rand_sd=rand.std(axis=1, ddof=1), best_knob=best,
                   best_mean=float(_per_block(t5, blocks, best.name, "c").mean()))


def _stats(lens: Mapping[float, pd.DataFrame], blocks: tuple[str, ...], m: MethodArms,
           ref: Reference, arm: Arm, part: str) -> dict[str, object]:
    """An arm's numbers in the screen (NOTES, "Transparency for the road methods")."""
    t5 = lens[LENS_D]
    c = _per_block(t5, blocks, arm.name, "c")
    gain = c - ref.base
    a_loss = {d: float((_per_block(lens[d], blocks, m.base.name, "a")
                        - _per_block(lens[d], blocks, arm.name, "a")).median()) for d in GUARD_D}
    r: dict[str, object] = dict(
        method=m.method, arm=arm.name, part=part,
        **{f"c_{_short(b)}": float(c[b]) for b in blocks},
        mean_c=float(c.mean()), mean_gain=float(gain.mean()), min_gain=float(gain.min()),
        veto=bool((gain < -VETO).any()), above_random=int((c > ref.rand_mean).sum()),
        **{f"a_loss_D{d:g}": a_loss[d] for d in GUARD_D},
        a_ok=all(v <= A_LOSS for v in a_loss.values()),
        **{f"gap_{_short(b)}": float(gain[b] / ref.gap[b]) for b in blocks},
        mean_gap_closed=float((gain / ref.gap).mean()),
        vs_best_knob=float(c.mean()) - ref.best_mean,
        reach_c=bool(_per_block(t5, blocks, arm.name, "reach_area").all()),
        reach_a=bool(all(_per_block(lens[d], blocks, arm.name, "reach_count").all()
                         for d in GUARD_D)))
    r["passes"] = bool(not r["veto"] and float(gain.mean()) > 0 and int(r["above_random"]) >= 3
                       and r["a_ok"])
    return r


def _screen(lens: Mapping[float, pd.DataFrame], blocks: tuple[str, ...], m: MethodArms,
            ref: Reference) -> tuple[Arm, dict[str, dict[str, object]]]:
    """The screen on one method: the main form (its f with the larger mean gain) and every arm's
    numbers, `advances` among them."""
    st = {arm.name: _stats(lens, blocks, m, ref, arm, part) for part, arm in m.parts()}

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


def _simp(blocks: tuple[str, ...]) -> pd.Series:
    """SIMP's default's (c) per block: item 4's rows (the same scorer and mesh)."""
    cl4 = pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, "clearings"))
    return (cl4[(cl4.mesh == "h0.5") & (cl4.clearing == "simp_default")].set_index("block")
            .perm.reindex(list(blocks)))


def _scorer_check(stage: Stage) -> None:
    """The stored clearings rescored in this stage's run against item 4's rows."""
    cl4 = pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, "clearings"))
    mine = rr._read_all(sorted(rr.score_dir(stage.rows, MESH, "area").glob("*__clearings.parquet")))
    both = mine.merge(cl4[["block", "mesh", "clearing", "perm"]],
                      on=["block", "mesh", "clearing"], suffixes=("", "_item4"))
    print(f"SCORER CHECK: the stored clearings rescored here against item 4's rows, "
          f"{len(both)} rows on {both.block.nunique()} blocks "
          f"({', '.join(sorted(set(both.clearing)))}): max |diff| "
          f"{(both.perm - both.perm_item4).abs().max():.1e}")


@dataclass(frozen=True, kw_only=True)
class Screen:
    """The sentinel's screen (NOTES, "Transparency for the road methods: the sentinel"), fixed
    before its results."""

    plan: tuple[MethodArms, ...]

    @property
    def arms(self) -> tuple[Arm, ...]:
        return tuple(a for m in self.plan for a in m.arms)

    def report(self, stage: Stage) -> None:
        """The screen and the tables behind it; writes <rows>/summary_lens.csv (per block and
        arm: every lens) and summary_screen.csv (per arm: the screen's numbers)."""
        pd.set_option("display.width", 250)
        blocks = stage.blocks
        lens = {d: rr.lens_table(stage.rows, stage.lineup, MESH, d) for d in GUARD_D}
        t5 = lens[LENS_D]
        missing = sorted(set(stage.lineup) - set(t5.method))
        if missing or t5.groupby("method").size().ne(len(blocks)).any():
            raise SystemExit(f"not every arm is scored on every block: missing {missing}")
        if any((t.enc_c > 0).any() or (t.enc_b > 0).any() for t in lens.values()):
            raise SystemExit("an arm leaves population enclosed: rank in B's order, not by (c) "
                             "alone")
        rec = pd.read_csv(stage.rows / "summary_records.csv")
        # item 4: SIMP's default on the same scorer and mesh (its rows), and the base networks
        item4 = {d: pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, f"lens_D{d:g}"))
                 for d in GUARD_D}
        simp = _simp(blocks)
        _scorer_check(stage)
        for m in self.plan:
            for arm in (m.base, *[a for a in m.knobs if a.reuse_item4]):
                for d in GUARD_D:
                    ref4 = item4[d][item4[d].method == arm.preset].set_index("block")
                    diffs = [float((_per_block(lens[d], blocks, arm.name, col)
                                    - ref4[col].reindex(list(blocks))).abs().max())
                             for col in ("a", "b", "c")]
                    print(f"  {arm.name} at D {d:g} against item 4's {arm.preset}: max |diff| "
                          f"(a) {diffs[0]:.1e}, (b) {diffs[1]:.1e}, (c) {diffs[2]:.1e}")
        screen_rows = []
        for m in self.plan:
            ref = Reference.of(m, blocks, t5, simp)
            main, st = _screen(lens, blocks, m, ref)
            screen_rows += list(st.values())
            short = {a.name: a.change.label for a in m.arms}
            print(f"\n{m.method.upper()}: (c) at area D {LENS_D:g} per block, its mean gain over "
                  f"base, the gap to SIMP's default closed (mean over blocks), the (a) loss "
                  f"against base (median over blocks) at count D 0.05 | 0.10, the blocks above "
                  f"the random sets' mean; * vetoed, > advances")
            hdr = "".join(f"{_short(b):>9s}" for b in blocks)
            print(f"  {'arm':28s}{hdr}     gain     gap  a-loss .05 |   .10  >rand")
            for _part, arm in m.parts():
                r = st[arm.name]
                mark = ("*" if r["veto"] else "") + (">" if r["advances"] else "")
                cells = "".join(f"{r['c_' + _short(b)]:9.4f}" for b in blocks)
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
        pd.DataFrame(screen_rows).to_csv(stage.rows / "summary_screen.csv", index=False)
        keep = ["block", "method", "n", "m_count", "a", "b", "enc_b", "m_area", "c", "enc_c",
                "Dc_count", "Da_area", "road_m_count", "road_m_area", "reach_count", "reach_area"]
        out = pd.concat([lens[d][keep].assign(budget=d) for d in GUARD_D], ignore_index=True)
        out.rename(columns={"method": "arm"}).merge(
            rec[["block", "arm", "share_simp", "share_own", "area_ratio", "jaccard"]],
            on=["block", "arm"]).to_csv(stage.rows / "summary_lens.csv", index=False)


# ------------------------------------------------------------------ the 13's choice
def reduced_lens(stage: Stage, budget: float) -> pd.DataFrame:
    """Per (block, arm) at `budget`: (a), the shipped permeability, on the count prefix (its
    prefix row's P_old, as rr.lens_table reads it through the count rows), and (c) on the area
    prefix (its area score row's P2), with B (i), enc_c, the population share that prefix leaves
    enclosed; each prefix's displacement and whether it reaches the budget. For a stage whose
    scoring solves (c) alone (Stage.pops)."""
    keys = ["block", "method"]
    pre = rr._read_all([rr.prefix_dir(stage.rows) / f"{b}__{n}.parquet"
                        for b in stage.blocks for n in stage.lineup])
    cnt = pre[(pre["pop"] == "count") & np.isclose(pre.budget, budget)]
    cnt = cnt[keys + ["n", "m", "D_count", "P_old"]].rename(
        columns={"m": "m_count", "D_count": "Dc_count", "P_old": "a"})
    area = rr.score_rows(stage.rows, stage.lineup, MESH, "area")
    area = area[np.isclose(area.budget, budget)][keys + ["m", "D_area", "P2", "enclosed"]]
    area = area.rename(columns={"m": "m_area", "D_area": "Da_area", "P2": "c",
                                "enclosed": "enc_c"})
    t = cnt.merge(area, on=keys, how="outer")
    t["reach_count"] = t.Dc_count >= budget - 1e-9
    t["reach_area"] = t.Da_area >= budget - 1e-9
    return t


def boot_weights(n: int) -> np.ndarray:
    """The Bayesian bootstrap's weights over `n` blocks: BOOT_DRAWS Dirichlet(1, ..., 1) draws,
    seeded BOOT_SEED, one matrix (draws x blocks) for every arm, so the intervals are paired."""
    return np.random.default_rng(BOOT_SEED).dirichlet(np.ones(n), size=BOOT_DRAWS)


def _wide(t: pd.DataFrame, blocks: tuple[str, ...], names: tuple[str, ...],
          col: str) -> pd.DataFrame:
    """`col` per arm (rows, in `names`' order) and block (columns, in `blocks`' order)."""
    w = t.pivot(index="method", columns="block", values=col)
    return w.reindex(index=list(names), columns=list(blocks))


def _complete(stage: Stage) -> dict[float, pd.DataFrame]:
    """reduced_lens at each of GUARD_D, every arm with (a) and (c) on every block (raises
    otherwise); notes each network that falls short of a budget (its prefix is all of it)."""
    lens = {d: reduced_lens(stage, d) for d in GUARD_D}
    for d, t in lens.items():
        n = t.groupby("method").size()
        if (sorted(set(stage.lineup) - set(t.method)) or n.ne(len(stage.blocks)).any()
                or t[["a", "c"]].isna().any().any()):
            raise SystemExit(f"not every arm has (a) and (c) on every block at D {d:g}")
        short_of = t[~(t.reach_count & t.reach_area)]
        for _, r in short_of.iterrows():
            print(f"  NOTE: {r.block} {r.method}'s whole network falls short of D {d:g} "
                  f"(count {r.Dc_count:.4f}, area {r.Da_area:.4f}): its prefix is all of it")
    return lens


def _against_item4(lens: Mapping[float, pd.DataFrame], blocks: tuple[str, ...],
                   arms: list[Arm]) -> None:
    """Opaque preset arms' (a), (c) and B (i) at each of GUARD_D against item 4's rows (the same
    scorer and mesh): the reduced scoring's check where the networks are item 4's."""
    item4 = {d: pd.read_csv(rr.summary_path(ITEM4_ROWS, MESH, f"lens_D{d:g}")) for d in GUARD_D}
    for arm in arms:
        for d in GUARD_D:
            ref4 = item4[d][item4[d].method == arm.preset].set_index("block")
            diffs = [float((_per_block(lens[d], blocks, arm.name, col)
                            - ref4[col].reindex(list(blocks))).abs().max())
                     for col in ("a", "c", "enc_c")]
            print(f"  {arm.name} at D {d:g} against item 4's {arm.preset}: max |diff| "
                  f"(a) {diffs[0]:.1e}, (c) {diffs[1]:.1e}, B (i) {diffs[2]:.1e}")


@dataclass(frozen=True, kw_only=True)
class Choice:
    """The 13 tuning blocks' choice rule (NOTES, "Transparency for the road methods: the 13
    tuning blocks"), fixed before any result. Per method, among its candidates (C and C' at
    F_ONE), the one with the largest mean (c) gain over base at area D LENS_D that also
      (i) is on no block more than VETO below base on (c), nor behind it on B (i) (it leaves
          more population enclosed than base: B's order puts (i) first);
      (ii) loses at most A_LOSS of (a) against base, the median over the blocks, at count D 0.05
          and at 0.10;
      (iii) gains more on average than the best own-knob point (the knob with the largest mean
          gain; the desire-routed preset is one of cycle_native's knobs);
      (iv) has a mean (c) above its own random sets' mean (the five seeds in its form).
    The chosen arm, with base, the best own-knob point and its random sets, goes to the
    held-out blocks; a method with no candidate that qualifies stops there. Beside it: each arm's
    mean gain with a Bayesian-bootstrap 95% interval over the blocks (paired by block: one
    Dirichlet(1, ..., 1) weight matrix, BOOT_DRAWS draws seeded BOOT_SEED, for every arm), the
    share of the gap to SIMP's default closed, and the per-block values."""

    plan: tuple[ChoiceArms, ...]

    @property
    def arms(self) -> tuple[Arm, ...]:
        return tuple(a for m in self.plan for a in m.arms)

    def report(self, stage: Stage) -> None:
        """The rule's verdict and the tables behind it; writes <rows>/summary_choice.csv (per
        arm) and summary_lens.csv (per block and arm, at each of GUARD_D)."""
        pd.set_option("display.width", 250)
        blocks = stage.blocks
        lens = _complete(stage)
        rec = pd.read_csv(stage.rows / "summary_records.csv")
        simp = _simp(blocks)
        _scorer_check(stage)
        _against_item4(lens, blocks, [a for m in self.plan
                                      for a in (m.base, *[k for k in m.knobs if k.reuse_item4])])
        W = boot_weights(len(blocks))
        out = []
        for m in self.plan:
            st, best_knob, chosen = self.decide(m, lens, simp, W)
            out += list(st.values())
            self._print(m, st, _wide(lens[LENS_D], blocks, tuple(a.name for a in m.arms), "c"),
                        simp, best_knob, chosen)
        pd.DataFrame(out).to_csv(stage.rows / "summary_choice.csv", index=False)
        keep = ["block", "method", "n", "m_count", "a", "m_area", "c", "enc_c", "Dc_count",
                "Da_area", "reach_count", "reach_area"]
        both = pd.concat([lens[d][keep].assign(budget=d) for d in GUARD_D], ignore_index=True)
        both.rename(columns={"method": "arm"}).merge(
            rec[["block", "arm", "share_simp", "share_own", "area_ratio", "jaccard"]],
            on=["block", "arm"]).to_csv(stage.rows / "summary_lens.csv", index=False)

    @staticmethod
    def decide(m: ChoiceArms, lens: Mapping[float, pd.DataFrame], simp: pd.Series,
               W: np.ndarray) -> tuple[dict[str, dict[str, object]], Arm, Arm | None]:
        """The rule on one method: every arm's numbers (per arm, the columns of
        summary_choice.csv), its best own-knob point and the chosen candidate (None: none
        qualifies). `lens` per budget of GUARD_D (reduced_lens's columns), `simp` SIMP's default's
        (c) per block (its index the blocks, in order), `W` the bootstrap's weights (draws x
        blocks, boot_weights)."""
        blocks = tuple(simp.index)
        names = tuple(a.name for a in m.arms)
        c5 = _wide(lens[LENS_D], blocks, names, "c")
        enc = _wide(lens[LENS_D], blocks, names, "enc_c")
        a = {d: _wide(lens[d], blocks, names, "a") for d in GUARD_D}
        base = m.base.name
        gain = c5.sub(c5.loc[base], axis=1)
        behind = enc.gt(enc.loc[base] + ENCLOSED_TOL, axis=1)
        ahead = enc.lt(enc.loc[base] - ENCLOSED_TOL, axis=1)
        st: dict[str, dict[str, object]] = {}
        for part, arm in m.parts():
            g = gain.loc[arm.name].to_numpy(dtype=float)
            boot = W @ g
            loss = {d: float((a[d].loc[base] - a[d].loc[arm.name]).median()) for d in GUARD_D}
            up = ahead.loc[arm.name] | (~behind.loc[arm.name] & (gain.loc[arm.name] > 0))
            st[arm.name] = dict(
                method=m.method, arm=arm.name, part=part, mean_c=float(c5.loc[arm.name].mean()),
                mean_gain=float(g.mean()), lo=float(np.quantile(boot, 0.025)),
                hi=float(np.quantile(boot, 0.975)), min_gain=float(g.min()),
                blocks_up=int(up.sum()), behind_i=int(behind.loc[arm.name].sum()),
                ahead_i=int(ahead.loc[arm.name].sum()),
                veto=bool((g < -VETO).any() or behind.loc[arm.name].any()),
                **{f"a_loss_D{d:g}": loss[d] for d in GUARD_D},
                a_ok=all(v <= A_LOSS for v in loss.values()),
                gap_closed=float((gain.loc[arm.name] / (simp - c5.loc[base])).mean()),
                **{f"c_{_short(b)}": float(c5.loc[arm.name, b]) for b in blocks})
        best_knob = max(m.knobs, key=lambda k: float(st[k.name]["mean_gain"]))
        knob_gain = float(st[best_knob.name]["mean_gain"])
        for arm in m.arms:
            st[arm.name].update(best_knob=arm is best_knob,
                                vs_best_knob=float(st[arm.name]["mean_gain"]) - knob_gain)
        qualifying = []
        for cand in m.candidates:
            r = st[cand.name]
            own = [x.name for x in m.own_random(cand)]
            rand_c = float(c5.loc[own].to_numpy().mean())
            r.update(own_random_mean_c=rand_c, above_own_random=float(r["mean_c"]) > rand_c,
                     beats_knob=float(r["mean_gain"]) > knob_gain)
            r["qualifies"] = bool(not r["veto"] and r["a_ok"] and r["beats_knob"]
                                  and r["above_own_random"])
            if r["qualifies"]:
                qualifying.append(cand)
        chosen = max(qualifying, key=lambda x: float(st[x.name]["mean_gain"]), default=None)
        for arm in m.arms:
            st[arm.name]["chosen"] = arm is chosen
            st[arm.name]["goes_on"] = chosen is not None and (
                arm is chosen or arm is m.base or arm is best_knob
                or arm in m.own_random(chosen))
        return st, best_knob, chosen

    @staticmethod
    def _print(m: ChoiceArms, st: Mapping[str, Mapping[str, object]], c5: pd.DataFrame,
               simp: pd.Series, best_knob: Arm, chosen: Arm | None) -> None:
        short = {a.name: a.change.label for a in m.arms}
        blocks = tuple(c5.columns)
        print(f"\n{m.method.upper()}: (c) at area D {LENS_D:g}, the mean gain over base with its "
              f"Bayesian-bootstrap 95% interval, the blocks up in B's order, the worst block, "
              f"the (a) loss against base (median) at count D 0.05 | 0.10, the gap to SIMP's "
              f"default closed (mean), against the best own knob; * vetoed, = chosen, + goes on")
        print(f"  {'arm':26s} {'mean (c)':>8s} {'gain':>8s} {'95% interval':>19s} {'up':>3s} "
              f"{'worst':>8s} {'a .05':>7s} {'a .10':>7s} {'gap':>7s} {'vs knob':>8s}")
        for _part, arm in m.parts():
            r = st[arm.name]
            mark = (("*" if r["veto"] else "") + ("=" if r["chosen"] else "")
                    + ("+" if r["goes_on"] else ""))
            print(f"  {mark:3s}{short[arm.name]:23s} {r['mean_c']:8.4f} {r['mean_gain']:+8.4f} "
                  f"[{r['lo']:+8.4f},{r['hi']:+8.4f}] {r['blocks_up']:3d} {r['min_gain']:+8.4f} "
                  f"{r['a_loss_D0.05']:+7.4f} {r['a_loss_D0.1']:+7.4f} {r['gap_closed']:+7.3f} "
                  f"{r['vs_best_knob']:+8.4f}")
        for cand in m.candidates:
            r = st[cand.name]
            print(f"  {short[cand.name]}: own random sets' mean (c) {r['own_random_mean_c']:.4f} "
                  f"(above: {r['above_own_random']}), beats the best knob "
                  f"{short[best_knob.name]}: {r['beats_knob']}, (a) ok: {r['a_ok']}, vetoed: "
                  f"{r['veto']} (behind on B (i) on {r['behind_i']} blocks) -> qualifies: "
                  f"{r['qualifies']}")
        print(f"  chosen: {short[chosen.name] if chosen is not None else 'none: the method stops'}")
        print(f"  per block (c): {'':12s}" + "".join(f"{_short(b):>8s}" for b in blocks))
        for _part, arm in m.parts():
            print(f"    {short[arm.name]:26s}" + "".join(f"{v:8.4f}" for v in c5.loc[arm.name]))
        print(f"    {'SIMP default':26s}" + "".join(f"{v:8.4f}" for v in simp))


# ------------------------------------------------------------------ the held-out blocks' claim
def _interval(W: np.ndarray, diff: np.ndarray, keep: np.ndarray) -> tuple[float, float, float]:
    """The mean of `diff` over the blocks `keep` and its Bayesian-bootstrap 95% interval: `W`'s
    weights of those blocks renormalized, a Dirichlet(1, ..., 1) over them, paired with every
    other interval drawn from `W`. No block kept: NaN, which holds nothing."""
    if not keep.any():
        return np.nan, np.nan, np.nan
    w = W[:, keep] / W[:, keep].sum(axis=1, keepdims=True)
    boot = w @ diff[keep]
    return (float(diff[keep].mean()), float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)))


class RuleRow(NamedTuple):
    """One test of a method's claim on the held-out blocks (a row of summary_rules.csv)."""

    method: str
    rule: int              # 1 -- 4, as NOTES numbers them
    against: str           # what the claim is set against
    blocks: int            # the blocks the test is over
    behind_i: int          # rule 1: the claim's blocks behind base on B (i); rule 2: the knob's
    value: float           # 1, 2: mean (c) gain; 3: mean difference; 4: median (a) loss
    lo: float              # 1 -- 3: the 95% interval
    hi: float
    holds: bool


@dataclass(frozen=True, kw_only=True)
class Claim:
    """The held-out blocks' rules (NOTES, "Transparency for the road methods: the held-out
    blocks"), fixed before any held-out result. Per method, over the blocks, paired by block,
    each interval a Bayesian bootstrap's 95% (boot_weights over the blocks: one matrix for every
    arm and test), "above 0" its lower end above 0:
      1. (primary) the claim's mean (c) gain over base at area D LENS_D, its interval above 0,
         and the claim behind base on B (i) (it leaves more population enclosed) on no block;
      2. against each own knob, on the blocks where that knob is not behind base on B (i) (the
         owner's amendment, 2026-10-12): the claim's mean (c) gain over the knob, its interval
         above 0 (the weights of those blocks renormalized); for every knob;
      3. the claim's (c) above its random sets' per-block mean, the mean difference's interval
         above 0;
      4. the claim's median (a) loss against base at most A_LOSS at count D 0.05, and at D 0.10.
    A shift of the frontier on the held-out blocks if 1 -- 4 all hold; otherwise the rules that
    fail. Beside it, description, not verdict: per arm the median gain, the mean without the
    three blocks of largest gain, the blocks up in B's order, the gap to SIMP's default closed,
    (c) at area D 0.10, the share of the (c) prefix's displacement on C, and the random draws'
    area ratio and overlap."""

    plan: tuple[ClaimArms, ...]

    @property
    def arms(self) -> tuple[Arm, ...]:
        return tuple(a for m in self.plan for a in m.arms)

    def report(self, stage: Stage) -> None:
        """The rules' verdict and the tables behind it; writes <rows>/summary_rules.csv (per
        method and test), summary_claim.csv (per arm) and summary_lens.csv (per block and arm,
        at each of GUARD_D)."""
        pd.set_option("display.width", 250)
        blocks = stage.blocks
        lens = _complete(stage)
        rec = pd.read_csv(stage.rows / "summary_records.csv")
        simp = _simp(blocks)
        _scorer_check(stage)
        _against_item4(lens, blocks, [m.base for m in self.plan])
        W = boot_weights(len(blocks))
        tests: list[RuleRow] = []
        stats: list[dict[str, object]] = []
        for m in self.plan:
            t, st = self.judge(m, lens, simp, W)
            for arm in m.arms:
                r = rec[rec.arm == arm.name]
                if len(r) != len(blocks):
                    raise SystemExit(f"{arm.name}: records on {len(r)} of {len(blocks)} blocks")
                st[arm.name].update(
                    share_simp=float(r.share_simp.mean()), share_own=float(r.share_own.mean()),
                    area_ratio_median=float(r.area_ratio.median()),
                    area_ratio_min=float(r.area_ratio.min()),
                    area_ratio_max=float(r.area_ratio.max()),
                    jaccard_median=float(r.jaccard.median()), jaccard_max=float(r.jaccard.max()),
                    D_area_whole_min=float(r.D_area_whole.min()))
            tests += t
            stats += list(st.values())
            draws = rec[rec.arm.isin([a.name for a in m.random])]
            own = draws.groupby("arm").share_own.mean()
            self._print(m, t, st, simp, (
                f"random sets ({len(draws)} draws): their own share {own.min():.3f} -- "
                f"{own.max():.3f}; area over C's median {draws.area_ratio.median():.4f} (p10 "
                f"{draws.area_ratio.quantile(0.1):.3f}, p90 {draws.area_ratio.quantile(0.9):.3f}, "
                f"{draws.area_ratio.min():.3f} -- {draws.area_ratio.max():.3f}); Jaccard with C "
                f"median {draws.jaccard.median():.3f}, at most {draws.jaccard.max():.3f}"))
        pd.DataFrame([x._asdict() for x in tests]).to_csv(stage.rows / "summary_rules.csv",
                                                           index=False)
        pd.DataFrame(stats).to_csv(stage.rows / "summary_claim.csv", index=False)
        keep = ["block", "method", "n", "m_count", "a", "m_area", "c", "enc_c", "Dc_count",
                "Da_area", "reach_count", "reach_area"]
        both = pd.concat([lens[d][keep].assign(budget=d) for d in GUARD_D], ignore_index=True)
        both.rename(columns={"method": "arm"}).merge(
            rec[["block", "arm", "share_simp", "share_own", "area_ratio", "jaccard"]],
            on=["block", "arm"]).to_csv(stage.rows / "summary_lens.csv", index=False)

    @staticmethod
    def judge(m: ClaimArms, lens: Mapping[float, pd.DataFrame], simp: pd.Series,
              W: np.ndarray) -> tuple[list[RuleRow], dict[str, dict[str, object]]]:
        """The rules on one method: each test's row, and every arm's numbers (summary_claim.csv's
        columns but the records'). `lens` per budget of GUARD_D (reduced_lens's columns), `simp`
        SIMP's default's (c) per block (its index the blocks, in order), `W` the bootstrap's
        weights (draws x blocks, boot_weights)."""
        blocks = tuple(simp.index)
        names = tuple(a.name for a in m.arms)
        c = {d: _wide(lens[d], blocks, names, "c") for d in GUARD_D}
        enc = {d: _wide(lens[d], blocks, names, "enc_c") for d in GUARD_D}
        a = {d: _wide(lens[d], blocks, names, "a") for d in GUARD_D}
        base, claim = m.base.name, m.claim.name
        behind = {d: enc[d].gt(enc[d].loc[base] + ENCLOSED_TOL, axis=1) for d in GUARD_D}
        ahead = {d: enc[d].lt(enc[d].loc[base] - ENCLOSED_TOL, axis=1) for d in GUARD_D}
        every = np.ones(len(blocks), dtype=bool)
        c5, behind5 = c[LENS_D], behind[LENS_D]

        def gain(name: str, d: float) -> np.ndarray:
            return (c[d].loc[name] - c[d].loc[base]).to_numpy(dtype=float)

        def a_loss(name: str, d: float) -> float:
            return float((a[d].loc[base] - a[d].loc[name]).median())

        claim_behind = int(behind5.loc[claim].sum())
        mean, lo, hi = _interval(W, gain(claim, LENS_D), every)
        tests = [RuleRow(method=m.method, rule=1, against="base", blocks=len(blocks),
                         behind_i=claim_behind, value=mean, lo=lo, hi=hi,
                         holds=bool(lo > 0 and claim_behind == 0))]
        for k in m.knobs:
            kept = ~behind5.loc[k.name].to_numpy(dtype=bool)
            diff = (c5.loc[claim] - c5.loc[k.name]).to_numpy(dtype=float)
            mean, lo, hi = _interval(W, diff, kept)
            tests.append(RuleRow(method=m.method, rule=2, against=k.change.label,
                                 blocks=int(kept.sum()), behind_i=int((~kept).sum()), value=mean,
                                 lo=lo, hi=hi, holds=bool(lo > 0)))
        rand = c5.loc[[r.name for r in m.random]].mean(axis=0)
        mean, lo, hi = _interval(W, (c5.loc[claim] - rand).to_numpy(dtype=float), every)
        tests.append(RuleRow(method=m.method, rule=3, against="random sets", blocks=len(blocks),
                             behind_i=0, value=mean, lo=lo, hi=hi, holds=bool(lo > 0)))
        for d in GUARD_D:
            v = a_loss(claim, d)
            tests.append(RuleRow(method=m.method, rule=4, against=f"(a) at count D {d:g}",
                                 blocks=len(blocks), behind_i=0, value=v, lo=np.nan, hi=np.nan,
                                 holds=v <= A_LOSS))
        gap = (simp - c5.loc[base]).to_numpy(dtype=float)
        st: dict[str, dict[str, object]] = {}
        for part, arm in m.parts():
            g5, g10 = gain(arm.name, LENS_D), gain(arm.name, 0.1)
            mean5, lo5, hi5 = _interval(W, g5, every)
            mean10, lo10, hi10 = _interval(W, g10, every)
            up = {d: ahead[d].loc[arm.name] | (~behind[d].loc[arm.name] & (gain(arm.name, d) > 0))
                  for d in GUARD_D}
            top3 = np.argsort(-g5, kind="stable")[:3]          # the three largest gains
            rest = np.ones(len(blocks), dtype=bool)
            rest[top3] = False
            st[arm.name] = dict(
                method=m.method, arm=arm.name, part=part, mean_c=float(c5.loc[arm.name].mean()),
                mean_gain=mean5, lo=lo5, hi=hi5, median_gain=float(np.median(g5)),
                mean_gain_without_top3=float(g5[rest].mean()),
                top3=",".join(_short(blocks[i]) for i in top3), min_gain=float(g5.min()),
                blocks_up=int(up[LENS_D].sum()), behind_i=int(behind5.loc[arm.name].sum()),
                ahead_i=int(ahead[LENS_D].loc[arm.name].sum()),
                **{f"a_loss_D{d:g}": a_loss(arm.name, d) for d in GUARD_D},
                gap_closed=float((g5 / gap).mean()), gap_closed_pooled=float(g5.sum() / gap.sum()),
                mean_c_D10=float(c[0.1].loc[arm.name].mean()), mean_gain_D10=mean10,
                lo_D10=lo10, hi_D10=hi10, blocks_up_D10=int(up[0.1].sum()),
                **{f"c_{_short(b)}": float(c5.loc[arm.name, b]) for b in blocks})
        return tests, st

    @staticmethod
    def verdict(tests: list[RuleRow]) -> str:
        """`tests`' verdict, one method's: a shift if every rule holds, else the rules that fail."""
        failing = sorted({t.rule for t in tests if not t.holds})
        if not failing:
            return "a shift of the frontier on the held-out blocks: rules 1 -- 4 all hold"
        return "no shift: rule " + ", ".join(str(r) for r in failing) + " fails"

    @staticmethod
    def _print(m: ClaimArms, tests: list[RuleRow], st: Mapping[str, Mapping[str, object]],
               simp: pd.Series, draws: str) -> None:
        short = {a.name: a.change.label for a in m.arms}
        print(f"\n{m.method.upper()} on {len(simp)} held-out blocks: (c) at area D {LENS_D:g}, "
              f"the mean gain over base with its 95% interval, the median gain, the mean without "
              f"the three largest gains, the blocks up in B's order, the worst block, the (a) "
              f"loss against base (median) at count D 0.05 | 0.10, the gap to SIMP's default "
              f"closed (mean of the blocks'), the mean gain at area D 0.10, the share of the (c) "
              f"prefix's displacement on C")
        print(f"  {'arm':23s} {'mean (c)':>8s} {'gain':>8s} {'95% interval':>19s} {'median':>8s} "
              f"{'w/o top3':>8s} {'up':>3s} {'worst':>8s} {'a .05':>7s} {'a .10':>7s} "
              f"{'gap':>7s} {'gain .10':>8s} {'on C':>6s}")
        for _part, arm in m.parts():
            r = st[arm.name]
            print(f"  {short[arm.name]:23s} {r['mean_c']:8.4f} {r['mean_gain']:+8.4f} "
                  f"[{r['lo']:+8.4f},{r['hi']:+8.4f}] {r['median_gain']:+8.4f} "
                  f"{r['mean_gain_without_top3']:+8.4f} {r['blocks_up']:3d} {r['min_gain']:+8.4f} "
                  f"{r['a_loss_D0.05']:+7.4f} {r['a_loss_D0.1']:+7.4f} {r['gap_closed']:+7.3f} "
                  f"{r['mean_gain_D10']:+8.4f} {r['share_simp']:6.3f}")
        cl = st[m.claim.name]
        print(f"  the claim's three largest gains: {cl['top3']}; at area D 0.10 "
              f"{cl['mean_gain_D10']:+.4f} [{cl['lo_D10']:+.4f}, {cl['hi_D10']:+.4f}], up on "
              f"{cl['blocks_up_D10']}; the gap closed, pooled: {cl['gap_closed_pooled']:+.3f}")
        print(f"  {draws}")
        for t in tests:
            what = {1: "the claim's mean (c) gain over base",
                    2: f"the claim's mean (c) gain over {t.against} where it is not behind base "
                       f"on B (i)",
                    3: "the claim's mean (c) above its random sets' per-block mean",
                    4: f"the claim's median {t.against} loss against base"}[t.rule]
            num = (f"{t.value:+.4f} (at most {A_LOSS:g})" if t.rule == 4 else
                   f"{t.value:+.4f} [{t.lo:+.4f}, {t.hi:+.4f}]")
            extra = (f", behind base on B (i) on {t.behind_i}" if t.rule == 1 else
                     f", on {t.blocks} blocks ({t.behind_i} left out)" if t.rule == 2 else "")
            print(f"  rule {t.rule}: {what}: {num}{extra} -> {'holds' if t.holds else 'FAILS'}")
        print(f"  verdict: {Claim.verdict(tests)}")


# ------------------------------------------------------------------ the stages
SENTINEL_STAGE = Stage(
    name="sentinel", rows=HERE / "transparency_rows", blocks=SENTINEL, verdict=Screen(plan=PLAN),
    pops=("count", "area"), scored=rr.Scored(budgets=rr.BUDGETS, whole=True))
TUNING_STAGE = Stage(
    name="tuning", rows=HERE / "transparency_rows" / "tuning", blocks=simp_compare.TUNING,
    verdict=Choice(plan=TUNING_PLAN), pops=("area",), scored=rr.Scored(budgets=GUARD_D,
                                                                       whole=False))


def held_stage() -> Stage:
    """The held-out blocks, scored as the 13; its arms read from HELD_PLAN (held_plan)."""
    return Stage(name="held", rows=HELD_ROWS, blocks=HELD, verdict=Claim(plan=held_plan(HELD_PLAN)),
                 pops=("area",), scored=rr.Scored(budgets=GUARD_D, whole=False))


# Each stage by name, built when named (the held stage reads its plan file then).
STAGES: dict[str, Callable[[], Stage]] = {"sentinel": lambda: SENTINEL_STAGE,
                                          "tuning": lambda: TUNING_STAGE, "held": held_stage}


# ------------------------------------------------------------------ main
def _stage(name: str) -> Stage:
    if name not in STAGES:
        raise SystemExit(f"stages are {sorted(STAGES)}, not {name!r}")
    return STAGES[name]()


if __name__ == "__main__":
    a = sys.argv[1:]
    stage, cmd = _stage(a[0]), a[1]
    if cmd == "arms":
        write_arms(stage)
    elif cmd == "dry":
        dry(stage, a[2])
    elif cmd == "propose":
        propose(stage, int(a[2]), float(a[3]) * 3600, stage.ids_of(a[4]))
    elif cmd == "propose-one":
        try:
            propose_one(stage, a[2], a[3])
        except Exception:
            traceback.print_exc()
            raise
    elif cmd == "arm":
        run_arm(stage, stage.ids_of(a[2])[0], a[3])
    elif cmd == "recompute":
        recompute(stage, stage.ids_of(a[2])[0], a[3])
    elif cmd == "check-base":
        check_base(stage, stage.ids_of(a[2]), stage.arms_of(a[3]))
    elif cmd == "same-as":
        same_as(stage, _stage(a[2]), stage.ids_of(a[3]))
    elif cmd == "same-lens":
        same_lens(stage, _stage(a[2]), stage.ids_of(a[3]))
    elif cmd == "pulled":
        pulled(stage, stage.ids_of(a[2]), stage.arms_of(a[3]))
    elif cmd == "prefixes":
        prefixes(stage, int(a[2]), stage.ids_of(a[3]))
    elif cmd == "cpu-tars":
        cpu_tars(stage, Path(a[2]), stage.ids_of(a[3]))
    elif cmd == "tars":
        tars(stage, Path(a[2]), stage.ids_of(a[3]))
    elif cmd == "records":
        records(stage, int(a[2]), stage.ids_of(a[3]))
    elif cmd == "report":
        stage.verdict.report(stage)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
