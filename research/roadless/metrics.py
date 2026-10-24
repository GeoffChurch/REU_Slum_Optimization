"""A run's metric, resolved once where its configuration is read (NOTES, "The complex as the
runtime: one fixed mesh"): the gap complex's (gap_clear.ComplexMetric: its name carries every
constant that can change a score, the code's among them; its factorization the run's device's,
gap_factor.factorization_of). The raster's metric is retired (NOTES, "The complex as the runtime:
the raster retired"): its rows stand on disk as records no code reads, and no complex name reads as
one of its names (gap_clear.check_disjoint).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/metrics.py
"""
from __future__ import annotations

import dataclasses

import gap_clear
import gap_complex as gc
import gap_exits as gx
import gap_factor
import gap_walls

COMPLEX = "cx+"           # every complex metric's name starts so (gap_clear.ComplexMetric.name)
# The per-block metric (owner 2026-10-14: the shy distance at 0.5 m on the gap complex, its Spec and
# FlowSpec as phase 1 fixed them, each kblock's whole street ring its exit), this code's constants'
# digest ending its name: the runtime until the consolidated pass, and the core every guard of the
# complex checks against its oracles (gap_clear_check)
PER_BLOCK = f"cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v{gap_clear.CODE}"
# The drivable streets: the highway classes (`_link` folded) the street boost raises where they are
# tagged paved (owner 2026-10-16: "a 10x boost is I think reasonable" on drivable streets;
# footways, paths, tracks not; 2026-10-20: "only boost if it's explicitly tagged paved")
DRIVE_CLASSES = ("living_street", "motorway", "primary", "residential", "secondary", "service",
                 "tertiary", "trunk", "unclassified")
# The one metric for every block and region (owner 2026-10-14 -- 16; BACKLOG 4, decisions (1) --
# (4); NOTES, "The consolidated pass"): the per-block metric with the region model's exits, each
# spelt in the name (gap_exits' grammar, read back by gap_exits.exits_of): the railways and water
# as barriers 8 m wide, open at real crossings 3 m wide (`B8c3`); the edge open, a block's street
# ring its exit; the leaks' central constants, chosen once on 5810@major's closed reference for a
# 500 m median trip with 90% of the leak at buildings (`Lo6214b562`: region_leaks.CALIBRATED); the
# paved drivable streets' conductance 10x, and a proposed road's corridor's (`S10spavedh<classes>`)
BARRIERS = gx.OsmBarriers(line_m=8.0, cross_m=3.0)
LEAK = gx.InternalLeak(open_m=6214.0, foot_m=562.0, sites=gx.LiveSites())
BOOST = gx.StreetBoost(factor=10.0, surfaces=gx.PAVED, classes=DRIVE_CLASSES)
EXITS = gx.Exits(barriers=BARRIERS, edge=gx.OpenEdge(), leak=LEAK, boost=BOOST)
RUNTIME = PER_BLOCK + EXITS.suffix
# The uniform rule for walled-in homes, measured beside RUNTIME (owner 2026-10-21: "A uniform rule
# would be nicer if it works"; NOTES, "A uniform rule for walled-in homes: destinations in their
# pockets, no free first"): RUNTIME with the leak on every cell open at baseline, the sealed
# pockets among them (`Lp`: gap_exits.OpenSites), and each walled-in home's demand on its own
# pocket's open ground (`+pockets`: gap_clear.PocketDemand); its optimizers run without free first
# (relax's `.f0`)
UNIFORM_EXITS = dataclasses.replace(EXITS, leak=dataclasses.replace(LEAK, sites=gx.OpenSites()))


def _uniform() -> str:
    runtime = gap_clear.complex_of(RUNTIME, factorization=gap_factor.SuperLU())
    return dataclasses.replace(runtime, placement=gap_clear.PocketDemand(),
                               exits=UNIFORM_EXITS).name


UNIFORM = _uniform()


# Every home has demand under the uniform rule (NOTES, "Walled-in homes, round 2: T, U and W, each
# with its best optimizer": F1, the owner's "self-sufficient moonbase"): UNIFORM with each home
# that has no open ground within ring_m placed on its own ground as B (i) reads it, its building's
# destination there (`+homes`: gap_clear.HomeDemand), so its escape time at baseline is its own
# building's and a clearing that opens ground within ring_m of its outline joins it
def _homes() -> str:
    uniform = gap_clear.complex_of(UNIFORM, factorization=gap_factor.SuperLU())
    return dataclasses.replace(uniform, placement=gap_clear.HomeDemand()).name


HOMES = _homes()


def walls_on_uniform(part: str, *, device: str) -> gap_walls.WallMetric:
    """Walls that leak (gap_walls; owner 2026-10-21, NOTES "Walls that leak: every home's source
    inside it, each wall priced as a length"): UNIFORM's mesh, leaks (on every cell open at
    baseline) and doors (where its homes' buildings' destinations leak) with the walls `part`
    names (`Wk<k_in>l<ell>`, `Wk<k_in>o<ell_out>i<ell_in>n<passes>`), on `device`; read back from
    its name (gap_walls.walls_of)."""
    metric = gap_walls.WallMetric(base=metric_of(UNIFORM, device=device),
                                  walls=gap_walls.walls_part_of(f"+{part}"))
    return gap_walls.walls_of(metric.name, factorization=gap_factor.factorization_of(device))


def metric_of(token: str, *, device: str) -> gap_clear.ComplexMetric:
    """The complex metric `token` names (gap_clear.complex_of: refused unless it is one of this
    code's), its factorization on `device`: SuperLU on a core, cuDSS on a GPU."""
    return gap_clear.complex_of(token, factorization=gap_factor.factorization_of(device))


def spec_of(s: float) -> gc.Spec:
    """The runtime's Spec at the outline spacing `s`: its ring and band as the per-block metric's
    name (RUNTIME's complex) spells them (the guards' complexes, and the rebuild's,
    gap_compare.ComplexModel)."""
    runtime = gap_clear.complex_of(PER_BLOCK, factorization=gap_factor.SuperLU())
    return dataclasses.replace(runtime.spec, s=s)
