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
LEAK = gx.InternalLeak(open_m=6214.0, foot_m=562.0)
BOOST = gx.StreetBoost(factor=10.0, surfaces=gx.PAVED, classes=DRIVE_CLASSES)
EXITS = gx.Exits(barriers=BARRIERS, edge=gx.OpenEdge(), leak=LEAK, boost=BOOST)
RUNTIME = PER_BLOCK + EXITS.suffix


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
