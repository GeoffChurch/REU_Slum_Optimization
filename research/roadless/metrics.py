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
import gap_factor

COMPLEX = "cx+"           # every complex metric's name starts so (gap_clear.ComplexMetric.name)
# The runtime's metric (owner 2026-10-14: the shy distance at 0.5 m on the gap complex, its Spec and
# FlowSpec as phase 1 fixed them), this code's constants' digest ending its name
RUNTIME = f"cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v{gap_clear.CODE}"


def metric_of(token: str, *, device: str) -> gap_clear.ComplexMetric:
    """The complex metric `token` names (gap_clear.complex_of: refused unless it is one of this
    code's), its factorization on `device`: SuperLU on a core, cuDSS on a GPU."""
    return gap_clear.complex_of(token, factorization=gap_factor.factorization_of(device))


def spec_of(s: float) -> gc.Spec:
    """The runtime's Spec at the outline spacing `s`: its ring and band as RUNTIME's name spells
    them (the guards' complexes, and the rebuild's, gap_compare.ComplexModel)."""
    runtime = gap_clear.complex_of(RUNTIME, factorization=gap_factor.SuperLU())
    return dataclasses.replace(runtime.spec, s=s)
