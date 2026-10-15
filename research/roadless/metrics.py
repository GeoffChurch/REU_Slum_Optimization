"""A run's metric, resolved once where its configuration is read (seam.Metric; NOTES, "The complex
as the runtime: one fixed mesh"): the raster's (relax.RasterMetric: a conductance, `along`, on a
grid, on a device) or the gap complex's (gap_clear.ComplexMetric: its name carries every constant
that can change a score, the code's among them, and no raster conductance's name reads as it; its
factorization the run's device's, gap_factor.factorization_of). Both are live until phase 2
measures which dominates.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/metrics.py
"""
from __future__ import annotations

import gap_clear
import gap_factor
import lifted
import relax
import seam

COMPLEX = "cx+"           # every complex metric's name starts so (gap_clear.ComplexMetric.name)
# The runtime's metric (owner 2026-10-14: the shy distance at 0.5 m on the gap complex, its Spec and
# FlowSpec as phase 1 fixed them), this code's constants' digest ending its name
RUNTIME = f"cx+shy0.5+s0.125r1b1+R0.5g4x2q4+v{gap_clear.CODE}"


def metric_of(token: str, *, device: str, mesh: lifted.MeshSpec) -> seam.Metric:
    """`token` a complex metric's name (its factorization on `device`: SuperLU on a core, cuDSS on
    a GPU; no raster grid: `mesh` the uniform h 0.5 a plan leaves unnamed) or a raster conductance
    (relax.RasterMetric on `device` and `mesh`)."""
    if token.startswith(COMPLEX):
        if mesh.suffix:
            raise ValueError(f"{token}: a raster grid ({mesh.name}) named for the complex")
        return gap_clear.complex_of(token, factorization=gap_factor.factorization_of(device))
    return relax.RasterMetric(along=token, device=device, mesh=mesh)
