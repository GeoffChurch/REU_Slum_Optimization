"""A run's metric, resolved once where its configuration is read (seam.Metric; NOTES, "The complex
as the runtime: one fixed mesh"): the raster's (relax.RasterMetric: a conductance, `along`, on a
grid, on a device) or the gap complex's (gap_clear.ComplexMetric: its name carries every constant
that can change a score, and no raster conductance's name reads as it). Both are live until
phase 2 measures which dominates.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/metrics.py
"""
from __future__ import annotations

import gap_clear
import lifted
import relax
import seam

COMPLEX = "cx+"           # every complex metric's name starts so (gap_clear.ComplexMetric.name)


def metric_of(token: str, *, device: str, mesh: lifted.MeshSpec) -> seam.Metric:
    """`token` a complex metric's name (cpu only, and no raster grid: `mesh` the uniform h 0.5 a
    plan leaves unnamed) or a raster conductance (relax.RasterMetric on `device` and `mesh`)."""
    if token.startswith(COMPLEX):
        if device != "cpu":
            raise ValueError(f"{token}: the complex's solves are on the host (cpu), not {device}")
        if mesh.suffix:
            raise ValueError(f"{token}: a raster grid ({mesh.name}) named for the complex")
        return gap_clear.complex_of(token)
    return relax.RasterMetric(along=token, device=device, mesh=mesh)
