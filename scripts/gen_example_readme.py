"""Machine-generated README for a metric example variant. PURE dir-reader: reads the run outputs
already on disk (meta.json of structured stats, the frontier curve PNGs, figure files) and returns
the markdown. Each section is emitted only if its artifacts are present, so the numbers can never
drift from the data and a partial run yields a partial (never-erroring) README."""
from __future__ import annotations

import json
from pathlib import Path


def _n(x: float) -> str:
    return f"{x:,.0f}"


def gen_example_readme(run_dir: Path, *, metric_name: str, formula: str, blurb: str) -> str:
    parts: list[str] = []
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    parts.append(f"# Multiblock, screened by `{metric_name}`\n")
    parts.append(f"*{blurb}*\n")
    parts.append(f"**Metric:** `{formula}` — one metric drives the screen, region growth, and "
                 f"colouring end to end.\n")
    flagged, total_blocks = meta.get("flagged"), meta.get("total_blocks")
    top_block, top_depth = meta.get("deepest_block"), meta.get("deepest_depth")
    if (flagged is not None and total_blocks is not None
            and top_block is not None and top_depth is not None):
        parts.append("## 1. Screen the metro\n")
        parts.append(f"`{metric_name}` flagged "
                     f"**{_n(flagged)} of {_n(total_blocks)}** blocks. "
                     f"Top-scoring: `{top_block}` (peel depth {top_depth:.0f}).\n")
        if (run_dir / "screen.jpg").exists():
            parts.append("![screen](screen.jpg)\n")
        maps_url = meta.get("maps_url")
        if maps_url:
            parts.append(f"**Location:** [see the grown region on Google Maps]({maps_url}).\n")
    region_members, region_parcels = meta.get("region_members"), meta.get("region_parcels")
    region_mean_depth = meta.get("region_mean_depth")
    region_mean_density = meta.get("region_mean_density_per_ha")
    if (region_members is not None and region_parcels is not None
            and region_mean_depth is not None and region_mean_density is not None):
        parts.append("## 2. Grow the region\n")
        parts.append(f"The metric grows a **{region_members}-block** region "
                     f"(**{_n(region_parcels)} parcels**), mean depth "
                     f"{region_mean_depth:.1f} rings, mean density "
                     f"{region_mean_density:.0f} bldg/ha.\n")
        if (run_dir / "region.jpg").exists():
            parts.append("![region](region.jpg)\n")
    curve_ext = sorted(run_dir.glob("curve_external_connectivity_*.png"))
    curve_int = sorted(run_dir.glob("curve_internal_connectivity_*.png"))
    curve_disp = sorted(run_dir.glob("displacement_*.png"))
    if curve_ext or curve_int or curve_disp:
        parts.append("## 3. The method frontier (benefit vs added road)\n")
        parts.append("Each method's benefit as cumulative added road grows — the full trade-off "
                     "whose fixed-depth and matched-budget slices are tabulated in "
                     "`lens_a_depth.csv` and `lens_b_matched.csv` (this dir). External "
                     "connectivity (access burden removed), internal connectivity (backup-route "
                     "redundancy), and displacement (a rising cost):\n")
        for caption, pngs in (("external connectivity", curve_ext),
                              ("internal connectivity", curve_int),
                              ("displacement", curve_disp)):
            for p in pngs:
                parts.append(f"![{caption}]({p.name})\n")
    return "\n".join(parts) + "\n"


def write_readme(run_dir: Path, out_dir: Path, *,
                 metric_name: str, formula: str, blurb: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = gen_example_readme(run_dir, metric_name=metric_name, formula=formula, blurb=blurb)
    path = out_dir / "README.md"
    path.write_text(md)
    return path
