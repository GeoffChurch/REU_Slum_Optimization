"""Machine-generated README for a metric example variant. PURE dir-reader: reads the run outputs
already on disk (meta.json of structured stats, the two-lens lens_*.csv, frontier CSVs, figure
files) and returns the markdown. Each section is emitted only if its artifacts are present, so the
numbers can never drift from the data and a partial run yields a partial (never-erroring) README."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def gen_example_readme(run_dir: Path, *, metric_name: str, formula: str, blurb: str) -> str:
    parts: list[str] = []
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    parts.append(f"# Multiblock, screened by `{metric_name}`\n")
    parts.append(f"*{blurb}*\n")
    parts.append(f"**Metric:** `{formula}` — one metric drives the screen, region growth, and "
                 f"colouring end to end.\n")
    if meta:
        parts.append("## 1. Screen the metro\n")
        parts.append(f"`{metric_name}` flagged "
                     f"**{_n(meta['flagged'])} of {_n(meta['total_blocks'])}** blocks. "
                     f"Deepest: `{meta['deepest_block']}` at "
                     f"{meta['deepest_depth']:.0f} rings.\n")
        if (run_dir / "screen.jpg").exists():
            parts.append("![screen](screen.jpg)\n")
        parts.append("## 2. Grow the region\n")
        parts.append(f"The metric grows a **{meta['region_members']}-block** region "
                     f"(**{_n(meta['region_parcels'])} parcels**), mean depth "
                     f"{meta['region_mean_depth']:.1f} rings, mean density "
                     f"{meta['region_mean_density_per_ha']:.0f} bldg/ha.\n")
        if (run_dir / "region.jpg").exists():
            parts.append("![region](region.jpg)\n")
    lens_a, lens_b = run_dir / "lens_a_depth.csv", run_dir / "lens_b_matched.csv"
    if lens_a.exists() and lens_b.exists():
        parts.append("## 3. Compare the methods (two lenses)\n")
        parts.append("**Lens A — every parcel to the depth target:**\n")
        parts.append(_md_table(_read_csv(lens_a)))
        parts.append("\n**Lens B — matched road budget:**\n")
        parts.append(_md_table(_read_csv(lens_b)))
    return "\n".join(parts) + "\n"


def _md_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(r[c] for c in cols) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n"


def write_readme(run_dir: Path, out_dir: Path, *,
                 metric_name: str, formula: str, blurb: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = gen_example_readme(run_dir, metric_name=metric_name, formula=formula, blurb=blurb)
    path = out_dir / "README.md"
    path.write_text(md)
    return path
