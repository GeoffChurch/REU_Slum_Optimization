"""Machine-generated README for a metric example variant. PURE dir-reader: reads the run outputs
already on disk (meta.json of structured stats, the frontier curve PNG, the two lens CSVs, and
figure files) and returns the markdown. Each section is emitted only if its artifacts are present,
so the numbers can never drift from the data and a partial run yields a partial (never-erroring)
README."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _pct(x: str) -> str:
    return f"{float(x) * 100:.1f}%"


def _after_method(name: str, lens: str, coloring: str) -> str:
    # after_<method>_<lens>_<coloring>.jpg -> <method> (method may itself contain underscores, so
    # strip the exact known prefix/suffix rather than splitting on "_").
    suffix = f"_{lens}_{coloring}.jpg"
    return name[len("after_"):-len(suffix)]


def _gif_method(name: str) -> str:
    return name[len("reblock_"):-len(".gif")]                # reblock_<method>.gif -> <method>


def _img_table(items: list[tuple[str, str]]) -> str:
    # one markdown table: each method a column (in list order, so rows align), its image in the row.
    head = "| " + " | ".join(lbl for lbl, _ in items) + " |"
    sep = "|" + "|".join(["---"] * len(items)) + "|"
    cells = " | ".join(f"![{lbl}]({fn})" for lbl, fn in items)
    return f"{head}\n{sep}\n| {cells} |\n"


def _lens_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def _lens_methods(run_dir: Path, lens: str, rows: dict[str, dict[str, str]]) -> list[str]:
    # canonical method order for a lens: the "depth"-coloring image glob if present (so the two
    # coloring image tables AND the CSV-derived table below all share one column order); falls
    # back to the CSV's own row order if no images are present (partial run).
    depth_imgs = sorted(run_dir.glob(f"after_*_{lens}_depth.jpg"))
    if depth_imgs:
        return [_after_method(p.name, lens, "depth") for p in depth_imgs]
    return list(rows)


def _lens_table(rows: dict[str, dict[str, str]], methods: list[str], *,
                flag_col: str, unmet_note: str) -> str:
    lines = ["| Method | Road | Displacement | Permeability | Note |", "|---|---|---|---|---|"]
    for m in methods:
        row = rows.get(m)
        if row is None:
            continue
        met = row[flag_col] == "True"
        note = "" if met else unmet_note
        lines.append(f"| {m} | {_n(float(row['road_m']))} m | {_pct(row['displacement'])} | "
                     f"{_pct(row['permeability'])} | {note} |")
    return "\n".join(lines) + "\n"


def _lens_images(run_dir: Path, lens: str, coloring: str,
                 methods: list[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for m in methods:
        fn = f"after_{m}_{lens}_{coloring}.jpg"
        if (run_dir / fn).exists():
            items.append((m, fn))
    return items


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
        if meta.get("maps_qr") and (run_dir / meta["maps_qr"]).exists():
            parts.append(f'\n<a href="{meta.get("maps_url","")}">'
                         f'<img src="{meta["maps_qr"]}" alt="Google Maps QR" width="120"></a>\n')
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

    # §3: the single permeability frontier -- permeability (benefit, the only benefit axis) vs
    # displacement (cost, the only cost axis), one line per method; Pareto-dominance reads straight
    # off it. Gated on the frontier PNG existing (the raw per-method samples backing every point
    # are in frontier_permeability.csv, alongside it -- not embedded, just present).
    frontier_pngs = sorted(run_dir.glob("frontier_*.png"))
    if frontier_pngs:
        parts.append("## 3. The permeability frontier (benefit vs added road)\n")
        parts.append("The frontier is the whole trade-off: **permeability** (benefit — the only "
                     "benefit axis) on the y-axis against **displacement** (cost — the only cost "
                     "axis) on the x-axis, one line per method. Pareto-dominance — which method "
                     "buys more permeability for less displacement — reads straight off it (raw "
                     "per-method samples are in `frontier_permeability.csv`, this dir):\n")
        for p in frontier_pngs:
            parts.append(f"![permeability vs displacement]({p.name})\n")
        before = [(lbl, fn) for lbl, fn in
                 (("access-depth", "before_depth.jpg"),
                  ("permeability potential", "before_perm.jpg"))
                 if (run_dir / fn).exists()]
        if before:
            parts.append("**Before any road is added**, the same region in both colorings: "
                         "access-depth (blue = at a street, red = deep interior) vs permeability "
                         "potential (dark = hard to escape, light = easy):\n")
            parts.append(_img_table(before))

    # §4: each method on the ground -- the GIF row, then the two lenses. Lens A (matched
    # displacement) truncates every method to the same home-cost and compares the permeability
    # each buys; Lens B (matched permeability) truncates every method to the same permeability
    # outcome and compares the displacement each spends. Each lens's table comes straight from its
    # CSV; its two after-image tables (access-depth coloring, permeability-potential coloring)
    # share one method order with that table.
    gifs = sorted(run_dir.glob("reblock_*.gif"))
    disp_csv, perm_csv = run_dir / "lens_displacement.csv", run_dir / "lens_permeability.csv"
    disp_rows = _lens_rows(disp_csv) if disp_csv.exists() else {}
    perm_rows = _lens_rows(perm_csv) if perm_csv.exists() else {}
    if gifs or disp_rows or perm_rows:
        parts.append("## 4. Each method on the ground\n")
        if gifs:
            parts.append("**Watch each method reblock** — its full road set added in drainage "
                         "order, the deep interior draining as the network reaches in:\n")
            parts.append(_img_table([(_gif_method(p.name), p.name) for p in gifs]))
        if disp_rows:
            methods = _lens_methods(run_dir, "disp", disp_rows)
            parts.append("### Matched displacement\n")
            parts.append("Every method truncated to the same displacement %, so this compares the "
                         "**permeability each buys for the same home-cost**:\n")
            parts.append(_lens_table(disp_rows, methods, flag_col="at_budget",
                                     unmet_note="converged below budget"))
            depth_imgs = _lens_images(run_dir, "disp", "depth", methods)
            if depth_imgs:
                parts.append("Access-depth coloring:\n")
                parts.append(_img_table(depth_imgs))
            perm_imgs = _lens_images(run_dir, "disp", "perm", methods)
            if perm_imgs:
                parts.append("Permeability-potential coloring:\n")
                parts.append(_img_table(perm_imgs))
        if perm_rows:
            methods = _lens_methods(run_dir, "perm", perm_rows)
            parts.append("### Matched permeability\n")
            parts.append("Every method truncated where permeability first reaches the standard "
                         "target, so this compares the **displacement each spends** for the same "
                         "permeability outcome:\n")
            parts.append(_lens_table(perm_rows, methods, flag_col="reached",
                                     unmet_note="unreached"))
            depth_imgs = _lens_images(run_dir, "perm", "depth", methods)
            if depth_imgs:
                parts.append("Access-depth coloring:\n")
                parts.append(_img_table(depth_imgs))
            perm_imgs = _lens_images(run_dir, "perm", "perm", methods)
            if perm_imgs:
                parts.append("Permeability-potential coloring:\n")
                parts.append(_img_table(perm_imgs))

    cmd = meta.get("command")
    if cmd:
        log_link = "\nThe full run log is in [`run.log`](run.log)." if (run_dir / "run.log").exists() else ""
        parts.append("\n## How this was generated\n\n"
                     "This example is machine-generated — one self-logging command emits the data, "
                     "maps, curves, and this README:\n\n"
                     f"```bash\n{cmd}\n```{log_link}\n")
    return "\n".join(parts) + "\n"


def write_readme(run_dir: Path, out_dir: Path, *,
                 metric_name: str, formula: str, blurb: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = gen_example_readme(run_dir, metric_name=metric_name, formula=formula, blurb=blurb)
    path = out_dir / "README.md"
    path.write_text(md)
    return path
