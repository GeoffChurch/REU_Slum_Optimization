"""Machine-generated README for a metric example variant. PURE dir-reader: reads the run outputs
already on disk (meta.json of structured stats, the frontier curve PNGs, figure files) and returns
the markdown. Each section is emitted only if its artifacts are present, so the numbers can never
drift from the data and a partial run yields a partial (never-erroring) README."""
from __future__ import annotations

import json
from pathlib import Path


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _after_method(name: str) -> str:
    # after_<method>_<tag>.jpg -> <method> (tag = matched | extNN; method may contain underscores)
    return name[len("after_"):-len(".jpg")].rpartition("_")[0]


def _gif_method(name: str) -> str:
    return name[len("reblock_"):-len(".gif")]                # reblock_<method>.gif -> <method>


def _img_table(items: list[tuple[str, str]]) -> str:
    # one markdown table: each method a column (in list order, so rows align), its image in the row.
    head = "| " + " | ".join(lbl for lbl, _ in items) + " |"
    sep = "|" + "|".join(["---"] * len(items)) + "|"
    cells = " | ".join(f"![{lbl}]({fn})" for lbl, fn in items)
    return f"{head}\n{sep}\n| {cells} |\n"


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
    depth_curve = sorted(run_dir.glob("depth_vs_road_*.png"))
    curve_ext = sorted(run_dir.glob("curve_external_connectivity_*.png"))
    curve_int = sorted(run_dir.glob("curve_internal_connectivity_*.png"))
    curve_disp = sorted(run_dir.glob("displacement_*.png"))
    if depth_curve or curve_ext or curve_int or curve_disp:
        parts.append("## 3. The method frontier (benefit vs added road)\n")
        if depth_curve:
            parts.append("How far each method's road drives the region's **max access depth**, "
                         "shown **for reference** (the method budget below is now the "
                         "external-connectivity outcome) — a dot marks where it first reaches each "
                         "new integer depth. `clearance` is **continued past its depth target** (a "
                         "full-drainage run) out to the longest method's road, so every method is "
                         "compared at the same budget: the as-built `osm_footpaths` network "
                         "plateaus at its floor while `clearance` reaches the same depth for a "
                         "fraction of the road:\n")
            for p in depth_curve:
                parts.append(f"![access depth vs added road]({p.name})\n")
    if curve_ext or curve_int or curve_disp:
        parts.append("Each method's benefit as cumulative added road grows — the full trade-off "
                     "whose fixed-depth and matched-budget slices are tabulated in "
                     "`lens_a_external.csv` and `lens_b_matched.csv` (this dir). External "
                     "connectivity (access burden removed), internal connectivity (backup-route "
                     "redundancy), and displacement (a rising cost):\n")
        for caption, pngs in (("external connectivity", curve_ext),
                              ("internal connectivity", curve_int),
                              ("displacement", curve_disp)):
            for p in pngs:
                parts.append(f"![{caption}]({p.name})\n")
    gifs = sorted(run_dir.glob("reblock_*.gif"))
    matched = sorted(run_dir.glob("after_*_matched.jpg"))
    depth_imgs = sorted(run_dir.glob("after_*_ext*.jpg"))
    if gifs or matched or depth_imgs:
        parts.append("## 4. Each method on the ground\n")
        parts.append("The same region on the same access-depth colour scale (blue = at a street, "
                     "red = deep) with displaced buildings marked — so the maps are directly "
                     "comparable across methods.\n")
        if gifs:
            parts.append("**Watch each method reblock** — its full road set added in drainage "
                         "order, the deep interior draining as the network reaches in:\n")
            parts.append(_img_table([(_gif_method(p.name), p.name) for p in gifs]))
        if matched:
            parts.append("**Matched road budget** — every method truncated to the same total added "
                         "road, so this compares the access each *buys for the same cost*:\n")
            parts.append(_img_table([(_after_method(p.name), p.name) for p in matched]))
        if depth_imgs:
            parts.append("**Matched external-connectivity target** — every method truncated where "
                         "external connectivity (access-burden removed) reaches 0.70, so this "
                         "compares the *road each takes* for the same outcome:\n")
            parts.append(_img_table([(_after_method(p.name), p.name) for p in depth_imgs]))
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
