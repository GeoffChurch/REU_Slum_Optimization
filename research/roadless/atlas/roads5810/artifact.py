"""The "5810 Build-out" page as a claude.ai artifact: plates/artifact/, index.html and its files.
An artifact version holds at most 511 files and 256 MB, and the page names ~1900 plates (240 MB),
so each method's frames of one view become one sprite sheet (s/<its chart class>_<view>.webp: its
distinct frames in slider order, stacked, at WIDTH px), which the page crops to the frame the
slider picks (SHEET, SID); the baselines and the gallery are scaled the same, the colour bars
copied. Refuses on a missing plate (local.py's check). Rerun after editing index.html, then
publish plates/artifact/index.html with s/, base/ and the bars as its files: one publish sends at
most 255 files and 64 MB, so in batches (BATCH_MB, BATCH_FILES), written to
plates/artifact_publish.json (outside the artifact): the first publish the page with the first
batch as its `files`, each later one the same page to the same url with the next batch (files
left out are kept). https://claude.ai/artifact/UnDkhPymCPcmxWXbHASys3 (atlas/__init__.py).

    PYTHONPATH=. uv run python research/roadless/atlas/roads5810/artifact.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local  # noqa: E402

OUT = local.PLATES / "artifact"
WIDTH = 800           # px: a plate's width on the artifact (the page shows two ~480 px columns)
QUALITY = 80          # lossy WebP's
CLASSES = {"DATA": "CLASS", "CDATA": "CCLASS"}     # a mode's frames -> its chart classes
BATCH_MB = 60.0       # a publish's files at most (the limit 64 MB, the page and headroom beside)
BATCH_FILES = 250     # a publish's files at most (the limit 255)
LIMIT_FILES, LIMIT_MB = 511, 256.0     # an artifact version's


def scaled(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)


def patch(html: str, old: str, new: str) -> str:
    if html.count(old) != 1:
        raise SystemExit(f"index.html changed: {old[:60]!r} found {html.count(old)} times")
    return html.replace(old, new)


def main() -> None:
    html = (local.PAGE / "index.html").read_text()
    local.main()                                   # every plate present, or it refuses
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "s").mkdir(parents=True)
    (OUT / "base").mkdir()
    sheet: dict[str, list[str]] = {}
    sid: dict[str, str] = {}
    for var, views in local.VIEWS.items():
        cls = local.js(html, CLASSES[var])
        for m, rows in local.js(html, var).items():
            fs = local.frames(rows)
            sheet[m], sid[m] = fs, cls[m]
            for v in views:
                frames = [scaled(local.PLATES / f"{m}_{f}_{v}.webp") for f in fs]
                out = Image.new("RGB", (WIDTH, sum(f.height for f in frames)), "white")
                y = 0
                for f in frames:
                    out.paste(f, (0, y))
                    y += f.height
                out.save(OUT / "s" / f"{cls[m]}_{v}.webp", quality=QUALITY, method=6)
    if len(set(sid.values())) != len(sid):
        raise SystemExit("two methods share a chart class")
    for side in ("a", "b"):                        # the two plates: crops of a sheet
        html, n = re.subn(rf'<div class="plate"><img id="img-{side}" src="[^"]*" width="1200" '
                          r'height="1086"', f'<div class="plate crop"><img id="img-{side}" '
                          'src="baseline_network.webp"', html)
        if n != 1:
            raise SystemExit(f"index.html changed: plate {side} not found")
    singles = set(re.findall(r'"([a-z_0-9.]+\.(?:webp|png))"', html))
    for name in sorted(singles):
        if name.endswith(".png"):                  # the colour bars, as drawn
            shutil.copy(local.PLATES / name, OUT / name)
        else:
            scaled(local.PLATES / name).save(OUT / "base" / name, quality=QUALITY, method=6)
            html = html.replace(f'"{name}"', f'"base/{name}"')
    h = round(1086 * WIDTH / 1200)
    html = html.replace('width="1200" height="1086"', f'width="{WIDTH}" height="{h}"')
    html = patch(html, ".plate { ", f".plate.crop {{ aspect-ratio: {WIDTH} / {h}; }}\n"
                 ".plate.crop img { display: block; width: 100%; height: auto; max-width: none; }\n"
                 ".plate { ")
    html = patch(html, "  var BASE = {",
                 f"  var SHEET = {json.dumps(sheet, ensure_ascii=False)};\n"
                 f"  var SID = {json.dumps(sid, ensure_ascii=False)};\n  var BASE = {{")
    html = patch(html, '''    if (k === 0) return BASE[view] || BASE.power;
    return method + "_" + M().data[method][k - 1].f + "_" + view + ".webp";''',
                 '''    if (k === 0) return { file: BASE[view] || BASE.power, i: 0, n: 1 };
    var fs = SHEET[method], f = M().data[method][k - 1].f;
    return { file: "s/" + SID[method] + "_" + view + ".webp", i: fs.indexOf(f), n: fs.length };''')
    html = patch(html, "    img.src = src(method, kk, view);",
                 '    var s = src(method, kk, view);\n    img.src = s.file;\n'
                 '    img.style.transform = "translateY(" + (-100 * s.i / s.n) + "%)";')
    (OUT / "index.html").write_text(html)
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {mb:.1f} MB in {OUT}")
    if len(files) > LIMIT_FILES or mb > LIMIT_MB:
        raise SystemExit(f"over an artifact version's {LIMIT_FILES} files and {LIMIT_MB} MB")
    batches: list[dict[str, str]] = [{}]
    size = 0.0
    for f in files:
        if f.name == "index.html" and f.parent == OUT:
            continue
        mbf = f.stat().st_size / 1e6
        if batches[-1] and (size + mbf > BATCH_MB or len(batches[-1]) >= BATCH_FILES):
            batches.append({})
            size = 0.0
        batches[-1][str(f.relative_to(OUT))] = str(f)
        size += mbf
    (local.PLATES / "artifact_publish.json").write_text(json.dumps(
        dict(page=str(OUT / "index.html"), batches=batches), indent=1))
    for i, batch in enumerate(batches, start=1):
        bmb = sum(Path(v).stat().st_size for v in batch.values()) / 1e6
        print(f"publish {i}: {len(batch)} files, {bmb:.1f} MB")


if __name__ == "__main__":
    main()
