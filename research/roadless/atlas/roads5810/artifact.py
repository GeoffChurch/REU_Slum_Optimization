"""The "5810 Build-out" page as a claude.ai artifact: plates/artifact/, index.html and its files.
An artifact version holds at most 511 files and 256 MB, and the page names ~1900 plates (240 MB),
so each method's frames of one view become one sprite sheet (s/<its chart class>_<view>.webp: its
distinct frames in slider order, stacked, at WIDTH px), which the page crops to the frame the
slider picks (SHEET, SID); the baselines and the gallery are scaled the same, the colour bars
copied. Refuses on a missing plate (local.py's check). Rerun after editing index.html, then
publish plates/artifact/index.html with s/, base/ and the bars as its files.

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


def scaled(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)


def classes(html: str, var: str) -> dict[str, str]:
    """A JS object literal of method -> chart class (keys bare or quoted)."""
    body = re.search(rf"var {var} = \{{(.*?)\}};\n", html)[1]  # type: ignore[index]
    return {k.strip('"'): v for k, v in re.findall(r'("[^"]+"|[\w.]+): "(\w+)"', body)}


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
        cls = classes(html, CLASSES[var])
        for m, rows in json.loads(re.search(rf"var {var} = (\{{.*?\}});\n", html)[1]).items():  # type: ignore[index]
            fs = list(dict.fromkeys(r["f"] for r in rows))
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
    html = patch(html, "    img.src = src(method, k, view);",
                 '    var s = src(method, k, view);\n    img.src = s.file;\n'
                 '    img.style.transform = "translateY(" + (-100 * s.i / s.n) + "%)";')
    (OUT / "index.html").write_text(html)
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {mb:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
