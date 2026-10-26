"""The "5810 Build-out" page to open straight from disk: plates/index.html, index.html with the
doctype and head the artifact host adds, beside every plate the page names (render.py's PLATES:
the frames of each network in DATA and each clearing in CDATA, the baselines, the colour bars,
the gallery). Refuses when a named plate is missing. Rerun after editing index.html.

    python3 research/roadless/atlas/roads5810/local.py    # then open the printed path
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parent
PLATES = PAGE / "plates"           # render.PLATES (this script runs without render's imports)
FLOWS = ("current", "power", "current_gain", "power_gain")
VIEWS = {"DATA": ("network",) + FLOWS, "CDATA": ("clearing",) + FLOWS}   # the modes' frames
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}'
        '[hidden]{display:none!important}</style>\n</head>\n<body>\n')


def main() -> None:
    html = (PAGE / "index.html").read_text()
    names = {f"{m}_{r['f']}_{v}.webp"
             for var, views in VIEWS.items()
             for m, rows in json.loads(re.search(rf"var {var} = (\{{.*?\}});\n", html)[1]).items()
             for r in rows for v in views}
    names |= set(re.findall(r'"([a-z_0-9.]+\.(?:webp|png))"', html))
    missing = sorted(n for n in names if not (PLATES / n).is_file())
    if missing:
        raise SystemExit(f"{len(missing)} plates missing from {PLATES}: {missing[:5]} ...")
    (PLATES / "index.html").write_text(HEAD + html + "\n</body>\n</html>\n")
    print(f"{len(names)} plates present; open file://{PLATES / 'index.html'}")


if __name__ == "__main__":
    main()
