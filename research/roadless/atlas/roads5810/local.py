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
from typing import Any

PAGE = Path(__file__).resolve().parent
PLATES = PAGE / "plates"           # render.PLATES (this script runs without render's imports)
FLOWS = ("current", "power", "current_gain", "power_gain", "time")
VIEWS = {"DATA": ("network",) + FLOWS, "CDATA": ("clearing",) + FLOWS}   # the modes' frames
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}'
        '[hidden]{display:none!important}</style>\n</head>\n<body>\n')


def js(html: str, var: str) -> Any:
    """The page's `var <var> = <JSON>;` (its tables are JSON: DATA, CDATA, NETS, CLASS, DOT, KEY,
    POINTS, CCLASS, ...)."""
    at = html.index(f"var {var} = ") + len(f"var {var} = ")
    value, end = json.JSONDecoder().raw_decode(html, at)
    if html[end] != ";":
        raise ValueError(f"var {var}: not followed by ';'")
    return value


def with_js(html: str, var: str, value: Any, *, lines: bool) -> str:
    """`html` with `var <var>`'s JSON replaced by `value` (one entry a line if `lines`)."""
    at = html.index(f"var {var} = ") + len(f"var {var} = ")
    _, end = json.JSONDecoder().raw_decode(html, at)
    if not lines:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif isinstance(value, dict):
        text = "{\n" + ",\n".join(f"    {json.dumps(k, ensure_ascii=False)}: "
                                  f"{json.dumps(v, ensure_ascii=False)}"
                                  for k, v in value.items()) + "\n  }"
    else:
        text = "[\n" + ",\n".join(f"    {json.dumps(v, ensure_ascii=False)}"
                                  for v in value) + "\n  ]"
    return html[:at] + text + html[end:]


def frames(rows: list[Any]) -> list[str]:
    """A method's distinct frames in slider order (a per-budget method's rows are null below
    its first answer)."""
    return list(dict.fromkeys(r["f"] for r in rows if r is not None))


def main() -> None:
    html = (PAGE / "index.html").read_text()
    names = {f"{m}_{f}_{v}.webp"
             for var, views in VIEWS.items()
             for m, rows in js(html, var).items()
             for f in frames(rows) for v in views}
    names |= set(re.findall(r'"([a-z_0-9.]+\.(?:webp|png))"', html))
    missing = sorted(n for n in names if not (PLATES / n).is_file())
    if missing:
        raise SystemExit(f"{len(missing)} plates missing from {PLATES}: {missing[:5]} ...")
    (PLATES / "index.html").write_text(HEAD + html + "\n</body>\n</html>\n")
    print(f"{len(names)} plates present; open file://{PLATES / 'index.html'}")


if __name__ == "__main__":
    main()
