"""Every page carries its QR-code slot (a MkDocs hook, mkdocs.yml `hooks:`).

overrides/main.html puts a slot on each page with a canonical URL -- `data-sbu-qr` holding that
URL, which web/src/qr/qr.ts encodes with the live query and fragment. After the build this audits
every HTML page: each must have exactly one slot, naming its own canonical URL (the page's
`<link rel="canonical">`), or the build fails -- so a template change, a theme upgrade that renames
the block, or a page built outside the theme cannot drop the QR code quietly. The 404 page is the
one exception, and is checked to be one: it has no canonical URL, so no slot.

Stdlib-only, so it is type-checked and tested without MkDocs (tests/test_qr_slot.py)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Its attributes as written; `hidden` follows in whichever spelling the page's HTML serializer
# leaves (glightbox re-serializes pages, writing `hidden=""`).
_SLOT = re.compile(r'<div class="sbu-qr" data-sbu-qr="([^"]*)"')
_CANONICAL = re.compile(r'<link rel="canonical" href="([^"]*)">')
# The page MkDocs renders from the theme's 404 template: no canonical URL, so no slot.
NOT_FOUND = "404.html"


def audit(site_dir: Path) -> list[str]:
    """What is wrong with the built site's QR slots."""
    pages = sorted(site_dir.rglob("*.html"))
    if not pages:
        return [f"{site_dir} has no HTML pages to audit"]
    problems: list[str] = []
    for page in pages:
        name = page.relative_to(site_dir).as_posix()
        html = page.read_text(encoding="utf-8")
        slots = _SLOT.findall(html)
        canonical = _CANONICAL.findall(html)
        if name == NOT_FOUND:
            if slots or canonical:
                problems.append(f"{name}: expected no canonical URL and no QR slot")
            continue
        if len(canonical) != 1:
            problems.append(f"{name}: {len(canonical)} canonical links, not one")
        elif slots != canonical:
            problems.append(f"{name}: QR slots {slots}, not one naming {canonical[0]}")
    return problems


def on_post_build(config: Any) -> None:
    """Fails the build (SystemExit, as hooks/cache_bust.py does) on any page without its slot."""
    problems = audit(Path(config.site_dir))
    if problems:
        raise SystemExit("QR slot: " + "; ".join(problems[:10])
                         + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
