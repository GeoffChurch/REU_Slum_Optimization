"""Cache-busting for the site's own scripts and stylesheets (a MkDocs hook, mkdocs.yml `hooks:`).

GitHub Pages serves every file with `max-age=600`, so after a deploy a browser can pair the new
HTML with a cached old `js/widgets.js` or `stylesheets/sbu.css` for up to ten minutes -- after
#81 that showed "unknown data-widget: build-out-compare" and an unstyled figure on the owner's
phone. Every local `extra_css`/`extra_javascript` entry is therefore loaded as
`<path>?v=<hash of its content>`: a changed file is a new URL, an unchanged one stays cached.

The URLs are rewritten in each built page (`on_post_page`) and static template (404.html,
`on_post_template`), so every page is covered without a template override. Then the built site
is audited (`on_post_build`): every HTML page must reference every local entry, each with the
current hash, or the build fails -- the guard that a template change, a new entry or a missed
attribute cannot quietly ship unversioned URLs again.

Stdlib-only, so it is type-checked and tested without MkDocs (tests/test_cache_bust.py).
Material's own assets need none of this: their file names already carry a hash."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# The attributes a page loads a script or stylesheet through.
_ATTR = re.compile(r'\b(src|href)="([^"]*)"')
# Long enough that two versions never collide in practice, short enough to read in a URL.
_HASH_CHARS = 10


def versions(docs_dir: Path, entries: list[str]) -> dict[str, str]:
    """Each local entry (a path under the docs dir, not a URL) and its content hash. An entry
    with no file is refused: the page would load a 404, versioned or not."""
    out: dict[str, str] = {}
    for e in entries:
        if urlsplit(e).scheme:
            continue
        path = docs_dir / e
        if not path.is_file():
            raise FileNotFoundError(f"mkdocs.yml lists {e}, but {path} does not exist")
        out[e] = hashlib.sha256(path.read_bytes()).hexdigest()[:_HASH_CHARS]
    return out


def _key(value: str, base_path: str) -> str:
    """The docs-relative path an attribute value names: relative (`../js/widgets.js`, as most
    pages write it) or absolute under the site's path (`/REU_Slum_Optimization/js/widgets.js`,
    as the 404 page does). A value with a query or a scheme names nothing here."""
    if urlsplit(value).scheme or "?" in value:
        return value
    if value.startswith("/"):
        return value[len(base_path):] if value.startswith(base_path) else value
    while value.startswith(("../", "./")):
        value = value[value.index("/") + 1:]
    return value


def rewrite(html: str, vers: dict[str, str], base_path: str) -> str:
    """`html` with every reference to a versioned entry carrying its `?v=`."""
    def one(m: re.Match[str]) -> str:
        attr, value = m.groups()
        key = _key(value, base_path)
        return f'{attr}="{value}?v={vers[key]}"' if key in vers else m.group(0)
    return _ATTR.sub(one, html)


def audit(site_dir: Path, vers: dict[str, str], base_path: str) -> list[str]:
    """What is wrong with the built site: per HTML page, any entry it does not load with the
    current hash, and any reference to an entry left unversioned or carrying another hash."""
    problems: list[str] = []
    pages = sorted(site_dir.rglob("*.html"))
    if not pages:
        return [f"{site_dir} has no HTML pages to audit"]
    for page in pages:
        html = page.read_text(encoding="utf-8")
        seen: set[str] = set()
        for _, value in _ATTR.findall(html):
            path, _, query = value.partition("?")
            key = _key(path, base_path)
            if key not in vers:
                continue
            if query != f"v={vers[key]}":
                problems.append(f"{page.relative_to(site_dir)}: {value} is not {key}?v={vers[key]}")
            else:
                seen.add(key)
        problems += [f"{page.relative_to(site_dir)}: never loads {key}?v={vers[key]}"
                     for key in sorted(set(vers) - seen)]
    return problems


def _entries(config: Any) -> list[str]:
    return [str(e) for e in [*config.extra_css, *config.extra_javascript]]


def _base_path(config: Any) -> str:
    """The path the site is served under, `/` when no site_url is set."""
    path = urlsplit(config.site_url or "").path or "/"
    return path if path.endswith("/") else path + "/"


def on_post_page(output: str, page: Any, config: Any) -> str:
    return rewrite(output, versions(Path(config.docs_dir), _entries(config)), _base_path(config))


def on_post_template(output_content: str, template_name: str, config: Any) -> str:
    """The static templates (404.html), which MkDocs renders outside `on_post_page`."""
    return rewrite(output_content, versions(Path(config.docs_dir), _entries(config)),
                   _base_path(config))


def on_post_build(config: Any) -> None:
    """Fails the build (SystemExit: MkDocs is not in the type-checked environment, so its own
    error type is not used) when the built site loads any entry unversioned or stale."""
    problems = audit(Path(config.site_dir), versions(Path(config.docs_dir), _entries(config)),
                     _base_path(config))
    if problems:
        raise SystemExit("cache-busting: " + "; ".join(problems[:10])
                          + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
