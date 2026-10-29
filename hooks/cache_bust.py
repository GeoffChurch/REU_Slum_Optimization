"""Cache-busting for everything the site loads from itself (a MkDocs hook, mkdocs.yml `hooks:`).

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

The same goes for data. After the context deploy the Demo's face-off opened at an old default:
the new widget had fetched a cached old build-out file (`assets/demo/buildout_*.json`, a plain
URL). So, after the build (`on_post_build`, `version_data`), every file under `assets/` that a
page or a widget fetches is a versioned URL too:
- in each bundle (a JSON file under `assets/`), every string that names a file beside it -- the
  sprite sheets a build-out resolves against its own URL -- gets the `?v=` of that file, a bundle
  named by another versioned after its own rewrite;
- in each page, every `src`, `href`, `data-bundle*` and `data-wheel` that resolves to a file under
  `assets/` -- images, the lightbox's links, bundles, the wheel -- gets the `?v=` of that file as
  finally written.
Then `audit_data` checks it: ANY attribute of a page, and any string of a bundle, that names a
file under `assets/` must carry its current hash, or the build fails -- so a new kind of reference
the rewrite does not know fails rather than ships unversioned.

Stdlib-only, so it is type-checked and tested without MkDocs (tests/test_cache_bust.py)."""
from __future__ import annotations

import hashlib
import json
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


# ------------------------------------------------------------------------------- data assets

# The attributes a page fetches a file through, which `version_data` versions.
_FETCHED = re.compile(r'\b(src|href|data-bundle[a-z-]*|data-wheel)="([^"]*)"')
# Every attribute, which `audit_data` checks: one the rewrite does not know fails.
_ANY_ATTR = re.compile(r'\b([a-z][a-z-]*)="([^"]*)"')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:_HASH_CHARS]


# What an attribute value must look like to be a URL at all: no whitespace, a sane length.
_URLISH = re.compile(r"[^\s\"'<>]{1,500}")


def _asset(page: Path, value: str, site_dir: Path, base_path: str) -> Path | None:
    """The file under `site_dir`/assets that `value`, an attribute of `page`, names -- relative,
    or absolute under the site's path -- or None. Its query and fragment do not count."""
    if _URLISH.fullmatch(value) is None:
        return None                     # an attribute that is not a URL (an SVG path, prose)
    path = urlsplit(value)
    if path.scheme or path.netloc or path.path == "":
        return None
    if path.path.startswith("/"):
        if not path.path.startswith(base_path):
            return None
        target = site_dir / path.path[len(base_path):]
    else:
        target = page.parent / path.path
    target = target.resolve()
    assets = (site_dir / "assets").resolve()
    return target if target.is_file() and assets in target.parents else None


# What a bundle string must look like to be a file's name: no path, an extension.
_FILE_NAME = re.compile(r"[\w.@+-]{1,200}\.[A-Za-z0-9]{1,8}")


def _sibling(bundle: Path, name: str) -> Path | None:
    """The file beside `bundle` that `name` (a string in it, its query aside) names, or None."""
    if _FILE_NAME.fullmatch(name) is None:
        return None
    sib = bundle.parent / name
    return sib if sib.is_file() and sib != bundle else None


def _strings(data: Any, fn: Any) -> Any:
    """`data` with `fn` applied to every string in it."""
    if isinstance(data, str):
        return fn(data)
    if isinstance(data, list):
        return [_strings(v, fn) for v in data]
    if isinstance(data, dict):
        return {k: _strings(v, fn) for k, v in data.items()}
    return data


def version_data(site_dir: Path, base_path: str) -> None:
    """Every bundle's names for the files beside it, then every page's references to files under
    `assets/`, rewritten with their `?v=` (module docstring)."""
    final: dict[Path, str] = {}

    def digest(path: Path) -> str:
        """`path`'s hash once it is final: a bundle's after its own names are versioned."""
        if path not in final:
            if path.suffix == ".json":
                text = path.read_text(encoding="utf-8")
                data = json.loads(text)

                def name(v: str) -> str:
                    sib = _sibling(path, v)
                    return v if sib is None else f"{v}?v={digest(sib)}"
                new = _strings(data, name)
                if new != data:
                    path.write_text(json.dumps(new, separators=(",", ":")), encoding="utf-8")
            final[path] = _digest(path)
        return final[path]

    for bundle in sorted((site_dir / "assets").rglob("*.json")):
        digest(bundle.resolve())
    for page in sorted(site_dir.rglob("*.html")):
        html = page.read_text(encoding="utf-8")

        def one(m: re.Match[str], page: Path = page) -> str:
            attr, value = m.groups()
            target = _asset(page, value, site_dir, base_path)
            if target is None or "?" in value:
                return m.group(0)
            return f'{attr}="{value}?v={digest(target)}"'
        new = _FETCHED.sub(one, html)
        if new != html:
            page.write_text(new, encoding="utf-8")


def audit_data(site_dir: Path, base_path: str) -> list[str]:
    """Every page attribute, and every bundle string, that names a file under `assets/` without
    that file's current hash."""
    problems: list[str] = []
    for page in sorted(site_dir.rglob("*.html")):
        for attr, value in _ANY_ATTR.findall(page.read_text(encoding="utf-8")):
            target = _asset(page, value, site_dir, base_path)
            if target is not None and urlsplit(value).query != f"v={_digest(target)}":
                problems.append(f"{page.relative_to(site_dir)}: {attr}={value} is not versioned "
                                f"v={_digest(target)}")
    for bundle in sorted((site_dir / "assets").rglob("*.json")):
        data = json.loads(bundle.read_text(encoding="utf-8"))
        found: list[str] = []

        def check(v: str, bundle: Path = bundle, found: list[str] = found) -> str:
            name, _, query = v.partition("?")
            sib = _sibling(bundle, name)
            if sib is not None and query != f"v={_digest(sib)}":
                found.append(f"{bundle.relative_to(site_dir)}: {v} is not versioned "
                             f"v={_digest(sib)}")
            return v
        _strings(data, check)
        problems += found
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
    """Versions the data assets (`version_data`), then fails the build (SystemExit: MkDocs is not
    in the type-checked environment, so its own error type is not used) when the built site loads
    any entry or asset unversioned or stale."""
    site_dir, base = Path(config.site_dir), _base_path(config)
    version_data(site_dir, base)
    problems = (audit(site_dir, versions(Path(config.docs_dir), _entries(config)), base)
                + audit_data(site_dir, base))
    if problems:
        raise SystemExit("cache-busting: " + "; ".join(problems[:10])
                          + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
