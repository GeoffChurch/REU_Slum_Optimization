"""hooks/cache_bust.py: the site's own scripts and stylesheets are loaded as `<path>?v=<hash>` on
every page, and a built site that is not is refused. Stdlib only, like the hook's logic: the
Python test environment has no MkDocs, so the hook's MkDocs-facing functions are exercised by the
deploy's `mkdocs build --strict` and these test what they call."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cache_bust", ROOT / "hooks" / "cache_bust.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _hook()
BASE = "/REU_Slum_Optimization/"
# A page as Material writes one, a directory down, and the 404 page with absolute paths.
PAGE = ('<link rel="stylesheet" href="../stylesheets/sbu.css">'
        '<link rel="stylesheet" href="../assets/stylesheets/main.ec1eaa64.min.css">'
        '<script src="../js/widgets.js"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
        '<a href="../stylesheets/">not an entry</a>')
NOT_FOUND = ('<link rel="stylesheet" href="/REU_Slum_Optimization/stylesheets/sbu.css">'
             '<script src="/REU_Slum_Optimization/js/widgets.js"></script>')


def _docs(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    docs = tmp_path / "docs"
    (docs / "stylesheets").mkdir(parents=True)
    (docs / "js").mkdir()
    (docs / "stylesheets" / "sbu.css").write_text("body {}", encoding="utf-8")
    (docs / "js" / "widgets.js").write_text("void 0;", encoding="utf-8")
    vers = cb.versions(docs, ["stylesheets/sbu.css", "js/widgets.js",
                              "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"])
    return docs, vers


def test_the_site_registers_the_hook_and_versions_its_own_entries() -> None:
    """The hook only works while mkdocs.yml runs it, and only on the entries mkdocs.yml lists."""
    yml = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^hooks:\n  - hooks/cache_bust\.py$", yml, flags=re.M)
    for entry in ("stylesheets/sbu.css", "stylesheets/fonts.css", "js/widgets.js",
                  "javascripts/mathjax.js"):
        assert f"  - {entry}" in yml, entry


def test_a_content_change_is_a_new_url_and_urls_are_left_alone(tmp_path: Path) -> None:
    docs, vers = _docs(tmp_path)
    assert set(vers) == {"stylesheets/sbu.css", "js/widgets.js"}, "a CDN URL is not ours to hash"
    out = cb.rewrite(PAGE, vers, BASE)
    assert f'href="../stylesheets/sbu.css?v={vers["stylesheets/sbu.css"]}"' in out
    assert f'src="../js/widgets.js?v={vers["js/widgets.js"]}"' in out
    assert 'main.ec1eaa64.min.css"' in out and 'tex-mml-chtml.js"' in out
    assert 'href="../stylesheets/"' in out
    (docs / "js" / "widgets.js").write_text("void 1;", encoding="utf-8")
    again = cb.versions(docs, ["js/widgets.js"])
    assert again["js/widgets.js"] != vers["js/widgets.js"]


def test_the_404_pages_absolute_paths_are_versioned(tmp_path: Path) -> None:
    _, vers = _docs(tmp_path)
    out = cb.rewrite(NOT_FOUND, vers, BASE)
    assert f'src="/REU_Slum_Optimization/js/widgets.js?v={vers["js/widgets.js"]}"' in out


def test_an_entry_with_no_file_is_refused(tmp_path: Path) -> None:
    docs, _ = _docs(tmp_path)
    with pytest.raises(FileNotFoundError, match="js/mathjax.js"):
        cb.versions(docs, ["js/mathjax.js"])


def test_the_audit_passes_a_versioned_site_and_names_every_fault(tmp_path: Path) -> None:
    _, vers = _docs(tmp_path)
    site = tmp_path / "site"
    (site / "demo").mkdir(parents=True)
    (site / "demo" / "index.html").write_text(cb.rewrite(PAGE, vers, BASE), encoding="utf-8")
    (site / "404.html").write_text(cb.rewrite(NOT_FOUND, vers, BASE), encoding="utf-8")
    assert cb.audit(site, vers, BASE) == []
    # An unversioned page, as a page the hook never saw would be.
    (site / "404.html").write_text(NOT_FOUND, encoding="utf-8")
    problems = cb.audit(site, vers, BASE)
    assert any("404.html: /REU_Slum_Optimization/js/widgets.js is not" in p for p in problems)
    assert any("404.html: never loads js/widgets.js" in p for p in problems)
    # A stale hash: last build's URL in this build's site.
    stale = cb.rewrite(PAGE, {**vers, "js/widgets.js": "0000000000"}, BASE)
    (site / "demo" / "index.html").write_text(stale, encoding="utf-8")
    assert any("demo/index.html: ../js/widgets.js?v=0000000000 is not" in p
               for p in cb.audit(site, vers, BASE))
    # A page that loads an entry not at all.
    (site / "demo" / "index.html").write_text(
        cb.rewrite(PAGE.replace('<script src="../js/widgets.js"></script>', ""), vers, BASE),
        encoding="utf-8")
    assert any("demo/index.html: never loads js/widgets.js" in p
               for p in cb.audit(site, vers, BASE))


def _data_site(tmp_path: Path) -> Path:
    """A built site as the Demo's: a page whose build-out widget fetches a bundle, which names the
    sprite sheet beside it, and a figure image with its lightbox link."""
    site = tmp_path / "site"
    (site / "assets" / "demo").mkdir(parents=True)
    (site / "demo").mkdir()
    (site / "assets" / "demo" / "sheet.webp").write_bytes(b"pixels")
    (site / "assets" / "demo" / "plate.png").write_bytes(b"png")
    (site / "assets" / "demo" / "buildout.json").write_text(
        json.dumps({"series": [{"sheet": {"file": "sheet.webp"}, "label": "Our method"}]}),
        encoding="utf-8")
    (site / "assets" / "demo" / "index.json").write_text(
        json.dumps({"buildout": "buildout.json", "note": "a sentence, not a file"}),
        encoding="utf-8")
    (site / "demo" / "index.html").write_text(
        '<figure data-widget="build-out" data-bundle="../assets/demo/buildout.json">'
        '<a class="glightbox" href="../assets/demo/plate.png"><img src="../assets/demo/plate.png">'
        '</a></figure><svg><path d="M0 0h1v1h-1z"/></svg><a href="../">home</a>',
        encoding="utf-8")
    return site


def test_every_fetched_asset_is_versioned_and_bundles_version_what_they_name(
        tmp_path: Path) -> None:
    site = _data_site(tmp_path)
    cb.version_data(site, BASE)
    demo = site / "assets" / "demo"
    bundle = json.loads((demo / "buildout.json").read_text(encoding="utf-8"))
    sheet = bundle["series"][0]["sheet"]["file"]
    assert sheet == f"sheet.webp?v={cb._digest(demo / 'sheet.webp')}"
    # A bundle named by another is versioned after its own rewrite: its final bytes.
    index = json.loads((demo / "index.json").read_text(encoding="utf-8"))
    assert index["buildout"] == f"buildout.json?v={cb._digest(demo / 'buildout.json')}"
    assert index["note"] == "a sentence, not a file"
    html = (site / "demo" / "index.html").read_text(encoding="utf-8")
    assert f'data-bundle="../assets/demo/buildout.json?v={cb._digest(demo / "buildout.json")}"' \
        in html
    plate = cb._digest(demo / "plate.png")
    assert f'href="../assets/demo/plate.png?v={plate}"' in html
    assert f'src="../assets/demo/plate.png?v={plate}"' in html
    assert 'href="../"' in html, "a link to a page is not an asset"
    assert cb.audit_data(site, BASE) == []


def test_the_audit_fails_an_unversioned_bundle_or_name(tmp_path: Path) -> None:
    site = _data_site(tmp_path)
    cb.version_data(site, BASE)
    page = site / "demo" / "index.html"
    # A deliberately unversioned bundle: the page's fetch as it was before 2026-10-29.
    html = page.read_text(encoding="utf-8")
    page.write_text(re.sub(r"buildout\.json\?v=[0-9a-f]+", "buildout.json", html),
                    encoding="utf-8")
    problems = cb.audit_data(site, BASE)
    assert problems and "data-bundle=../assets/demo/buildout.json is not versioned" in problems[0]
    # A sheet the bundle names plainly, and a reference through an attribute the rewrite does not
    # know, fail too.
    cb.version_data(site, BASE)
    demo = site / "assets" / "demo"
    (demo / "buildout.json").write_text(
        json.dumps({"series": [{"sheet": {"file": "sheet.webp"}}]}), encoding="utf-8")
    page.write_text(page.read_text(encoding="utf-8")
                    + '<div data-poster="../assets/demo/plate.png"></div>', encoding="utf-8")
    problems = cb.audit_data(site, BASE)
    assert any("buildout.json: sheet.webp is not versioned" in p for p in problems)
    assert any("data-poster=../assets/demo/plate.png" in p for p in problems)
