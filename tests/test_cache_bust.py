"""hooks/cache_bust.py: the site's own scripts and stylesheets are loaded as `<path>?v=<hash>` on
every page, and a built site that is not is refused. Stdlib only, like the hook's logic: the
Python test environment has no MkDocs, so the hook's MkDocs-facing functions are exercised by the
deploy's `mkdocs build --strict` and these test what they call."""
from __future__ import annotations

import importlib.util
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
