"""The site's method list must cover what the examples actually run.

Third instance of one failure mode in this codebase: a hand-maintained list that silently falls
behind the config. The derivation cache key missed every method added after it was written, so a
regeneration republished stale results; FRIENDLY_METHOD_NAMES missed two, so a published legend
mixed friendly labels with bare config keys; and this list described `greedy_arterial_buildable`
and `dream_come_true` while omitting `cycle_native`, which is in every example lineup.

So this reads the lineups out of `conf/example/*.yaml` rather than naming methods itself.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example_methods() -> set[str]:
    named: set[str] = set()
    for cfg in sorted((ROOT / "conf" / "example").glob("*.yaml")):
        m = re.search(r"^methods:\s*\[([^\]]*)\]", cfg.read_text(), flags=re.M)
        if m:
            named |= {x.strip() for x in m.group(1).split(",") if x.strip()}
    named.add("osm_footpaths")      # joins any variant with a committed OSM snapshot
    return named


def _site_methods() -> set[str]:
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    return set(re.findall(r'^    M\("([a-z_]+)"', src, flags=re.M))


def test_every_example_method_has_a_site_page() -> None:
    missing = sorted(_example_methods() - _site_methods())
    assert not missing, f"methods benchmarked in examples/ but absent from the site: {missing}"


def test_every_published_site_method_is_in_the_mkdocs_nav() -> None:
    """A generated page that is not in `nav:` is built but unreachable.

    mkdocs.yml's own comment concedes this list is manual and "adding or retiring a method therefore
    still means a line here" -- MkDocs cannot auto-populate one section of an explicit nav without
    another pinned dependency. So it is guarded rather than automated.
    """
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    # published=False pages are deliberately excluded from the built site (see exclude_docs)
    unpublished = set(re.findall(r'M\("([a-z_]+)"[^)]*?published=False', src, flags=re.S))
    nav = (ROOT / "mkdocs.yml").read_text()
    listed = set(re.findall(r"methods/([a-z_]+)\.md", nav))
    missing = sorted(_site_methods() - unpublished - listed)
    assert not missing, f"site pages generated but absent from mkdocs.yml nav: {missing}"
