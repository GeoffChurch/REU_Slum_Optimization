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


def _partials() -> dict[str, str]:
    """Every committed partial. NOT tolerant of a missing directory: Path.glob() on one that does
    not exist yields nothing rather than raising, which would make every test below pass while
    checking nothing."""
    d = ROOT / "docs" / "_partials"
    assert d.is_dir(), f"{d} does not exist; the partials tests would be vacuous"
    out = {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))}
    assert out, f"{d} holds no partials; the partials tests would be vacuous"
    return out


def _producers() -> set[str]:
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    found = set(re.findall(r'^    "([A-Z]+)": ', src, flags=re.M))
    assert found, "no MARKERS entries found; the marker tests would be vacuous"
    return found


def _markers_used() -> set[str]:
    used: set[str] = set()
    for text in _partials().values():
        used |= set(re.findall(r"<!-- ([A-Z]+) -->", text))
    return used


def test_every_marker_in_a_partial_has_a_producer() -> None:
    """A marker with no producer survives substitution and ships as a literal HTML comment."""
    orphans = sorted(_markers_used() - _producers())
    assert not orphans, f"markers used in partials with no producer: {orphans}"


def test_every_producer_is_used_by_a_partial() -> None:
    """A producer nothing references is dead code that silently stops being rendered."""
    unused = sorted(_producers() - _markers_used())
    assert not unused, f"producers defined but referenced by no partial: {unused}"


def test_published_method_count_is_generated_not_typed() -> None:
    """Defect 4: '_intro.md' said "Seven" while ten methods were published."""
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    # Scoped to the M(...) call itself (same pattern as the sibling test above), not a bare
    # substring search: an unscoped 'published=False' search also matches the phrase anywhere it
    # is used in prose -- e.g. a comment explaining what the kwarg does -- and silently miscounts.
    published = len(re.findall(r'^    M\("[a-z_]+"', src, flags=re.M)) - len(
        re.findall(r'M\("[a-z_]+"[^)]*?published=False', src, flags=re.S))
    assert published == 10, f"expected 10 published methods, registry says {published}"
    intro = (ROOT / "docs" / "_partials" / "intro.md").read_text(encoding="utf-8")
    assert "<!-- METHODCOUNT -->" in intro
    for word in ("Seven", "seven", "Eight", "Nine", "Ten", "Eleven"):
        assert f"{word} road-generation" not in intro, (
            f"'{word}' typed into prose; the count must come from METHODCOUNT")


def test_no_partial_links_to_a_retired_path() -> None:
    """methodology.md and benchmark.md are gone; a link to either 404s. mkdocs --strict catches
    this at build time, but only in CI -- this fails in the unit suite.

    "](methods/index.md)" is retired everywhere EXCEPT the four partials that already live under
    docs/methodology/ (methodology.md, screening.md, permeability.md, displacement.md): from any
    of those, it is a CORRECT relative link to the current methodology/methods/ directory -- e.g.
    methodology.md's own live "[reblocker](methods/index.md)", which resolves to
    methodology/methods/index.md and is not retired at all. From every OTHER partial -- both the
    two at the docs/ ROOT (intro.md -> docs/index.md, reproduce.md -> docs/reproduce.md) AND the
    two nested under docs/results/ (bakeoff.md, nairobi.md) -- the identical string IS genuinely
    retired: it would resolve to docs/methods/index.md or docs/results/methods/index.md, neither
    of which has ever existed. (An earlier version of this test scoped the exemption the other way
    -- "only check the two root partials" -- which silently stopped checking the two results/
    partials entirely: a "not root" partial was treated as safe whether it was nested under
    docs/methodology/, where the string really is safe, or under docs/results/, where it is not.
    RULING F18 caught this: injecting "](methods/index.md)" into bakeoff.md against that version
    of the guard returned zero offenders.) Allowlisting the four partials where the string is
    actually correct, and checking it everywhere else, leaves no partial silently exempt by
    omission.
    """
    retired_everywhere = ("](methodology.md)", "](benchmark.md)")
    retired_at_docs_root = "](methods/index.md)"
    methods_index_correct_from = ("methodology.md", "screening.md", "permeability.md",
                                   "displacement.md")

    offenders: list[str] = []
    for name, text in _partials().items():
        paths = retired_everywhere + (() if name in methods_index_correct_from
                                       else (retired_at_docs_root,))
        for path in paths:
            if path in text:
                offenders.append(f"{name}: {path}")
    assert not offenders, f"links to retired paths: {offenders}"


def test_unpublished_methods_are_excluded_from_the_build() -> None:
    """published=False keeps a method out of the overview, but exclude_docs is the actual publish
    switch. If they drift, the page is BUILT and reachable by URL while linked from nowhere.

    mkdocs.yml's own comment concedes this pair is manual and warns: the method's slug must match
    in both the M() entry in gen_site_pages.py and the exclude_docs path in mkdocs.yml, or the
    unpublished page silently reappears in the built site as an orphan.
    """
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    unpublished = set(re.findall(r'M\("([a-z_]+)"[^)]*?published=False', src, flags=re.S))
    assert unpublished, "no unpublished methods found; this guard would be vacuous"
    nav_text = (ROOT / "mkdocs.yml").read_text()
    excluded = set(re.findall(r"^\s*\S*methods/([a-z_]+)\.md\s*$", nav_text, flags=re.M))
    leaked = sorted(unpublished - excluded)
    assert not leaked, f"published=False but not in exclude_docs, so still built: {leaked}"
