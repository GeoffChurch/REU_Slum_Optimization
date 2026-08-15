"""The site's method list must cover what the examples actually run.

Third instance of one failure mode in this codebase: a hand-maintained list that silently falls
behind the config. The derivation cache key missed every method added after it was written, so a
regeneration republished stale results; FRIENDLY_METHOD_NAMES missed two, so a published legend
mixed friendly labels with bare config keys; and this list described `greedy_arterial_buildable`
and `dream_come_true` while omitting `cycle_native`, which is in every example lineup.

So this reads the lineups out of `conf/example/*.yaml` rather than naming methods itself.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

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
    # Every word _COUNT_WORDS could produce, both cases -- not a hand-picked subset. A hardcoded
    # tuple here is exactly the hand-maintained list gen_site_pages.py's own module docstring
    # opens by warning about: it silently stops covering new entries (One-Six and Twelve were
    # missing here until this was generalised).
    m = re.search(r"_COUNT_WORDS = (\{[^}]*\})", src)
    assert m, "_COUNT_WORDS literal not found in gen_site_pages.py in the expected shape"
    count_words = ast.literal_eval(m.group(1))
    words = {w for w in count_words.values()} | {w.lower() for w in count_words.values()}
    for word in sorted(words):
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


def _generator_methods_path(slug: str) -> str:
    """The generator's actual on-disk path for a method's page, relative to docs/ -- derived from
    main()'s own directory-building assignments (`methodology_dir = DOCS / "..."`, `methods_dir =
    methodology_dir / "..."`) rather than retyped as a literal here. A hardcoded second copy of
    "methodology/methods/" could drift from the generator silently: if the methods directory ever
    moves and exclude_docs is updated to match, a hardcoded expectation in THIS test would still
    fail (comparing against the old path) even though nothing is actually broken -- eroding trust
    in the guard. Deriving it means a real move and a matching exclude_docs update both pass, and
    a real move with a forgotten exclude_docs update both fail, exactly the two cases that matter.
    """
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    methodology = re.search(r'methodology_dir = DOCS / "([a-z_]+)"', src)
    methods = re.search(r'methods_dir = methodology_dir / "([a-z_]+)"', src)
    assert methodology and methods, (
        "main()'s methodology_dir/methods_dir assignments moved; update this derivation")
    return f"{methodology.group(1)}/{methods.group(1)}/{slug}.md"


def _exclude_docs_paths() -> set[str]:
    """Every literal path line in mkdocs.yml's exclude_docs block scalar (comments and the
    directory-only entries like `_partials/` stay in; callers compare against a specific path)."""
    text = (ROOT / "mkdocs.yml").read_text()
    m = re.search(r"^exclude_docs:\s*\|\n((?:[ \t]+.*\n?)*)", text, flags=re.M)
    assert m, "mkdocs.yml's exclude_docs block scalar not found in the expected shape"
    return {line.strip() for line in m.group(1).splitlines()
            if line.strip() and not line.strip().startswith("#")}


def test_unpublished_methods_are_excluded_from_the_build() -> None:
    """published=False keeps a method out of the overview, but exclude_docs is the actual publish
    switch. If they drift, the page is BUILT and reachable by URL while linked from nowhere -- with
    no build warning to catch it: an orphan page (one outside nav) logs at INFO under `mkdocs
    build --strict`, not WARNING, and the build exits 0 regardless (verified empirically).

    A slug appearing SOMEWHERE in an exclude_docs-shaped line is not enough: that does not catch
    the path itself being wrong (e.g. the pre-renest `methods/<slug>.md` instead of
    `methodology/methods/<slug>.md`), only the slug being present at all. So this asserts the
    excluded path equals the generator's actual output path for the slug, exactly.
    """
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    unpublished = set(re.findall(r'M\("([a-z_]+)"[^)]*?published=False', src, flags=re.S))
    assert unpublished, "no unpublished methods found; this guard would be vacuous"
    excluded = _exclude_docs_paths()
    leaked = sorted(slug for slug in unpublished if _generator_methods_path(slug) not in excluded)
    assert not leaked, (
        f"published=False but exclude_docs has no line matching the generator's real output "
        f"path for: {leaked}")


def test_perm_graph_figures_quote_the_artifact_not_a_typed_number() -> None:
    """The captions' numbers must come from perm_graph.json. A hand-typed figure in the partial is
    exactly the drift the site's truth pass closed -- and 'seven methods' drifted because a count
    did not look like a metric."""
    import json

    from scripts.gen_site_pages import PERMGRAPH, _perm_graph_figures

    meta = json.loads((PERMGRAPH / "perm_graph.json").read_text(encoding="utf-8"))
    html = _perm_graph_figures()

    assert "graph_current_after.png" in html
    assert f"{meta['permeability_after'] * 100:.1f}" in html
    assert meta["block_id"] in html


def test_perm_graph_figures_carry_no_fix_round_1_regression() -> None:
    """Fix round 1 found two false claims live on the page (F1, F2), a load-bearing legend stated
    nowhere (F3), a stacked layout that defeats the shared scale it advertises (F4), and four-way
    caption repetition (F5). Each assertion below is the guard for one finding, named, so a
    regression fails as that finding rather than as a vague content diff."""
    import json

    from scripts.gen_site_pages import PERMGRAPH, _perm_graph_figures

    meta = json.loads((PERMGRAPH / "perm_graph.json").read_text(encoding="utf-8"))
    html = _perm_graph_figures()

    # F1: this page has no heatmaps above the graph figures -- these four images are the only
    # ones on the page. The true, shared thing across them is one vmax.
    assert "the heatmaps above" not in html
    assert "one scale shared" in html

    # F2 (amended in the 2026-08-14 fix wave): "footpath-mesh" alone does not scope the claim away
    # from the upgraded (blue) edges -- they are footpath-mesh edges too. So no "... edge width is
    # ..." sentence may claim width for anything but the GREY edges specifically, and it must say
    # what the blue ones do instead (draw at a fixed width, not a computed one).
    for claim in re.findall(r"\b(?:[A-Z][a-z]* )?[Ee]dge width is\b[^.]*\.", html):
        assert claim.startswith("Grey edge width is"), claim
        assert "fixed width instead" in claim, claim

    # F3: the blue-edge / haloed-node legend is load-bearing and must appear -- exactly once, not
    # once per caption (which would just be F5 wearing a different hat).
    assert html.count("Blue edges are the ones a road raised") == 1

    # F4: the four figures sit inside the CSS grid that makes the shared scale comparable, not as
    # four stacked <figure> blocks with no relation to each other in the markup. Matched as
    # "<figure" followed by a space or '>' (not the literal "<figure>") because the current/after
    # panel's <figure> now carries the widget's data-* mount-point attributes directly (fix wave,
    # I4) rather than being wrapped in a plain <div> -- a bare "<figure>" substring count would
    # silently drop to 3 and this guard would never notice the fourth panel moved.
    assert len(re.findall(r"<figure[ >]", html)) == 4
    assert '<div class="sbu-figure-grid">' in html
    assert html.index('<div class="sbu-figure-grid">') < html.index("<figure")

    # F5: the block id and parcel/edge counts are stated once, in the intro, not per caption. Task
    # 6 gave the block id a second, distinct occurrence -- the `data-block` attribute on the
    # current/after figure's widget mount point, which exists for the browser widget to read, not
    # for a reader to see repeated -- so it is not the caption-repetition F5 guarded against.
    # Assert both halves separately: exactly one PROSE occurrence (the intro sentence) and exactly
    # one machine-readable occurrence (the mount point), so a regression in either still fails as
    # itself rather than being absorbed into a single loosened count.
    assert html.count(f'data-block="{meta["block_id"]}"') == 1
    prose_id_count = html.count(meta["block_id"]) - html.count(f'data-block="{meta["block_id"]}"')
    assert prose_id_count == 1
    assert html.count(str(meta["n_parcels"])) == 1
    assert html.count(str(meta["n_edges"])) == 1


def test_write_page_rewrites_data_bundle_url_for_its_depth(tmp_path: Path) -> None:
    """I6: nothing previously exercised the `data-bundle="assets/` -> `../../assets/` rewrite in
    `_write_page` -- delete that one `.replace()` line and the whole suite stayed green while the
    widget's `fetch()` 404s in production (this exact gap already happened once on this branch, per
    the fix-wave report). `data-bundle` is raw HTML inside a <figure> block (like `src=`/`href=`),
    so MkDocs never rewrites it itself; `_write_page` must, against the page's SERVED url_depth."""
    from scripts.gen_site_pages import _write_page

    out = tmp_path / "permeability.md"
    body = ('<figure data-widget="perm-graph" data-bundle="assets/perm-graph/bundle.json">\n'
            "</figure>\n")
    # url_depth=2 mirrors permeability.md's real depth (methodology/permeability.md, served at
    # <base>/methodology/permeability/).
    _write_page(out, body, depth=1, url_depth=2, title="Permeability")
    text = out.read_text(encoding="utf-8")
    assert 'data-bundle="../../assets/perm-graph/bundle.json"' in text
    assert 'data-bundle="assets/' not in text


def test_assert_widget_bundle_present_fails_the_build_on_a_missing_bundle(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """I6: nothing previously exercised `_assert_widget_bundle_present` -- a site built without
    `pixi run web` would emit a <script> tag for a file that is not there, every widget would
    silently fail to boot behind an intact-looking PNG fallback, and no test would notice. Scoped to
    a throwaway `DOCS` (monkeypatched, not the real repo tree) so this test's pass/fail does not
    depend on whether `pixi run web` happens to have been run in this checkout."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)

    # No page carries a widget mount point: must stay silent regardless of the bundle's presence.
    gsp._assert_widget_bundle_present(False)

    # A page DOES carry one, but docs/js/widgets.js does not exist -- must fail the build rather
    # than ship a silent 404.
    with pytest.raises(SystemExit, match="docs/js/widgets.js is missing"):
        gsp._assert_widget_bundle_present(True)

    # Once the bundle actually exists, the identical call must not raise.
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "widgets.js").write_text("", encoding="utf-8")
    gsp._assert_widget_bundle_present(True)
