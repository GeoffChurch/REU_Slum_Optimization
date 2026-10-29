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
import tempfile
import tomllib
from pathlib import Path
from typing import Any

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
    assert published == 11, f"expected 11 published methods, registry says {published}"
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

    # F5: the block id and parcel/edge counts are stated once, in the intro, not per caption. The
    # count is back to a flat 1: task 6 had added a second, machine-readable occurrence in a
    # `data-block` mount-point attribute, and that attribute is gone -- no widget ever read it, and
    # every bundle already carries `block_id`, so it was a second source for one fact.
    assert html.count(meta["block_id"]) == 1
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
    `make web` would emit a <script> tag for a file that is not there, every widget would
    silently fail to boot behind an intact-looking PNG fallback, and no test would notice. Scoped to
    a throwaway `DOCS` (monkeypatched, not the real repo tree) so this test's pass/fail does not
    depend on whether `make web` happens to have been run in this checkout."""
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


def _frontier_figure_markup() -> str:
    """The Methods index's frontier `<figure>` element, opening tag through `</figure>`."""
    from scripts.gen_site_pages import gen_methods_overview

    markup = gen_methods_overview()
    opens = [m.start() for m in re.finditer(r"<figure[ >]", markup)
             if 'data-widget="frontier"' in markup[m.start():markup.index("</figure>", m.start())]]
    assert len(opens) == 1, f"expected exactly one frontier mount point, found {len(opens)}"
    return markup[opens[0]:markup.index("</figure>", opens[0]) + len("</figure>")]


def _frontier_mount_attrs() -> dict[str, str]:
    """Every `data-*` attribute on that figure. Parsed off the real generated markup rather than
    reconstructed, so what these tests assert is what a browser would actually read."""
    return dict(re.findall(r'(data-[a-z-]+)="([^"]*)"', _frontier_figure_markup()))


def test_methods_index_carries_one_frontier_mount_point_over_its_own_png_fallback() -> None:
    """The interactive figure must not replace the static one in the MARKUP: mount.ts's error path
    says "The static image above still applies", which is a lie unless the fallback `<img>` is
    sitting right there in the same `<figure>` (the widget removes it itself, and only once it has
    drawn a chart in its place). Guards the two halves that made the predecessor widget's failures
    invisible: a mount point with no image behind it, and a bundle URL in a form `_write_page`'s
    rewriter does not recognise."""
    import json

    from scripts.gen_site_pages import EXPLORE, gen_methods_overview

    bundle = json.loads((EXPLORE / "frontier.json").read_text(encoding="utf-8"))
    markup = gen_methods_overview()

    assert markup.count('data-widget="frontier"') == 1
    # The fallback image, and the bundle URL in the exact `assets/...` form _write_page rewrites.
    assert f'<img src="assets/multiblock_depth_density/frontier_{bundle["block_id"]}.png"' in markup
    assert 'data-bundle="assets/explore/frontier.json"' in markup
    # The mount point sits on the <figure> itself, never on a wrapping <div>: `.sbu-figure-grid >
    # figure` resets margin and min-width on DIRECT children only (see _figure's docstring).
    assert '<div data-widget="frontier"' not in markup


def test_frontier_mount_point_carries_only_scalars_no_json() -> None:
    """Fix round 1: the mount point passed two HTML-escaped JSON payloads, which put literal `{`/`}`
    into a markdown raw-HTML block. python-markdown very probably stashes such a block untouched --
    and fix round 2 confirmed it by rendering the generated page through mkdocs' exact extension
    set under `/usr/bin/python3`'s python-markdown 3.5.2 (the earlier claim that nothing here could
    check this was wrong). The payloads still belong in the bundle -- an attribute that needs no
    escaping cannot be broken by an escaping change -- so this keeps them out: every attribute is a
    bare scalar, and the whole element is brace-free."""
    figure = _frontier_figure_markup()
    assert "{" not in figure and "}" not in figure, figure
    assert "&quot;" not in figure, figure
    attrs = _frontier_mount_attrs()
    # No `data-block`: it was emitted and asserted here and read by nobody, while `frontier.json`
    # (which the widget fetches anyway) carries `block_id` -- and the widget quotes THAT in its
    # readout. Two sources for one fact is drift waiting to happen.
    assert set(attrs) == {"data-widget", "data-bundle", "data-target-displacement",
                          "data-target-permeability", "data-aspect"}, sorted(attrs)


def test_frontier_mount_point_states_the_bundles_own_targets() -> None:
    """Both guides boot at the calibrated standards read from `frontier.json`, and the caption
    states those two numbers in the same whole-percent form the fallback PNG's legend uses. A
    hand-typed target here would contradict the dashed guides on the image directly above it."""
    import json

    from scripts.gen_site_pages import EXPLORE

    bundle = json.loads((EXPLORE / "frontier.json").read_text(encoding="utf-8"))
    attrs = _frontier_mount_attrs()

    assert float(attrs["data-target-displacement"]) == bundle["matched_displacement"]
    assert float(attrs["data-target-permeability"]) == bundle["matched_permeability"]
    # data-aspect is the FALLBACK IMAGE's own shape, read from that PNG's IHDR header rather than
    # restated, so the widget occupies the space the image it replaces occupied.
    assert 1.0 < float(attrs["data-aspect"]) < 2.0

    caption = _frontier_figure_markup()
    assert f"{bundle['matched_displacement']:.0%} displacement" in caption
    assert f"{bundle['matched_permeability']:.0%} permeability" in caption
    # ...and NOT the site's usual one-decimal form, which would read 10.0% beside an image whose own
    # legend reads 10%.
    assert f"{bundle['matched_displacement'] * 100:.1f}% displacement" not in caption


def test_the_page_ships_the_bundle_it_points_at_with_everything_the_widget_needs(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`data-bundle` is a URL, and a URL that 404s (or resolves to a bundle predating the widget's
    requirements) fails silently behind an intact-looking PNG. So: the figure's own URL must resolve
    to a file the generator actually copied, byte-identical to the artifact, and that shipped copy
    must carry the chart block and the per-method label/colour the widget refuses to draw without.

    `DOCS`/`ASSETS` are redirected into `tmp_path` so this reads a copy no other test can be
    mid-write on: `uv run pytest` runs under xdist, several tests call `gen_methods_overview()`
    (which copies these assets), and comparing bytes against the shared `docs/assets` copy made this
    test fail intermittently -- a flake I introduced and would rather not ship. Generating the page
    is what performs the copy, so calling the generator here is the fixture."""
    import json

    import scripts.gen_site_pages as gsp
    from scripts.gen_site_pages import EXPLORE, gen_methods_overview

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    gen_methods_overview()
    url = _frontier_mount_attrs()["data-bundle"]
    shipped = tmp_path / url
    assert shipped.exists(), f"the page points at {url}, which was never copied into docs/"
    assert shipped.read_bytes() == (EXPLORE / "frontier.json").read_bytes()

    bundle = json.loads(shipped.read_text(encoding="utf-8"))
    chart = bundle["chart"]
    for name in ("line_width", "guide_width", "marker_radius", "grid_opacity", "tick_target",
                 "pad", "slider_step", "permeability_max"):
        assert isinstance(chart[name], int | float), (name, chart.get(name))
    for name in ("x_label", "y_label", "guide_colour", "guide_dash"):
        assert isinstance(chart[name], str) and chart[name], (name, chart.get(name))
    for method, curve in bundle["methods"].items():
        assert curve["label"] and isinstance(curve["label"], str), method
        assert re.fullmatch(r"#[0-9a-f]{6}", curve["colour"]), (method, curve["colour"])


def test_methods_index_rewrites_the_frontier_bundle_url_for_its_served_depth(
        tmp_path: Path) -> None:
    """The same gap piece C closed for permeability.md, now for the Methods index -- and on the REAL
    generated body, not a synthetic one, so it also catches the generator emitting a bundle URL in
    some other form (absolute, or already prefixed) that `_write_page`'s rewriter would silently
    leave alone. methodology/methods/index.md serves at <base>/methodology/methods/, so its raw-HTML
    asset URLs need two levels up; get it wrong and the widget's fetch 404s while the page and its
    PNG fallback still look perfect."""
    from scripts.gen_site_pages import _write_page, gen_methods_overview

    out = tmp_path / "index.md"
    _write_page(out, gen_methods_overview(), depth=2, url_depth=2, title="The methods")
    text = out.read_text(encoding="utf-8")
    assert 'data-bundle="../../assets/explore/frontier.json"' in text
    assert 'data-bundle="assets/' not in text
    assert 'src="../../assets/multiblock_depth_density/frontier_' in text


def test_methods_index_is_written_at_the_url_depth_its_widget_needs() -> None:
    """The test above proves `_write_page` rewrites correctly AT url_depth=2; this one proves main()
    actually passes 2. Without it the pair is vacuous -- the rewrite could be perfect and the call
    site could still pass the source depth (or the sibling method pages' 3) and 404 every asset on
    the page. Asserted against the source text, the same way this file's method-registry guards
    read the generator."""
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    m = re.search(r'_write_page\(methods_dir / "index\.md", gen_methods_overview\(\),\s*'
                  r"depth=(\d+), url_depth=(\d+)", src)
    assert m is not None, "could not find main()'s _write_page call for the methods index"
    # methodology/methods/index.md serves at <base>/methodology/methods/ -- two segments deep.
    assert m.group(2) == "2", f"methods index written at url_depth={m.group(2)}, needs 2"


# ------------------------------------------------- the Displacement page's field widget (D2)

@pytest.fixture
def displacement_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The rendered Displacement partial -- markers filled, exactly what `main()` writes out.

    Rendered here rather than read off `docs/methodology/displacement.md`, which is GITIGNORED: a
    fresh checkout does not have it, so a test reading that file would assert nothing (or error)
    depending on whether someone had run the generator first.

    `DOCS`/`ASSETS` are redirected into `tmp_path` because rendering a partial RUNS EVERY PRODUCER
    (`_render_partial` calls each entry of `MARKERS` whether or not that marker appears in the
    page), and producers copy assets. Left pointing at the real tree, every test using this would
    write megabytes into the working copy -- and `uv run pytest` runs under xdist, so several
    workers would do it to the same paths at once. The partials themselves still come from the real
    repo: `PARTIALS` is bound at import time, so moving `DOCS` does not move it."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    return gsp._render_partial("displacement")


def test_the_displacement_page_carries_exactly_one_field_widget(displacement_body: str) -> None:
    """One mount point, over its own PNG fallback, with the marker actually substituted.

    An unfilled `<!-- DISPFIELD -->` ships as a literal HTML comment and the page looks fine while
    the figure is simply absent -- the same silence a widget that never boots produces, one stage
    earlier."""
    body = displacement_body
    assert body.count('data-widget="displacement-field"') == 1
    # The bundle URL in the exact `assets/...` form `_write_page` rewrites (see the test below).
    assert 'data-bundle="assets/displacement-field/field.json"' in body
    # The fallback image stays IN the figure: dom/error.ts tells the reader "The static image above
    # still applies", which is only true while the <img> is there for the widget to remove itself.
    assert '<img src="assets/displacement-field/field.png"' in body
    assert "<!-- DISPFIELD -->" not in body, "the marker was emitted instead of replaced"
    # The mount point sits on the <figure> itself, never a wrapping <div> -- see `_figure`.
    assert '<div data-widget="displacement-field"' not in body


def test_the_field_figure_ships_the_bundle_and_png_it_points_at(
        displacement_body: str, tmp_path: Path) -> None:
    """Both halves of the 404 that fails silently: the asset has to be COPIED into docs/, and the
    URL has to be rewritten for the depth displacement.md is SERVED at. It serves at
    <base>/methodology/displacement/ -- two segments -- so `../assets/` would 404 the widget's
    fetch while the page and its PNG still look perfect.

    The `displacement_body` fixture has already redirected `ASSETS` into this test's own
    `tmp_path`, which is where the copies below are looked for -- and why comparing bytes here
    cannot race the shared `docs/assets` copy that other xdist workers may be mid-write on."""
    from scripts.gen_site_pages import EXAMPLES, _write_page

    out = tmp_path / "displacement.md"
    # depth/url_depth verbatim from main()'s own call for this page (asserted below).
    _write_page(out, displacement_body, depth=1, url_depth=2, title="Displacement")
    text = out.read_text(encoding="utf-8")

    assert 'data-bundle="../../assets/displacement-field/field.json"' in text
    assert 'src="../../assets/displacement-field/field.png"' in text
    assert 'data-bundle="assets/' not in text
    for name in ("field.json", "field.png"):
        shipped = tmp_path / "assets" / "displacement-field" / name
        assert shipped.exists(), f"the page points at {name}, which was never copied into docs/"
        assert shipped.read_bytes() == (EXAMPLES / "displacement-field" / name).read_bytes()

    # ...and that main() actually passes url_depth=2 for this page. Without this the pair is
    # vacuous: the rewrite above could be perfect and the call site could still pass 1.
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    m = re.search(r'_write_page\(methodology_dir / "displacement\.md",\s*'
                  r'_render_partial\("displacement"\),\s*depth=(\d+), url_depth=(\d+)', src)
    assert m is not None, "could not find main()'s _write_page call for displacement.md"
    assert m.group(2) == "2", f"displacement.md written at url_depth={m.group(2)}, needs 2"


def test_the_caption_quotes_baked_numbers_and_not_typed_ones(displacement_body: str) -> None:
    """Every number on this page comes off disk. The apart/coincident pair is the whole point of
    the caption -- it is how a reader with JavaScript off gets the overlap-is-free comparison, and
    the two roads merged costing exactly what one road costs is the claim the section above it
    makes in prose."""
    import json

    from scripts.gen_site_pages import EXAMPLES

    bundle = json.loads(
        (EXAMPLES / "displacement-field" / "field.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in bundle["reference"]}
    start = displacement_body.index('<figure data-widget="displacement-field"')
    figure = displacement_body[start:displacement_body.index("</figure>", start) + len("</figure>")]

    # First, what the artifact has to still SAY for the caption to be worth printing. Both are
    # properties of the bake, not of this file: `coincident` is road 1 twice, so its cost is
    # identically road 1's, and two roads held apart must cost more than the same two merged. If a
    # re-bake ever broke either, the caption would be publishing a false claim in correct numbers,
    # and this fails with a message that says to re-bake rather than to edit the test.
    assert cases["coincident"]["sum_c"] == cases["road1"]["sum_c"], (
        "the bundle no longer says merging two roads costs what one costs; the caption's whole "
        "comparison is stale -- re-bake before rewriting this test")
    assert cases["apart"]["sum_c"] > cases["coincident"]["sum_c"], (
        "the bundle no longer says pulling two roads apart costs more than merging them")

    # Then each number PINNED TO ITS OWN CLAUSE, which is what makes the assertions asymmetric.
    # Two weaker spellings were measured to guard nothing:
    #   * `f"{road1:.1f}" in body` is satisfied by the `coincident` sentence, since those two are
    #     the same number -- hand-typing the headline as 32.5 left it green;
    #   * counting occurrences (`figure.count(...) == 2`) is symmetric under SWAPPING `apart` and
    #     `coincident`, and that swap publishes the exact inverse of the claim -- that merging two
    #     roads makes them dearer -- with every test still green.
    # Matching the clause text is therefore deliberate, not incidental: the wording IS the claim,
    # and `_displacement_field_figure`'s docstring says so where a rewriter would see it.
    assert f"<strong>{cases['road1']['sum_c']:.1f}</strong>" in figure, (
        "the fallback PNG's own number is not the caption's headline")
    assert f"the two roads cost {cases['apart']['sum_c']:.1f} between them" in figure, (
        "the apart cost is not stated as what the two roads cost held apart")
    assert f"the same two cost {cases['coincident']['sum_c']:.1f}" in figure, (
        "the merged cost is not stated as what the same two cost dragged together")
    assert f"{cases['road1']['fraction']:.1%}" in figure
    assert str(bundle["n_buildings"]) in figure
    # Brace-free, like the frontier mount point: a literal `{` inside a raw-HTML block is what fix
    # round 1 put into the markdown, and the escaped-JSON attributes that caused it.
    assert "{" not in figure and "}" not in figure, figure

    # ...and no decimal literal in the PRODUCER at all. Every check above is a presence check on
    # the rendered page, and a number typed as the value it currently happens to have passes every
    # one of them -- `coincident`'s 32.0 could be hand-written today and nothing on this page would
    # look wrong until the next re-bake moved it. Read off the source instead, the same way
    # `test_published_method_count_is_generated_not_typed` does. Format specs (`:.1f`, `:.1%`) do
    # not match: the pattern needs a digit BEFORE the dot.
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    start = src.index("def _displacement_field_figure")
    producer = src[start:src.index("\ndef ", start)]
    typed = re.findall(r"\d+\.\d+", producer)
    assert not typed, (
        f"decimal literals in _displacement_field_figure: {typed}. Every number in that caption "
        f"must be read from field.json. (This scans the docstring too -- if one of these is prose, "
        f"spell it without a decimal point rather than weakening the guard.)")


def test_the_page_no_longer_claims_a_parcel_can_lack_a_building(displacement_body: str) -> None:
    """`src/reblock/mesh.py`: parcels are Voronoi cells OF the building points, so the
    correspondence is exactly one cell per building. The old sentence described a vacant lot this
    pipeline cannot produce."""
    body = displacement_body
    assert "no building standing on it" not in body
    assert "Voronoi" in body, "the corrected section should say what parcels actually are"


# ------------------------------------------------- the Screening page's two widgets (D3)

def render_page(name: str) -> str:
    """Fully render partial NAME the way `main()` ships it: fill its markers, then apply
    `_write_page`'s depth/url_depth rewrite for wherever `main()` actually writes it -- so a test
    reading this sees the SERVED markup, `../../assets/` prefixes and all, not `_render_partial`'s
    intermediate output before that rewrite runs. depth/url_depth are read off `main()`'s own
    `_write_page` call for this partial rather than duplicated here -- the same reasoning
    `test_the_field_figure_ships_the_bundle_and_png_it_points_at`'s tail applies to displacement.md
    below: a hardcoded second copy of those two numbers could silently drift from what `main()`
    actually passes.

    `DOCS`/`ASSETS` are redirected into a throwaway directory for the call, exactly
    `displacement_body`'s own fixture reasoning: `_render_partial` runs EVERY producer in MARKERS,
    whether or not that marker appears on THIS page, and producers copy assets -- pointed at the
    real tree this would write megabytes into the working copy on every test collection, and race
    other xdist workers doing the same. `PARTIALS` stays bound to the real repo (bound at import
    time), so the partial's own committed markup is still what gets rendered.
    """
    import scripts.gen_site_pages as gsp

    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    # The three receivers `main()` actually passes `_render_partial` output to: `DOCS` for the two
    # pages written straight into docs/ (reproduce.md, explore.md), `results_dir` for bakeoff.md
    # and nairobi.md, `methodology_dir` for the rest. Named rather than matched as `[a-z_]+` so a
    # partial written somewhere new fails the `m is not None` assertion below instead of being
    # matched against a receiver nobody checked.
    m = re.search(rf'_write_page\((?:DOCS|methodology_dir|results_dir) / "{name}\.md",\s*'
                  rf'_render_partial\("{name}"\),\s*depth=(\d+), url_depth=(\d+)', src)
    assert m is not None, f"could not find main()'s _write_page call for {name}.md"
    depth, url_depth = int(m.group(1)), int(m.group(2))

    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        tmp_path = Path(tmp)
        mp.setattr(gsp, "DOCS", tmp_path)
        mp.setattr(gsp, "ASSETS", tmp_path / "assets")
        out = tmp_path / f"{name}.md"
        gsp._write_page(out, gsp._render_partial(name), depth=depth, url_depth=url_depth,
                        title=name.title())
        return out.read_text(encoding="utf-8")


# ----------------------------------------------- the Permeability page's draw-road widget (Task 6)

@pytest.fixture
def permeability_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The rendered Permeability partial -- markers filled, `displacement_body`'s own fixture
    reasoning applied to this page: `DOCS`/`ASSETS` are redirected so producers don't write into
    the real tree or race other xdist workers, and `PARTIALS` stays bound to the real repo (bound
    at import time)."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    return gsp._render_partial("permeability")


def test_the_permeability_page_carries_the_draw_road_widget(permeability_body: str) -> None:
    assert permeability_body.count('data-widget="draw-road"') == 1
    assert 'data-bundle="assets/authoring/block.json"' in permeability_body
    assert 'data-wheel="assets/wheels/' in permeability_body


def test_the_draw_road_paths_are_rewritten_for_the_served_depth() -> None:
    """permeability.md serves at <base>/methodology/permeability/ -- url_depth 2. `data-wheel` is
    a fetch URL exactly as `data-bundle` is, so it needs its own `.replace()` in `_write_page`:
    `data-bundle="assets/` is not a substring of `data-wheel="assets/`, which is the same trap
    ScreenMap's two attributes sprang."""
    page = render_page("permeability")
    assert 'data-bundle="../../assets/authoring/block.json"' in page
    assert 'data-wheel="../../assets/wheels/' in page
    for attr in ("data-bundle", "data-wheel"):
        assert f'{attr}="assets/' not in page, attr


def test_a_stray_wheel_in_dist_cannot_become_what_the_page_publishes(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Plants the exact file the old code would have picked, and requires the page to ignore it.

    A value-only assertion (`data-wheel` names the declared version) passes on any checkout with
    one wheel in `dist/`, which is every CI run -- so it would go green against a regression to
    `sorted(glob("reblock-*.whl"))[-1]` and pin nothing. `dist/` is gitignored scratch that can
    hold a hand-renamed copy or a leftover from another version, and under the glob any name
    sorting after the real wheel silently became what the published page told every reader to
    install, with nothing raising and the page looking correct. So the stray has to actually be
    there while the page is rendered.

    `zzz-stray` sorts after `reblock-<version>-...` for any version this project will have, which
    is what makes it the file the glob would have chosen."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    name, version = pyproject["project"]["name"], pyproject["project"]["version"]
    real = ROOT / "dist" / f"{name}-{version}-py3-none-any.whl"
    assert real.exists(), (
        f"{real} is missing -- `make wheel` builds it, and `test-py` depends on that target, so "
        f"reaching this line without it means the dependency stopped firing")

    import scripts.gen_site_pages as gsp

    # Same redirection every other producer test uses: assets land in a throwaway directory rather
    # than the real tree, so this cannot race an xdist sibling or leave a stray in docs/assets/.
    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")

    stray = ROOT / "dist" / f"{name}-zzz-stray.whl"
    stray.write_bytes(real.read_bytes())
    try:
        body = gsp._render_partial("permeability")
        assert f'data-wheel="assets/wheels/{name}-{version}-py3-none-any.whl"' in body
        assert "zzz-stray" not in body, (
            "a stray wheel in dist/ became the URL the page publishes -- the wheel is being "
            "searched for rather than named from pyproject.toml's declaration")
    finally:
        stray.unlink()


def test_the_draw_road_caption_types_no_numbers_of_its_own() -> None:
    """The block id and parcel count are read from `block.json`; nothing else in the caption is a
    number. Same scan the displacement-field caption gets, for the same reason: a typed number is
    the drift class this site has a documented history of, and it survives review precisely
    because it looks right on the day it is written."""
    import inspect

    from scripts.gen_site_pages import _draw_road_figure

    body = inspect.getsource(_draw_road_figure)
    body = body[body.index("caption = "):body.index("return _figure")]
    assert not re.search(r"\d", body), (
        f"a literal digit appears in the draw-road caption -- every number there must be read "
        f"from examples/authoring/block.json:\n{body}")


def test_screening_page_mounts_both_widgets() -> None:
    """Each mount point carries data-widget and data-bundle, and the bundle path is relative to
    the GENERATED page's directory (docs/methodology/), not to docs/. D2 shipped `../assets/`
    where `../../assets/` was needed and the widget 404'd behind an intact-looking PNG."""
    page = render_page("screening")
    assert 'data-widget="screen-map"' in page
    assert 'data-widget="region-grow"' in page
    for url in re.findall(r'data-bundle="([^"]+)"', page):
        assert url.startswith("../../assets/"), url


def test_screening_page_rewrites_the_screen_maps_two_bundle_urls() -> None:
    """ScreenMap carries TWO bundle attributes -- `data-bundle-capetown`, `data-bundle-nairobi` --
    not the single `data-bundle` every other widget on this site uses, and the test right above
    this one cannot see either: its `data-bundle="([^"]+)"` regex does not match a `-capetown`/
    `-nairobi`-suffixed attribute name (that literal substring `data-bundle="` never occurs inside
    `data-bundle-capetown="`). Without a `_write_page` rewrite entry for each of these two exact
    names, ScreenMap's bundles would ship un-rewritten and 404 behind an intact-looking PNG --
    silently, since nothing else on the page would look wrong. This is the `data-bundle` path trap
    this task's own brief is named for, on the one mount point the general regex cannot audit."""
    page = render_page("screening")
    assert 'data-bundle-capetown="../../assets/screen-map/capetown.json"' in page
    assert 'data-bundle-nairobi="../../assets/screen-map/nairobi.json"' in page
    assert 'data-bundle-capetown="assets/' not in page
    assert 'data-bundle-nairobi="assets/' not in page


@pytest.fixture
def screening_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The rendered Screening partial -- markers filled, exactly `displacement_body`'s own
    reasoning applied to this page: `DOCS`/`ASSETS` are redirected so producers don't write into
    the real tree or race other xdist workers, and `PARTIALS` stays bound to the real repo (bound
    at import time)."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    return gsp._render_partial("screening")


def test_the_screening_page_carries_exactly_one_of_each_widget(screening_body: str) -> None:
    """One mount point each, over their own PNG fallbacks, with both markers actually substituted
    -- `test_the_displacement_page_carries_exactly_one_field_widget`'s own reasoning, doubled."""
    body = screening_body
    assert body.count('data-widget="region-grow"') == 1
    assert body.count('data-widget="screen-map"') == 1
    assert 'data-bundle="assets/region-grow/hood.json"' in body
    assert 'data-bundle-capetown="assets/screen-map/capetown.json"' in body
    assert 'data-bundle-nairobi="assets/screen-map/nairobi.json"' in body
    assert '<img src="assets/region-grow/hood.png"' in body
    assert '<img src="assets/screen-map/screen_map.png"' in body
    assert "<!-- SCREENMAP -->" not in body, "the marker was emitted instead of replaced"
    assert "<!-- REGIONGROW -->" not in body, "the marker was emitted instead of replaced"
    # The mount point sits on the <figure> itself, never a wrapping <div> -- see `_figure`.
    assert '<div data-widget="region-grow"' not in body
    assert '<div data-widget="screen-map"' not in body


def test_the_region_grow_caption_states_the_two_regimes_finding(screening_body: str) -> None:
    """The finding the widget publishes (design §1.3 / §2.2): at the shipped floor, growth stops
    at the seed alone because `max_buildings` is a block budget under the default data source and
    a building budget here. Both halves of the caption's headline numbers -- the default-budget
    boot state (3,000 buildings / 11 blocks / 3,072 buildings) AND the floor state (150 buildings /
    1 block / 165 buildings) -- are pinned to `hood.json`, never typed: guarded the same way
    `test_the_caption_quotes_baked_numbers_and_not_typed_ones` guards the displacement page.

    Fix round 1 (2026-08-21): this test originally asserted only the floor half. A reviewer's
    fault injection (`boot = by_budget[budget["default"]]` -> `by_budget[600]`, a plausible
    wrong-key bug producing "default budget of 600 buildings -- 3 blocks, 721 buildings") left it
    green, because nothing checked that the BOOT numbers in the figure actually came from the
    `budget["default"]` case. The three `boot`-derived assertions below close that gap."""
    import json

    from scripts.gen_site_pages import REGIONGROW

    bundle = json.loads((REGIONGROW / "hood.json").read_text(encoding="utf-8"))
    blocks, seed, budget = bundle["blocks"], bundle["seed"], bundle["budget"]
    seed_block = next(b for b in blocks if b["block_id"] == seed)
    by_budget = {r["max_buildings"]: r for r in bundle["reference"] if r["seed"] == seed}
    boot, floor_case = by_budget[budget["default"]], by_budget[budget["min"]]
    assert len(floor_case["order"]) == 1, (
        "the pinned seed no longer collapses to itself at the shipped floor; the caption's whole "
        "claim is stale -- re-bake before rewriting this test")

    start = screening_body.index('<figure data-widget="region-grow"')
    figure = screening_body[start:screening_body.index("</figure>", start) + len("</figure>")]

    assert f"{len(blocks):,}-block neighbourhood" in figure
    assert f"{seed_block['n']:,}" in figure
    assert f"{budget['min']:,}-building floor" in figure
    assert "the seed alone" in figure
    assert "block budget" in figure
    assert "building budget" in figure
    assert "two regimes" in figure

    # The boot (default-budget) half -- absent before fix round 1, which is exactly why a wrong
    # `by_budget` key there was invisible to this test.
    assert f"{boot['max_buildings']:,}" in figure
    assert f"{len(boot['order']):,}" in figure
    assert f"{boot['buildings']:,}" in figure

    # ...and no literal number in the PRODUCER itself -- `test_the_caption_quotes_baked_numbers_
    # and_not_typed_ones`'s own reasoning: a number typed as its current value passes every
    # presence check above and only breaks on the next re-bake. Decimals only (format specs like
    # `:,` carry no digit-dot-digit), matching the displacement-field guard's own scan.
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    start = src.index("def _region_grow_figure")
    producer = src[start:src.index("\ndef ", start)]
    typed = re.findall(r"\d+\.\d+", producer)
    assert not typed, (
        f"decimal literals in _region_grow_figure: {typed}. Every number in that caption must be "
        f"read from hood.json.")


def test_the_screen_map_caption_quotes_the_floor_and_is_honest_about_nairobi(
        screening_body: str) -> None:
    """Pool size, precision and recall are pinned to `capetown.json`'s own `floors`, never typed;
    Nairobi's absent ground truth must not be papered over with an invented number (design §3.4)."""
    import json

    from scripts.gen_site_pages import SCREENMAP

    capetown = json.loads((SCREENMAP / "capetown.json").read_text(encoding="utf-8"))
    nairobi = json.loads((SCREENMAP / "nairobi.json").read_text(encoding="utf-8"))
    cape_floor = next(f for f in capetown["floors"] if f["metric"] == "depth_density_proxy")
    nai_floor = next(f for f in nairobi["floors"] if f["metric"] == "depth_density_proxy")
    assert cape_floor["precision"] is not None and cape_floor["recall"] is not None
    assert nai_floor["precision"] is None and nai_floor["recall"] is None, (
        "nairobi.json now carries a precision/recall for its floor -- the caption's claim that no "
        "ground truth exists for Nairobi is stale")

    start = screening_body.index('<figure data-widget="screen-map"')
    figure = screening_body[start:screening_body.index("</figure>", start) + len("</figure>")]

    assert f"{cape_floor['n']:,}" in figure
    assert f"{capetown['n_blocks']:,}" in figure
    assert f"{100 * cape_floor['precision']:.1f}%" in figure
    assert f"{100 * cape_floor['recall']:.1f}%" in figure
    assert f"{nai_floor['n']:,}" in figure
    assert f"{nairobi['n_blocks']:,}" in figure
    # No Nairobi precision/recall number exists to quote, and the caption must say so rather than
    # silently omitting the topic.
    assert "no precision or recall" in figure

    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    start = src.index("def _screen_map_figure")
    producer = src[start:src.index("\ndef ", start)]
    typed = re.findall(r"\d+\.\d+", producer)
    assert not typed, (
        f"decimal literals in _screen_map_figure: {typed}. Every number in that caption must be "
        f"read from capetown.json/nairobi.json.")


def test_the_screen_map_figure_describes_all_of_its_encodings(screening_body: str) -> None:
    """The map encodes a block's fill -- red selected, gold surveyed informal and not selected, grey
    neither -- and a blue follow ring, and the site copy once described only some of them. Task 6
    added the ring to the picture and to examples/screen-map/README.md, and left the alt text and
    caption behind: a reader met an unexplained blue circle, and a screen-reader user met nothing
    at all, because a canvas and a PNG carry no accessible text beyond what the alt and caption
    say.

    ALT AND CAPTION ARE CHECKED SEPARATELY, not as one blob. They are read in different situations
    -- the alt stands in for the image, the caption sits beside it -- so a fix that named the ring
    in only one of them would still leave one of those two readers short, and a single search over
    the whole <figure> would call that fixed."""
    import json

    from scripts.gen_site_pages import SCREENMAP

    capetown = json.loads((SCREENMAP / "capetown.json").read_text(encoding="utf-8"))
    follow = capetown["follow"]

    start = screening_body.index('<figure data-widget="screen-map"')
    figure = screening_body[start:screening_body.index("</figure>", start) + len("</figure>")]
    alt = re.search(r'alt="([^"]*)"', figure)
    assert alt is not None, "the screen-map figure lost its alt text"

    # Three fills and the ring (owner, 2026-10-29): red is the selection, whatever the survey says;
    # gold the surveyed informal blocks it leaves; grey the rest. No outline encodes anything now.
    for encoding in ("red fills", "gold", "grey", "blue ring"):
        assert encoding in alt.group(1).lower(), f"{encoding!r} missing from the alt text"
    for encoding in ("red fills", "gold", "grey", "blue ring", "whatever the survey says"):
        assert encoding in figure.lower(), f"{encoding!r} missing from the caption"
    assert "outline" not in re.sub(r"a ring\s+rather than an outline", "", figure), (
        "the caption still describes an outline encoding")

    # The ring's block id comes off the bundle's own `follow`, never typed -- the same rule the
    # floor numbers above follow, and what makes a re-bake onto a different block move this caption
    # with it. Asserted against the CAPTION specifically: the alt deliberately omits the id (the
    # figcaption a screen reader reaches right after carries it, and repeating it would be F5).
    assert follow["block_id"] in figure
    assert follow["block_id"] not in alt.group(1)


# --------------------------------------------------- the Explore page's five mount points (T8)

@pytest.fixture
def explore_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The rendered Explore partial -- markers filled, `screening_body`'s own fixture reasoning
    applied to the page that carries every widget on the site: `DOCS`/`ASSETS` are redirected so
    producers don't write into the real tree or race other xdist workers, and `PARTIALS` stays
    bound to the real repo (bound at import time)."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    return gsp._render_partial("explore")


def test_explore_carries_a_mount_point_for_every_interactive_stage(explore_body: str) -> None:
    """The page the whole piece exists for: one mount point per INTERACTIVE stage, in pipeline
    order, each substituted from a marker rather than typed.

    Four, not five. The Permeability stage was a `perm-graph` mount point until 2026-09-20 and is
    now a committed GIF: the widget it booted was a slider over a per-prefix potential/current
    table, which cost 83 MB on the spine block, and `gen_example` already bakes exactly that
    build-order animation at 850 KB. So the stage is still there and still third -- it just is not
    a widget any more, which is what the separate assertion below pins."""
    order = ["screen-map", "region-grow", "displacement-field", "frontier"]
    assert re.findall(r'data-widget="([a-z-]+)"', explore_body) == order


def test_the_permeability_stage_is_the_build_order_gif(explore_body: str) -> None:
    """The stage between Growth and Displacement shows the animation, not a slider.

    Asserted positively AND negatively: that the GIF is there, and that no `perm-graph` mount
    point came back with it. Only the second catches a re-added widget sitting beside the GIF,
    which is what a partial revert would leave and what would quietly restore the 83 MB fetch."""
    assert "perm-graph" not in explore_body, "the perm-graph widget is gone; see gen_web_bundle"
    assert re.search(r'<img src="[^"]*reblock_[a-z_]+\.gif"', explore_body), \
        "the Permeability stage should carry gen_example's committed build-order GIF"


def test_explore_rewrites_the_screen_maps_two_bundle_urls() -> None:
    """explore.md serves at <base>/explore/ -- url_depth 1 -- so each bundle path needs exactly one
    `../`. ScreenMap carries TWO bundle attributes where every other widget carries one, and the
    general `data-bundle="([^"]+)"` regex matches NEITHER of them (that literal substring never
    occurs inside `data-bundle-capetown="`). This is the same trap the Screening page's own twin of
    this test guards, on the page that now carries the most mount points on the site."""
    page = render_page("explore")
    assert 'data-bundle-capetown="../assets/screen-map/capetown.json"' in page
    assert 'data-bundle-nairobi="../assets/screen-map/nairobi.json"' in page
    for attr in ("data-bundle-capetown", "data-bundle-nairobi"):
        # INSIDE the loop. Hoisted out, this runs once against whichever `attr` the loop left
        # behind -- checking one of the two, which is the exact half-blind shape D3's own guard
        # had and the reason this test exists.
        assert f'{attr}="assets/' not in page, attr
    for url in re.findall(r'data-bundle="([^"]+)"', page):
        assert url.startswith("../assets/"), url


def test_explore_is_generated_and_gitignored() -> None:
    """A generated page that is committed drifts from its artifacts the moment anything re-bakes --
    the reason docs/index.md and docs/reproduce.md are ignored too."""
    ignored = [line.strip()
               for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "docs/explore.md" in ignored


def test_the_widget_bundle_guard_can_see_the_explore_page() -> None:
    """`_assert_widget_bundle_present` scans a HAND-WRITTEN page list. Explore carries the most
    mount points of any page on the site; leaving it out would make the guard blind exactly where a
    missing docs/js/widgets.js does the most damage -- five dead widgets behind five intact PNGs."""
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text(encoding="utf-8")
    block = re.search(r"generated_pages = \(([^)]*)\)", src, flags=re.S)
    assert block is not None, "main()'s generated_pages assignment moved; update this derivation"
    assert '"explore.md"' in block.group(1)


def test_explore_is_in_the_nav() -> None:
    """An orphan page -- one outside nav -- logs at INFO under `mkdocs build --strict` and the
    build still exits 0, so nothing else here would catch its absence."""
    assert re.search(r"^\s+- Explore: explore\.md$",
                     (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), flags=re.M)


def _region_dir(root: Path, *, members: int) -> Path:
    """A minimal example directory: meta, the Lens A table, and one render per figure."""
    import json
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({
        "metric": "depth_density", "deepest_block": "B", "deepest_depth": 9.0,
        "region_parcels": 4321, "region_member_ids": ["B"] * members,
        "region_members": members, "region_mean_depth": 5.0,
        "region_mean_density_per_ha": 120.0, "flagged": 10, "total_blocks": 100}))
    (root / "lens_displacement.csv").write_text(
        "method,road_m,displacement,permeability,access_burden_reduction,at_budget\n"
        "clearance_looped,1234.5,0.0987,0.7,0.9,True\n")
    for name in ("region.png", "after_clearance_looped_disp_depth.png"):
        (root / name).write_bytes(b"png")
    return root


def test_the_hero_draws_roads_and_its_caption_quotes_the_run(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The hero once showed `region.png` -- the region map, which draws no roads -- captioned
    "proposed roads threaded through it" in "a multi-block region" of one block. It must show a
    road render, and every fact in its caption must come from the run.

    FAULT INJECTION: pointing the hero back at `region.png` fails the first assertion."""
    import scripts.gen_site_pages as gsp
    monkeypatch.setattr(gsp, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "docs" / "assets")
    monkeypatch.setattr(gsp, "MB", _region_dir(tmp_path / "one", members=1))
    hero = gsp._hero_block()
    assert f'src="assets/home/after_{gsp.HERO_METHOD}_disp_depth.png"' in hero
    assert "1,234 m" in hero and "9.9%" in hero and "4,321 parcels" in hero
    assert "one street-bounded block" in hero and "multi-block" not in hero
    monkeypatch.setattr(gsp, "MB", _region_dir(tmp_path / "many", members=7))
    assert "a 7-block region of 4,321 parcels" in gsp._hero_block()


def test_the_hero_method_is_the_explore_pages_pinned_method() -> None:
    """Spelled twice because `gen_site_pages` is stdlib-only and `_example_block` is not; held
    equal here so the home page and the Explore page open on the same roads."""
    import scripts.gen_site_pages as gsp
    from scripts._example_block import PINNED_METHOD
    assert gsp.HERO_METHOD == PINNED_METHOD


def test_a_single_block_region_is_never_called_grown_into_blocks(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A seed over the growth budget stays alone, and the captions said "a 1-block region" and
    "1 blocks" over it. The wording comes from `region_members`."""
    import scripts.gen_site_pages as gsp
    monkeypatch.setattr(gsp, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "docs" / "assets")
    one = gsp._region_section(_region_dir(tmp_path / "one", members=1), "x", "## h\n")
    assert "single block of 4,321 parcels" in one and "the seed alone" in one
    assert "1-block" not in one and "1 blocks" not in one
    many = gsp._region_section(_region_dir(tmp_path / "many", members=7), "y", "## h\n")
    assert "7-block region of 4,321 parcels" in many and "7 blocks, 4,321 parcels" in many


def test_the_headline_compares_only_what_lens_a_actually_matched(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lens A returns a network that cannot reach the displacement budget WHOLE, so its row is an
    unmatched number. The headline said "at a matched road budget" and set the winner against the
    footpath network, whose whole network displaced 1.4% against a 10% budget.

    FAULT INJECTION: picking the best row without `_matched` picks `greedy` here and fails."""
    import scripts.gen_site_pages as gsp
    root = _region_dir(tmp_path / "mb", members=1)
    (root / "lens_displacement.csv").write_text(
        "method,road_m,displacement,permeability,access_burden_reduction,at_budget\n"
        "clearance_looped,1234.5,0.1004,0.70,0.9,True\n"
        "greedy_arterial_access_displacement,9000.0,0.0350,0.95,0.9,False\n"
        "osm_footpaths,3094.1,0.0141,0.24,0.0,False\n")
    monkeypatch.setattr(gsp, "MB", root)
    headline = gsp._key_result()
    assert "Looped Tree" in headline and "70.0% permeability" in headline
    assert "matched <strong>10.0%</strong>" in headline and "road budget" not in headline
    assert "all 3,094 m of it, displacing only 1.4%" in headline and "never reaches" in headline
    figures = gsp._key_figures()
    assert "70.0%" in figures and "95.0%" not in figures
    assert "single-block benchmark region" in figures and "1-block" not in figures


# ------------------------------------------------------------------------------- the Demo page

@pytest.fixture
def demo_body(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The rendered Demo partial, markers filled -- `explore_body`'s fixture, for docs/demo.md."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    return gsp._render_partial("demo")


def _demo_bundle() -> dict[str, Any]:
    import json
    path = ROOT / "examples" / "demo" / "demo.json"
    assert path.is_file(), f"{path} is missing; the Demo tests would check nothing"
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _demo_buildout(name: str) -> dict[str, Any]:
    """One of the bundle's build-out files, as demo.json names it."""
    import json
    out: dict[str, Any] = json.loads(
        (ROOT / "examples" / "demo" / name).read_text(encoding="utf-8"))
    return out


def json_text(bundle: dict[str, Any]) -> str:
    import json
    return json.dumps(bundle)


def test_the_demo_partial_types_no_numerals() -> None:
    """The partial's header states the rule: every number on the page, the fires' quoted ones
    included, comes from examples/demo/demo.json through a marker. A digit typed into the prose is
    a second copy no re-bake moves. Markers are capitals only, so they cannot hide a digit."""
    text = (ROOT / "docs" / "_partials" / "demo.md").read_text(encoding="utf-8")
    body = re.sub(r"\A\s*<!--.*?-->", "", text, count=1, flags=re.S)
    assert "<!-- DEMOFIRES -->" in body, "the comment strip ate the body; this test is vacuous"
    digits = re.findall(r".{0,30}\d.{0,30}", body)
    assert not digits, f"digits typed into the Demo partial's prose: {digits}"


def test_the_demo_hides_its_table_of_contents(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner, 2026-10-28: the Demo page drops its table of contents for the width the build-out
    needs. Checked on the written page, front matter and all, as `main()` writes it."""
    import scripts.gen_site_pages as gsp

    src = (ROOT / "scripts" / "gen_site_pages.py").read_text(encoding="utf-8")
    call = re.search(r'_write_page\(DOCS / "demo\.md",[^\n]*\n[^\n]*', src)
    assert call is not None and 'hide=("toc",)' in call.group(0)
    out = tmp_path / "demo.md"
    gsp._write_page(out, "body", depth=0, url_depth=1, title="Demo", hide=("toc",))
    assert out.read_text(encoding="utf-8").startswith('---\ntitle: "Demo"\nhide:\n  - toc\n---\n')


def test_the_demo_calls_the_metric_escape_time_saved_and_names_permeability_once(
        demo_body: str) -> None:
    """Owner, 2026-10-28: on the Demo page the metric is "escape time saved". "Permeability", the
    rest of the site's term, appears once -- in the build-out caption's definition, as a link to
    the Permeability page -- and nowhere else on the page, the bundle's method blurb included."""
    import scripts.gen_site_pages as gsp

    assert gsp.DEMO_METRIC == "escape time saved"
    assert "the roads remove, 1 − P(roads) / P(no roads)" in demo_body
    text = re.sub(r'href="[^"]*"', "", demo_body)
    mentions = re.findall(r".{40}[Pp]ermeability.{10}", text)
    assert len(mentions) == 1, mentions
    assert '<a href="../methodology/permeability/">permeability</a>' in demo_body


def test_the_demo_names_its_metric_once(demo_body: str) -> None:
    """The metric's name may change (the owner is weighing "ease of access"), so the Demo page
    spells it in one place, DEMO_METRIC, and hands it to the build-out widget, which spells none of
    its own."""
    import scripts.gen_site_pages as gsp

    assert f'data-metric-label="{gsp.DEMO_METRIC}"' in demo_body
    widget = (ROOT / "web" / "src" / "widgets" / "build-out.ts").read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", widget, flags=re.S))
    assert "plotOf" in code, "the comment strip ate the code; this test is vacuous"
    # The string literals, with any template's ${...} code taken out: the bundle's own field
    # `f.permeability` is code, not a label.
    literals = [re.sub(r"\$\{[^}]*\}", "", lit)
                for lit in re.findall(r'"[^"\n]*"|`[^`]*`', code)]
    spelled = [lit for lit in literals if re.search(r"[Pp]ermeability", lit)]
    assert not spelled, (
        f"build-out.ts spells the metric's name in {spelled}; take it from data-metric-label")


def test_the_demo_mounts_the_sites_screen_map_untouched_and_the_build_out(demo_body: str) -> None:
    """Owner, 2026-10-28: no block switch, no screen-geometry diagram, no close-up. The Demo's
    screen map is the very figure Screening and Explore show, and the widget carries no close-up
    or page-following code left to rot."""
    import scripts.gen_site_pages as gsp

    assert re.findall(r'data-widget="([a-z-]+)"', demo_body) == [
        "screen-map", "peel-zoom", "build-out", "build-out-compare"]
    assert gsp._screen_map_figure() in demo_body
    for page in ("explore", "screening", "demo"):
        body = render_page(page)
        assert "data-closeup-pad-m" not in body and "data-follow-block" not in body, page
    web = ROOT / "web" / "src"
    assert not (web / "render" / "closeup.ts").exists()
    assert not (web / "widgets" / "screen-geometry.ts").exists()
    assert not (web / "widgets" / "block-switch.ts").exists()
    widget = (web / "widgets" / "screen-map.ts").read_text(encoding="utf-8")
    assert "closeup" not in widget.lower() and "followBlock" not in widget
    assert "data-demo-block" not in demo_body and "typical" not in demo_body


def test_the_demo_rewrites_its_asset_urls_for_its_served_depth() -> None:
    """docs/demo.md serves at <base>/demo/, url_depth 1: every asset it points at needs one `../`
    -- each build-out's file and fallback sheet, the peel figure."""
    page = render_page("demo")
    bundle = _demo_bundle()
    for name in (bundle["block"]["buildout"], bundle["comparison"]["buildout"]):
        assert f'data-bundle="../assets/demo/{name}"' in page, name
        sheet = _demo_buildout(name)["series"][0]["sheet"]["file"]
        assert f'src="../assets/demo/{sheet}"' in page, sheet
    assert 'src="../assets/multiblock_depth_density/before_depth.png"' in page
    for attr in ("src", "href", "data-bundle", "data-bundle-capetown", "data-bundle-nairobi"):
        assert f'{attr}="assets/' not in page, attr


def test_the_demo_ships_every_sheet_beside_its_build_out(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each build-out file, and every series' sheet beside it, where build-out.ts's
    `besideBundle` looks; no view switch or `view` URL key."""
    import scripts.gen_site_pages as gsp

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    gsp._demo_build_out()
    gsp._demo_compare()
    bundle = _demo_bundle()
    names = (bundle["block"]["buildout"], bundle["comparison"]["buildout"])
    assert [len(_demo_buildout(n)["series"]) for n in names] == [1, 2]
    for name in names:
        assert (tmp_path / "assets" / "demo" / name).is_file(), name
        for s in _demo_buildout(name)["series"]:
            assert (tmp_path / "assets" / "demo" / s["sheet"]["file"]).is_file(), s["method"]
    widget = (ROOT / "web" / "src" / "widgets" / "build-out.ts").read_text(encoding="utf-8")
    assert '"view"' not in widget, "the build-out widget still carries a view URL key"


def test_both_build_out_captions_tell_road_from_a_destroyed_building(demo_body: str) -> None:
    """Owner, 2026-10-29: a destroyed building is not road. Both build-outs' captions name the
    two encodings and a damaged home's, and no caption calls a destroyed building grey."""
    captions = re.findall(r"<figcaption>(.*?)</figcaption>", demo_body, flags=re.S)
    keyed = [c for c in captions if "Dark slate is new road" in c]
    assert len(keyed) == 2, captions
    for c in keyed:
        assert "a pale footprint with a thin outline is a building the roads destroyed" in c
        assert "keeps its colour inside the same outline" in c
    assert not re.search(r"[Gg]rey is|road,? or a destroyed building", demo_body)


def test_the_comparison_leads_with_what_the_build_out_shows(demo_body: str) -> None:
    """The finding over the comparison's slider is read from its build-out: the first of our
    networks to save as much escape time as topology's whole one, and at what displacement; and
    each method's stop is said with its reason."""
    bundle = _demo_bundle()
    ours, earlier = _demo_buildout(bundle["comparison"]["buildout"])["series"]
    target = earlier["cells"][-1]["step"]["permeability"]
    reach = next(c["step"] for c in ours["cells"] if c["step"]["permeability"] >= target)
    section = demo_body.split("## Our method and the earlier one")[1]
    takeaway = re.search(r'<p class="sbu-compare__takeaway">([^<]*)</p>', section)
    assert takeaway is not None
    assert f"with {reach['displacement']:.0%} of the floor area displaced" in takeaway.group(1)
    assert f"topology needs {earlier['cells'][-1]['step']['displacement']:.0%}" in takeaway.group(1)
    short = next(c for c in ours["cells"] if c["kind"] == "short")
    assert f"stops at {short['step']['displacement']:.0%}, with {short['why']}" in section
    assert "each in the order it built its roads" in section
    assert "whole run only" not in section and "drainage" not in section


def test_the_screens_two_factors_are_the_blocks_and_multiply_to_its_score(demo_body: str) -> None:
    """The intuition the prose gives in two sentences, as key figures: with s = √(A/n), the
    homes behind each parcel-wide stretch of street are n ÷ (P/s) = √(nA)/P exactly, for any
    shape, and times homes per area they are the shipped score."""
    p = _demo_bundle()["block"]["proxy"]
    n, area, perimeter = p["n"], p["area_m2"], p["perimeter_m"]
    side = (area / n) ** 0.5
    per_slot = n / (perimeter / side)
    assert per_slot == pytest.approx((n * area) ** 0.5 / perimeter, rel=1e-12)
    assert per_slot * n / area == pytest.approx(p["score"], rel=1e-9)
    assert (f"<dd>{per_slot:.0f}</dd><dt>homes behind each parcel-wide stretch of street"
            in demo_body)
    assert f"<dd>{n / area * 10_000:,.0f}</dd><dt>homes per hectare" in demo_body


def test_the_demo_figures_refuse_a_bundle_that_disagrees_with_the_map(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The figure rows are baked on research/roadless, the map's bundle on main; a re-bake of one
    alone must fail the build -- checked for the pool and for the comparison block."""
    import json

    import scripts.gen_site_pages as gsp

    assert gsp._demo_screen_facts(), "the committed bundles disagree already"
    monkeypatch.setattr(gsp, "DEMO", tmp_path)
    for spoil in ("pool", "comparison's n"):
        bundle = _demo_bundle()
        if spoil == "pool":
            bundle["screen"]["pool"] += 1
        else:
            bundle["comparison"]["n_buildings"] += 1
        (tmp_path / "demo.json").write_text(json.dumps(bundle), encoding="utf-8")
        with pytest.raises(SystemExit, match="disagrees"):
            gsp._demo_screen_facts()


def test_the_peel_figure_reads_its_numbers_in_one_zoomable_frame(
        demo_body: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The slower screen's figure: the flagship run's depth render in one zoomable frame, with no
    outline on it (owner, 2026-10-29); its numbers (the deepest ring and the proxy's top) read off
    that run's meta.json and checked against the bundle, the rank off the bundle; and the frame
    handed the image's native width, which caps the zoom."""
    import json

    from PIL import Image

    import scripts.gen_site_pages as gsp

    meta = json.loads((ROOT / "examples" / "multiblock_depth_density" / "meta.json").read_text())
    b = _demo_bundle()["block"]
    assert (f"The deepest home is {meta['deepest_depth']:.0f} rings in; zoom in to see its "
            f"parcels") in demo_body
    assert (f"rank {b['peel']['rank']:,} of the shipped screen's top {meta['flagged']:,}"
            in demo_body)
    assert f"Ring {meta['deepest_depth']:.0f}, the deepest" in demo_body
    png = Image.open(ROOT / "examples" / "multiblock_depth_density" / "before_depth.png")
    figure = re.search(r'<figure class="sbu-peel"([^>]*)>(.*?)</figure>', demo_body, flags=re.S)
    assert figure is not None
    assert f'data-widget="peel-zoom" data-native-px="{png.width}"' in figure.group(1)
    assert figure.group(2).count("<img") == 1, "one frame, not a whole view and a crop"
    assert 'class="skip-lightbox"' in figure.group(2), "a lightbox would fight the gestures"
    assert "data-mark" not in figure.group(1) and "sbu-peel__mark" not in figure.group(2)
    assert "outlined" not in figure.group(2)
    bundle = _demo_bundle()
    bundle["block"]["peel"]["rings"] += 1
    (tmp_path / "demo.json").write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setattr(gsp, "DEMO", tmp_path)
    with pytest.raises(SystemExit, match="disagrees"):
        gsp._demo_peel()


def test_the_fires_are_listed_and_carry_nothing_a_map_could_draw(demo_body: str) -> None:
    """Owner, 2026-10-28: the fires go on no map, since some reports' streets cannot be placed.
    The list shows every fire the bundle has, and the bundle gives a fire no coordinates at all,
    so no widget could draw one without the producer changing first."""
    fires = _demo_bundle()["block"]["fires"]
    assert fires, "no fires in the bundle; this test would check nothing"
    assert demo_body.count('<p class="sbu-fires__when">') == len(fires)
    for f in fires:
        assert set(f) == {"month", "where", "what", "reports"}, f
        for r in f["reports"]:
            assert r["url"] in demo_body


def _runtime_rows(section: str) -> dict[str, list[tuple[str, str, str]]]:
    """The comparison's runtime table: per row's block (its short id), each cell's (time, how,
    class)."""
    m = re.search(r'<table class="sbu-runtimes">(.*?)</table>', section, flags=re.S)
    assert m is not None, "the comparison has no runtime table"
    rows: dict[str, list[tuple[str, str, str]]] = {}
    for row in re.findall(r'<tr><th scope="row">(.*?)</tr>', m.group(1), flags=re.S):
        block = re.search(r"<code>([^<]*)</code>", row)
        assert block is not None
        rows[block.group(1)] = [
            (time, re.sub(r"<[^>]*>", "", how), cls)
            for cls, time, how in re.findall(
                r'<td(?: class="([^"]*)")?><strong>([^<]*)</strong><span>(.*?)</span></td>', row)]
    return rows


def test_the_demo_plots_displacement_and_road_and_the_faceoff_run_time(demo_body: str) -> None:
    """Owner, 2026-10-29: conductance leaves the Demo -- no plot, readout or bundle field; the
    5810 caption says in one sentence why it is not plotted. 5810's build-out plots escape time
    saved against displacement and new road; the face-off adds run time."""
    d = _demo_bundle()
    page, faceoff = (_demo_buildout(d[k]["buildout"]) for k in ("block", "comparison"))
    assert page["plots"] == ["displacement", "road"]
    assert faceoff["plots"] == ["displacement", "road", "time"]
    for bo in (page, faceoff):
        for s in bo["series"]:
            assert all("conductance_m2" not in c["step"] for c in s["cells"])
    assert "road_conductance" not in d["encoding"]
    assert demo_body.count("onductance") == 1
    assert ("The conductance the roads add gives a natural single number for a budget, but the "
            "page plots displacement and road length.") in demo_body
    widget = (ROOT / "web" / "src" / "widgets" / "build-out.ts").read_text(encoding="utf-8")
    assert "onductance" not in widget


def test_the_runtime_table_times_every_method_on_both_blocks(demo_body: str) -> None:
    """Owner, 2026-10-29: one small table near the top of the comparison -- the two blocks by
    topology's original implementation, our optimized port of it, and our method -- each cell
    saying whether it was measured, expected or estimated, every number from the bundle; our
    method's cells green and marked "fastest", so colour is not the only cue; the old times row
    at the end is gone."""
    import scripts.gen_site_pages as gsp

    d = _demo_bundle()
    c, b, name = d["comparison"], d["block"], d["method"]["name"]
    section = demo_body.split("## Our method and the earlier one")[1]
    head = re.search(r"<thead>(.*?)</thead>", section, flags=re.S)
    assert head is not None
    assert re.findall(r'<th scope="col">([^<]*)</th>', head.group(1)) == [
        "Block", "Topology, original implementation", "Topology, our optimized port", name]
    assert section.index('class="sbu-runtimes"') < section.index('data-widget="build-out-compare"')
    rows = _runtime_rows(section)
    short_c, short_b = gsp._short(c["block_id"]), gsp._short(b["block_id"])
    assert list(rows) == [short_c, short_b]
    t = gsp._topology_of(c["topology"], c["block_id"])
    pt = gsp._topology_of(b["topology"], b["block_id"])
    [ours] = [s for s in _demo_buildout(c["buildout"])["series"] if s["label"].endswith(name)]
    [page] = _demo_buildout(b["buildout"])["series"]
    (o1, h1, _), (p1, hp1, _), (s1, hs1, cs1) = rows[short_c]
    (o2, h2, _), (p2, hp2, _), (s2, hs2, cs2) = rows[short_b]
    assert (o1, h1) == (t.original(), "expected1")
    assert (p1, hp1) == (gsp._measured_time(c["topology"]["seconds"]), "measured2")
    assert (o2, h2) == (pt.original(), "estimated1")
    assert (p2, hp2) == (pt.figure(page["run"]["seconds"])[0], "estimated1")
    assert s1 == gsp._duration(ours["run"]["seconds"])
    assert s2 == gsp._duration(page["run"]["seconds"])
    for how, cls in ((hs1, cs1), (hs2, cs2)):
        assert cls == "sbu-runtimes__fastest" and "fastest" in how and "measured" in how
    # The original's time follows from the port's by the fit at the block's size.
    expected = c["topology"]["seconds"] * c["topology"]["original_factor"]
    assert t.original() == f"about {gsp._duration(expected)}"
    assert f"{b['n_buildings']:,} buildings" in section
    assert 'class="sbu-compare__time"' not in demo_body, "the old times row is back"
    assert section.count('aria-label="How these were estimated"') == 4
    assert section.count('aria-label="How this was measured"') == 2


def test_topology_is_named_as_our_port_with_the_originals_times_footnoted(
        demo_body: str) -> None:
    """Owner, 2026-10-29: the site's topology times are our optimized port's. The build-out heads
    topology's column as our port, and one footnote says how every estimate was made -- the
    port's extrapolation and the original's speed-up fit -- read from the bundle
    (atlas/topology_speed/table.csv through build.py)."""
    import scripts.gen_site_pages as gsp

    d = _demo_bundle()
    p = d["topology_port"]
    section = demo_body.split("## Our method and the earlier one")[1]
    labels = [s["label"] for s in _demo_buildout(d["comparison"]["buildout"])["series"]]
    assert f"Earlier method: {gsp.TOPOLOGY_LABEL}" in labels
    note = re.search(r'<p class="sbu-footnote" id="topology-note-1">(.*?)</p>', section)
    assert note is not None and "our optimized port" in note.group(1)
    for fact in (f"<code>{p['port']}</code>", f"<code>{p['original']}</code>",
                 f"{p['n_min']} to {p['n_max']} buildings",
                 f"{p['speedup_min']:.1f} to {p['speedup_max']:.1f} times as long",
                 f"n<sup>{p['exponent']:.2f}</sup>",
                 f"{d['comparison']['topology']['original_factor']:.1f}×",
                 f"{d['block']['topology']['original_factor']:.1f}×",
                 f"<code>{gsp._short(d['block']['block_id'])}</code>, never run"):
        assert fact in note.group(1), fact
    assert len(re.findall(r'class="sbu-footnote"', section)) == 2


def test_the_measured_note_says_where_topologys_time_went(demo_body: str) -> None:
    """Topology's "Built in" sits at its first road's time through the low budgets; the footnote
    on its measured time says why, every number from the bundle's run record."""
    import scripts.gen_site_pages as gsp

    t = _demo_bundle()["comparison"]["topology"]
    section = demo_body.split("## Our method and the earlier one")[1]
    k = re.search(r'aria-label="How this was measured">(\d+)<', section)
    assert k is not None
    note = re.search(rf'id="topology-note-{k.group(1)}"><sup>\d+</sup> (.*?)</p>', section)
    assert note is not None
    for fact in (f"about {gsp._duration(t['fit_central_s'])}", f"{t['setup_s']:.1f} seconds",
                 f"first lands at {gsp._duration(t['first_road_s'])}",
                 f"({t['first_road_s'] / t['seconds']:.0%} of the run)",
                 f"second at {gsp._duration(t['second_road_s'])}",
                 f"other {t['n_rest']:,} take a median {t['rest_median_s']:.2f} seconds"):
        assert fact in note.group(1), fact


def test_a_measured_time_reads_as_the_build_out_widget_states_it() -> None:
    """The generator and the build-out widget state a measured run alike (owner, 2026-10-29: the
    column read "1.9 hours" over prose saying "1.88 hours"): both are tested against one set of
    cases, web/test/fixtures/measured-times.json, which web/test/build-out-boot.test.ts reads
    too."""
    import json

    import scripts.gen_site_pages as gsp

    cases = json.loads((ROOT / "web" / "test" / "fixtures" / "measured-times.json")
                       .read_text(encoding="utf-8"))["cases"]
    assert cases
    for seconds, want in cases:
        assert gsp._measured_time(seconds) == want, seconds


def test_an_unknown_topology_kind_refuses_to_build(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    import json
    import shutil

    import scripts.gen_site_pages as gsp

    bundle = _demo_bundle()
    for f in (ROOT / "examples" / "demo").glob("buildout_*.json"):
        shutil.copy(f, tmp_path / f.name)
    for s in _demo_buildout(bundle["comparison"]["buildout"])["series"]:
        shutil.copy(ROOT / "examples" / "demo" / s["sheet"]["file"], tmp_path / s["sheet"]["file"])
    monkeypatch.setattr(gsp, "DEMO", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    (tmp_path / "demo.json").write_text(json.dumps(bundle), encoding="utf-8")
    assert gsp._demo_compare(), "the comparison renders nothing; this test would check nothing"
    bundle["block"]["topology"]["kind"] = "guessed"
    (tmp_path / "demo.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown topology kind"):
        gsp._demo_compare()


def test_the_survey_note_is_one_sentence_and_points_at_the_caveat(demo_body: str) -> None:
    """The City's survey does not flag the block though it is plainly informal: said once, in one
    sentence, with the bake-off's caveat on the survey's age linked."""
    b = _demo_bundle()["block"]
    assert not b["proxy"]["informal"] and b["plainly_informal"], "update this test"
    [note] = re.findall(r"<p>The City's informal-settlement survey[^\n]*</p>", demo_body)
    assert note.count(". ") == 0 and "plainly informal" in note
    assert '<a href="../results/bakeoff/#caveats">' in note


def test_demo_is_generated_gitignored_in_the_nav_and_seen_by_the_widget_guard() -> None:
    """explore.md's three guards, for demo.md: committed, it drifts; outside nav, it is an orphan
    `mkdocs build --strict` does not fail on; outside `generated_pages`, a missing widgets.js
    leaves its two widgets dead behind their fallbacks."""
    ignored = [line.strip()
               for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "docs/demo.md" in ignored
    assert re.search(r"^\s+- Demo: demo\.md$",
                     (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), flags=re.M)
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text(encoding="utf-8")
    block = re.search(r"generated_pages = \(([^)]*)\)", src, flags=re.S)
    assert block is not None and '"demo.md"' in block.group(1)
