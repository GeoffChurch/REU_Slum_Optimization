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

    from scripts.gen_site_pages import MC, gen_methods_overview

    bundle = json.loads((MC / "frontier.json").read_text(encoding="utf-8"))
    markup = gen_methods_overview()

    assert markup.count('data-widget="frontier"') == 1
    # The fallback image, and the bundle URL in the exact `assets/...` form _write_page rewrites.
    assert f'<img src="assets/method-comparison/frontier_{bundle["block_id"]}.png"' in markup
    assert 'data-bundle="assets/method-comparison/frontier.json"' in markup
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

    from scripts.gen_site_pages import MC

    bundle = json.loads((MC / "frontier.json").read_text(encoding="utf-8"))
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
    mid-write on: `pixi run pytest` runs under xdist, several tests call `gen_methods_overview()`
    (which copies these assets), and comparing bytes against the shared `docs/assets` copy made this
    test fail intermittently -- a flake I introduced and would rather not ship. Generating the page
    is what performs the copy, so calling the generator here is the fixture."""
    import json

    import scripts.gen_site_pages as gsp
    from scripts.gen_site_pages import MC, gen_methods_overview

    monkeypatch.setattr(gsp, "DOCS", tmp_path)
    monkeypatch.setattr(gsp, "ASSETS", tmp_path / "assets")
    gen_methods_overview()
    url = _frontier_mount_attrs()["data-bundle"]
    shipped = tmp_path / url
    assert shipped.exists(), f"the page points at {url}, which was never copied into docs/"
    assert shipped.read_bytes() == (MC / "frontier.json").read_bytes()

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
    assert 'data-bundle="../../assets/method-comparison/frontier.json"' in text
    assert 'data-bundle="assets/' not in text
    assert 'src="../../assets/method-comparison/frontier_' in text


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
    write megabytes into the working copy -- and `pixi run pytest` runs under xdist, so several
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
    m = re.search(rf'_write_page\(methodology_dir / "{name}\.md",\s*'
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
