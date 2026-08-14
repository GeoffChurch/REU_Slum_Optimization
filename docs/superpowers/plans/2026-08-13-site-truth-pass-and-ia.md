# Site truth pass + IA restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the published site true — fix four documented defects — and restructure it into the
methodology spine the redesign needs, with every number on every page generated from artifacts.

**Architecture:** `scripts/gen_site_pages.py` is a stdlib-only dir-reader that writes gitignored
pages from committed run artifacts. Today it folds four markers into one committed partial
(`docs/_intro.md`). This plan generalises that into a **partials mechanism**: handwritten prose
lives in `docs/_partials/*.md` with `<!-- MARKER -->` holes; the generator fills every hole from
artifacts and writes the real page. No number is ever typed into prose. Pages then move into the new
IA, guarded by `mkdocs build --strict`.

**Tech Stack:** Python 3.12 stdlib only (generator), MkDocs Material 9.7.7, mkdocs-glightbox 0.5.2,
pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-site-redesign-design.md` (§Why defects 1–4, §1 IA,
§6 page dispositions). This plan implements **piece A** of §7 only.

## Global Constraints

- **`scripts/gen_site_pages.py` must remain stdlib-only.** CI runs it with only `mkdocs-material`
  and `mkdocs-glightbox` installed; `reblock` is **not importable**. `method_labels.py` is loaded by
  path via `importlib`, never imported as a package. Do not add a third-party import.
- **Run the generator as `python3 scripts/gen_site_pages.py`** (works: no package imports). Scripts
  that *do* import `reblock` must be run as `pixi run python -m scripts.<name>` — pythonpath is
  configured for pytest only.
- **Generated pages are gitignored; partials are committed.** Never commit `docs/index.md`,
  `docs/methodology/`, `docs/results/`, `docs/reproduce.md`, `docs/methods/`.
- **No number in prose.** Every figure, percentage, count, threshold, or command in a partial comes
  from a marker. This is the rule defect 4 broke.
- **`_write_page(depth=, url_depth=)`**: `depth` = source directory depth below `docs/`;
  `url_depth` = number of path segments in the **served** URL under `use_directory_urls`. Getting it
  wrong 404s every figure on the page. Values for every page are given per task.
- **Every guard test must be fault-injected.** Break the thing it guards, watch it fail, restore.
  A guard test that has never failed is not evidence.
- **Pin any new CI dependency exactly**, matching `mkdocs-material==9.7.7`.

---

## File Structure

**New committed partials** (`docs/_partials/`, excluded from the built site by one `exclude_docs`
line):

| file | becomes | responsibility |
|---|---|---|
| `intro.md` | `docs/index.md` | Home prose (migrated from `docs/_intro.md`) |
| `methodology.md` | `docs/methodology/index.md` | pipeline overview + glossary |
| `screening.md` | `docs/methodology/screening.md` | the screen, the gate, region growth |
| `permeability.md` | `docs/methodology/permeability.md` | the egress graph and the metric |
| `displacement.md` | `docs/methodology/displacement.md` | the disk model and the cost axis |
| `bakeoff.md` | `docs/results/bakeoff.md` | screen validation vs. ground truth |
| `nairobi.md` | `docs/results/nairobi.md` | the second city |
| `reproduce.md` | `docs/reproduce.md` | the exact command behind every figure |

**Modified:** `scripts/gen_site_pages.py`, `mkdocs.yml`, `docs/background.md`,
`tests/test_gen_site_pages.py`, `.github/workflows/deploy-site.yml`, `.gitignore`.

**Deleted:** `docs/_intro.md` (migrated), `docs/methodology.md` (dissolved).

**Moved (generated):** `docs/methods/` → `docs/methodology/methods/`; `docs/benchmark.md` →
`docs/results/frontier.md`.

---

### Task 1: The partials mechanism and the method count

Fixes **defect 4** and builds the substrate every later task uses.

**Files:**
- Create: `docs/_partials/intro.md` (git mv from `docs/_intro.md`)
- Modify: `scripts/gen_site_pages.py` (marker loop in `main()` → `_render_partial`; add `_method_count`)
- Modify: `mkdocs.yml` (`exclude_docs`: `_intro.md` → `_partials/`)
- Modify: `.gitignore`
- Test: `tests/test_gen_site_pages.py`

**Interfaces:**
- Produces: `MARKERS: dict[str, Callable[[], str]]` — module-level mapping from marker name
  (e.g. `"KEYRESULT"`) to a zero-argument producer returning a markdown/HTML block.
- Produces: `_render_partial(name: str) -> str` — reads `docs/_partials/<name>.md`, replaces every
  `<!-- MARKER -->` whose name is in `MARKERS` with that producer's output, returns the body.
- Produces: `_method_count() -> str` — the English word for the number of `published=True` entries
  in `METHODS`, capitalised (`"Ten"`).
- Consumed by: Tasks 5, 6, 8, 9, 10, 11.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gen_site_pages.py`:

```python
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
    published = len(re.findall(r'^    M\("[a-z_]+"', src, flags=re.M)) - len(
        re.findall(r'published=False', src))
    assert published == 10, f"expected 10 published methods, registry says {published}"
    intro = (ROOT / "docs" / "_partials" / "intro.md").read_text(encoding="utf-8")
    assert "<!-- METHODCOUNT -->" in intro
    for word in ("Seven", "seven", "Eight", "Nine", "Ten", "Eleven"):
        assert f"{word} road-generation" not in intro, (
            f"'{word}' typed into prose; the count must come from METHODCOUNT")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: all three new tests FAIL.
- `test_every_marker_in_a_partial_has_a_producer` and `test_every_producer_is_used_by_a_partial` —
  `AssertionError: .../docs/_partials does not exist; the partials tests would be vacuous`.
- `test_published_method_count_is_generated_not_typed` — `FileNotFoundError` on
  `docs/_partials/intro.md`.

**If any of the three passes here, stop and fix the test.** Without the `is_dir()` guard these two
would pass vacuously: `Path.glob()` on a missing directory yields nothing rather than raising, so
`used` would be empty and both set differences would be empty.

- [ ] **Step 3: Move the partial and add the marker**

```bash
mkdir -p docs/_partials
git mv docs/_intro.md docs/_partials/intro.md
```

In `docs/_partials/intro.md`, replace the Methods card body (currently line 64) with:

```markdown
    <!-- METHODCOUNT --> road-generation methods, each shown on the ground with its own numbers.
```

Update that file's header comment: it currently says "writes docs/index.md — edit HERE, never
index.md". Add `METHODCOUNT` to its list of markers (`HEROLOGO`, `KEYRESULT`, `HERO`, `KEYFIGURES`)
and change the path reference from `docs/_intro.md` to `docs/_partials/intro.md`.

- [ ] **Step 4: Add the producer map and `_render_partial`**

In `scripts/gen_site_pages.py`, add after `_key_figures()`:

```python
_COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
                7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}


def _method_count() -> str:
    """The number of PUBLISHED methods, as an English word. Generated because typing it into prose
    is exactly how docs/_intro.md came to claim "Seven" while ten methods shipped."""
    n = sum(1 for m in METHODS if m.published)
    return _COUNT_WORDS.get(n, str(n))


# Marker name -> producer. A partial's <!-- NAME --> holes are filled from here; tests assert this
# mapping and the markers actually used in docs/_partials/ are the same set, in both directions.
MARKERS: dict[str, Callable[[], str]] = {
    "HEROLOGO": _hero_logo,
    "HERO": _hero_block,
    "KEYRESULT": _key_result,
    "KEYFIGURES": _key_figures,
    "METHODCOUNT": _method_count,
}

PARTIALS = DOCS / "_partials"


def _render_partial(name: str) -> str:
    """Read docs/_partials/<name>.md and fill every known marker. Unknown markers are left alone so
    the closed-set test can catch them; a missing partial is an error, not an empty page."""
    text = (PARTIALS / f"{name}.md").read_text(encoding="utf-8").rstrip()
    for marker, produce in MARKERS.items():
        text = text.replace(f"<!-- {marker} -->", produce())
    return text
```

The `Callable` import already exists at the top of the file.

- [ ] **Step 5: Use it in `main()`**

Replace the `intro_path` / `home` / marker-loop block in `main()` with:

```python
    (DOCS / "index.md").write_text(GENERATED_NOTE + _render_partial("intro") + "\n",
                                   encoding="utf-8")
```

- [ ] **Step 6: Update `mkdocs.yml` and `.gitignore`**

In `mkdocs.yml`, replace the `_intro.md` line in `exclude_docs` with `_partials/`, and update the
surrounding comment: partials are the handwritten prose the generator folds artifacts into; one
directory exclusion replaces a per-file list that would go stale as partials are added.

In `.gitignore`, confirm `docs/index.md` is still ignored and that `docs/_partials/` is **not**.

- [ ] **Step 7: Run the generator and the tests**

Run: `python3 scripts/gen_site_pages.py && pixi run pytest tests/test_gen_site_pages.py -v`
Expected: generator prints its wrote-line; all tests PASS.
Then confirm the count actually rendered: `grep -c "Ten road-generation" docs/index.md` → `1`.

- [ ] **Step 8: Fault-inject the guards**

Temporarily add `<!-- NOSUCHMARKER -->` to `docs/_partials/intro.md`.
Run: `pixi run pytest tests/test_gen_site_pages.py::test_every_marker_in_a_partial_has_a_producer -v`
Expected: FAIL listing `['NOSUCHMARKER']`. Remove it.

Temporarily set `published=False` on one published method in `METHODS`.
Run: `pixi run pytest tests/test_gen_site_pages.py::test_published_method_count_is_generated_not_typed -v`
Expected: FAIL — "expected 10 published methods, registry says 9". Restore.

- [ ] **Step 9: Commit**

```bash
git add docs/_partials/ scripts/gen_site_pages.py mkdocs.yml .gitignore tests/test_gen_site_pages.py
git commit -m "site: partials mechanism, and the method count stops being typed prose"
```

---

### Task 2: Guard the unpublished-method / exclude_docs pair

`mkdocs.yml:33` excludes `methods/dream_come_true.md` and its own comment warns that if the key
stops matching, the unpublished method **silently reappears as an un-navigable orphan**. The `M`
class docstring says the same: "THE TWO MUST BE KEPT IN SYNC". Nothing enforces it, and Task 7
changes exactly that path.

**Files:**
- Test: `tests/test_gen_site_pages.py`

**Interfaces:**
- Consumes: `_site_methods()` (already in the test module).
- Produces: nothing consumed later; this is a standing guard.

- [ ] **Step 1: Write the failing test**

```python
def test_unpublished_methods_are_excluded_from_the_build() -> None:
    """published=False keeps a method out of the overview, but exclude_docs is the actual publish
    switch. If they drift, the page is BUILT and reachable by URL while linked from nowhere."""
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text()
    unpublished = set(re.findall(r'M\("([a-z_]+)"[^)]*?published=False', src, flags=re.S))
    assert unpublished, "no unpublished methods found; this guard would be vacuous"
    nav_text = (ROOT / "mkdocs.yml").read_text()
    excluded = set(re.findall(r"^\s*\S*methods/([a-z_]+)\.md\s*$", nav_text, flags=re.M))
    leaked = sorted(unpublished - excluded)
    assert not leaked, f"published=False but not in exclude_docs, so still built: {leaked}"
```

Note the `assert unpublished` line: without it, deleting `dream_come_true` would make this test pass
by testing nothing.

- [ ] **Step 2: Run to verify it passes on current state**

Run: `pixi run pytest tests/test_gen_site_pages.py::test_unpublished_methods_are_excluded_from_the_build -v`
Expected: PASS (`dream_come_true` is both `published=False` and excluded).

- [ ] **Step 3: Fault-inject — the drift it exists to catch**

Comment out the `methods/dream_come_true.md` line in `mkdocs.yml`'s `exclude_docs`.
Run the test again.
Expected: FAIL — `published=False but not in exclude_docs, so still built: ['dream_come_true']`.
Restore the line.

- [ ] **Step 4: Fault-inject the vacuity guard**

Temporarily delete `published=False` from the `dream_come_true` entry in `METHODS`.
Run the test again.
Expected: FAIL — "no unpublished methods found; this guard would be vacuous". Restore.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gen_site_pages.py
git commit -m "test: guard the published=False / exclude_docs pair before the renest moves it"
```

---

### Task 3: `mkdocs build --strict` in CI

Every remaining task moves pages and rewrites links. `--strict` turns MkDocs' broken-link warnings
into build failures. Verified: the current site builds clean under `--strict`, so this can land now
and protect everything after it.

**Files:**
- Modify: `.github/workflows/deploy-site.yml`
- Modify: `mkdocs.yml` (comment at the top documenting the local preview command)

**Interfaces:**
- Produces: a CI gate the remaining tasks rely on. No code interface.

- [ ] **Step 1: Verify the clean baseline**

Run:
```bash
python3 scripts/gen_site_pages.py
pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict
```
Expected: exit 0, no `WARNING` lines.

- [ ] **Step 2: Fault-inject**

Add `[dead](does-not-exist.md)` to `docs/background.md`. Re-run the build command.
Expected: FAIL — MkDocs reports an unrecognised relative link and `--strict` aborts.
Remove the line and confirm it builds clean again.

- [ ] **Step 3: Add the flag**

In `.github/workflows/deploy-site.yml`, change `- run: mkdocs build` to:

```yaml
      # --strict promotes MkDocs' broken-link and orphan-page warnings to build failures. The site's
      # nav is explicit and its cross-page links are hand-written, so a moved page otherwise ships
      # a dead link silently.
      - run: mkdocs build --strict
```

- [ ] **Step 4: Update the local-preview comment**

At the top of `mkdocs.yml`, change the `mkdocs serve` note to mention `mkdocs build --strict` as the
check CI runs.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/deploy-site.yml mkdocs.yml
git commit -m "ci: build the site --strict, so a moved page fails instead of shipping a dead link"
```

---

### Task 4: Fix the screen attribution in Background (defect 2)

**Files:**
- Modify: `docs/background.md:35-40`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

`docs/background.md:37-40` currently reads:

> This project instead screens with a cheap **density × compactness** heuristic (`n/P²`) that scores
> every block in an entire metro in a single fast sweep, and which peaks in the **Khayelitsha**
> informal settlement of Cape Town — the region the [Results](benchmark.md) benchmark grows and
> reblocks.

Two problems: the metric is retired, and the Khayelitsha claim was **measured under the retired
metric**. Do not paraphrase the old claim into new words — verify it.

- [ ] **Step 1: Verify where the shipped metric actually peaks**

Run:
```bash
pixi run python -m scripts.gen_screen_bakeoff
```

This downloads the ~18 MB ground-truth layer once and writes
`examples/screen-bakeoff/screen_comparison.csv` plus its maps. Then inspect which settlement holds
the top-ranked blocks under `depth_density_proxy`, using the settlement clustering in
`reblock.data.informal` (189 settlements). If the script does not already report this, add a
`--top-settlement` print to it that names the settlement containing the highest-scoring block under
each metric.

- [ ] **Step 2: Write the sentence to match what Step 1 reported**

Replace the paragraph with prose that:
1. names `depth_density_proxy` (`√(nA)/P · n/A`) as the screen — deep **and** crowded, from the free
   kblock columns, no Voronoi and no peel;
2. keeps the contrast with Soman et al. intact (their per-block Voronoi tessellation is
   prohibitively expensive at metro scale; this is a single cheap sweep);
3. states where it peaks **according to Step 1's output** — Khayelitsha if that is what it says, the
   actual settlement if not;
4. links to the new bake-off page for the evidence: `[Results](results/bakeoff.md)`.

Do not state precision/recall figures here; those belong on the bake-off page where they are
generated. Background is a committed handwritten page with no marker mechanism, so it must contain
**no numbers at all** beyond the `n/P²`-free formula name.

- [ ] **Step 3: Verify no stale attribution survives**

Run: `grep -rn "density × compactness\|n/P²" docs/background.md`
Expected: either no output, or only a clause explicitly describing it as the *previous* screen.

- [ ] **Step 4: Build strictly**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0. (The `results/bakeoff.md` link will fail until Task 9 — if so, leave the link as
`benchmark.md` in this task and change it in Task 12's link audit.)

- [ ] **Step 5: Commit**

```bash
git add docs/background.md examples/screen-bakeoff/
git commit -m "docs: background credits the shipped screen, and the peak claim is re-measured"
```

---

### Task 5: Methodology index and Screening (defect 1)

**Files:**
- Create: `docs/_partials/methodology.md`, `docs/_partials/screening.md`
- Modify: `scripts/gen_site_pages.py` (`_screen_table` producer; `main()` writes the pages)
- Modify: `mkdocs.yml` (nav)
- Delete: `docs/methodology.md`

**Interfaces:**
- Consumes: `_render_partial`, `MARKERS` (Task 1).
- Produces: `_screen_table() -> str` — a markdown table of every metric in
  `examples/screen-bakeoff/screen_comparison.csv` with AUC and prec@1%, plus each shipped absolute
  floor read from `conf/metric/*.yaml`. Registered as marker `SCREENTABLE`.

- [ ] **Step 1: Write the producer**

In `scripts/gen_site_pages.py`:

```python
BAKEOFF = ROOT / "examples" / "screen-bakeoff"


def _screen_floors() -> dict[str, str]:
    """Absolute gate floors, read from conf/metric/*.yaml. Regex, not a YAML parse: this script is
    stdlib-only and the line shape is fixed (`metric_gate: {..., kind: absolute, value: X}`)."""
    out: dict[str, str] = {}
    for path in sorted((ROOT / "conf" / "metric").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"kind:\s*absolute,\s*value:\s*([0-9.eE+-]+)", text)
        if m:
            out[path.stem] = m.group(1)
    return out


def _screen_table() -> str:
    """The screen bake-off ranking. Empty string when the artifact is absent, per the dir-reader
    contract: a section is emitted only when its data exists."""
    rows = _read_csv(BAKEOFF / "screen_comparison.csv")
    if not rows:
        return ""
    floors = _screen_floors()
    body = [["metric", "AUC", "precision in top 1%", "shipped floor"]]
    for r in rows:
        key = r["metric"].split()[0]
        body.append([r["metric"], _num(float(r["auc"]), 3), _pct(float(r["prec@1%"])),
                     floors.get(key, "—")])
    return _table(body[0], body[1:])
```

`re` is already imported. Register `"SCREENTABLE": _screen_table` in `MARKERS`.

- [ ] **Step 2: Write `docs/_partials/methodology.md`**

Required content:
- The pipeline, `data → screen → region_builder → method → eval → render`, one short paragraph per
  stage, each linking to its section page.
- A **glossary** listing every term the four sections define, each linking to its full treatment:
  density, compactness, the depth proxy `√(nA)/P`, permeability, displacement.
- No numbers.

- [ ] **Step 3: Write `docs/_partials/screening.md`**

Required content, in order:

1. **What screening is for** — one cheap score over every block in a metro, so the expensive stage
   runs only on survivors.
2. **The shipped screen**, stated correctly: `depth_density_proxy` = `√(nA)/P · n/A` — the depth
   proxy times density. Deep **and** crowded. Computable from the free kblock columns
   (`building_count`, area, perimeter) — no Voronoi, no peel.
3. `<!-- SCREENTABLE -->`
4. **Why `n/P²` was retired**, stated as a superseded choice rather than an omission: it is beaten
   on precision *and* recall at equal pool size, and its own selling point (needing no peel) is not
   a differentiator because the replacement needs none either. Link to
   `../results/bakeoff.md` for the evidence.
5. **The gate is absolute, not a percentile** — a percentile redefines the population every time the
   corpus changes. (This reasoning is in `conf/metric/density_compactness.yaml`'s comment; restate
   it, do not quote the file.)
6. **From block to region** — `conf/region_builder/` (`dense_cluster`, `convex_hull`,
   `shape_standardizing`, `identity`), why a region rather than a block: proposed roads stay
   continuous across block boundaries instead of stopping at the edge of each separately-solved
   block. This is the hinge where the pipeline goes from whole-city-cheap to per-block-expensive.

**No typed numbers.** The floor and the precision figures come from `SCREENTABLE`.

- [ ] **Step 4: Wire both pages into `main()`**

```python
    methodology_dir = DOCS / "methodology"
    if methodology_dir.exists():
        shutil.rmtree(methodology_dir)
    methodology_dir.mkdir(parents=True, exist_ok=True)
    _write_page(methodology_dir / "index.md", _render_partial("methodology"),
                depth=1, url_depth=1, title="Methodology")
    _write_page(methodology_dir / "screening.md", _render_partial("screening"),
                depth=1, url_depth=2, title="Screening")
```

- [ ] **Step 5: Update nav and delete the old page**

In `mkdocs.yml`, replace `- Methodology: methodology.md` with:

```yaml
  - Methodology:
      - methodology/index.md
      - methodology/screening.md
```

(The remaining section pages are added in Task 6, the Methods subsection in Task 7.)

```bash
git rm docs/methodology.md
```

Add `docs/methodology/` to `.gitignore`.

- [ ] **Step 6: Generate and build**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0.
Then: `grep -n "0.0128" docs/methodology/screening.md` → the floor appears, from the config, not typed.

- [ ] **Step 7: Verify the defect is gone**

Run: `grep -rn "screening heuristic" docs/_partials/ docs/methodology/`
Expected: no line labelling `n/P²` as the screening heuristic.

- [ ] **Step 8: Commit**

```bash
git add docs/_partials/ scripts/gen_site_pages.py mkdocs.yml .gitignore
git commit -m "docs: methodology index + screening -- the shipped screen, with its floor generated"
```

---

### Task 6: Permeability and Displacement (defect 3)

**Files:**
- Create: `docs/_partials/permeability.md`, `docs/_partials/displacement.md`
- Modify: `scripts/gen_site_pages.py` (`main()`)
- Modify: `mkdocs.yml` (nav)

**Interfaces:**
- Consumes: `_render_partial` (Task 1).
- Produces: two pages. No new markers — both pages describe **models**, and their only numbers are
  symbolic parameters, which are written as symbols (`g_walk`, `P*`), not values.

- [ ] **Step 1: Write `docs/_partials/permeability.md`**

Source of truth for the content: `src/reblock/permeability.py`'s module docstring. Required content:

1. **The model.** Every parcel injects one unit of escape current; the existing street is ground at
   potential 0. The metric is total dissipated power `P = bᵀL⁻¹b` with `b` ones over parcels,
   reported as `permeability = 1 − P(roads)/P(no roads)`. Lower dissipated power means easier
   collective egress.
2. **Why it is monotone by construction.** Roads only ever *add* conductance, and a road-covered
   edge takes `max(footpath, road)` — so an upgrade can never lower an edge, for any edge and any
   region. Permeability is therefore monotone non-decreasing in the road set with no clamp needed.
   This is what makes the prefix curves and the target search valid.
3. **The graph.** Nodes are parcel centroids. Footpath mesh edges join adjacent parcels with
   conductance proportional to the **clearance fraction** — the share of the centroid-to-centroid
   line lying in neither building — so the estimate is local rather than a single block-wide
   corridor width. Ground edges attach parcels within tolerance of the street and are folded into
   the Laplacian diagonal; ground is eliminated, never a node.
4. **A placeholder for the figure**, as an HTML comment noting that piece B of the redesign supplies
   the rendered graph. Do **not** add a `data-widget` mount point — that is piece C.

- [ ] **Step 2: Write `docs/_partials/displacement.md`**

This page carries **defect 3**, so state the corrected definition exactly:

> **Displacement** is the expected number of **buildings** a road set displaces — not parcels.
> Each building is a disk of radius `rᵢ`, half its nearest-neighbour distance. Its contribution is
> the probability the road corridor grazes it under a uniform size prior,
> `cᵢ = max(0, 1 − dᵢ/rᵢ)`, where `dᵢ` is the distance from the building to the corridor.
> Displacement is `Σcᵢ`; the reported fraction divides by the number of buildings.

Then:
1. **Width is per-road.** Each road is buffered by its *own* `width_m`/2, so a narrow lane costs less
   corridor than a wide street. There is no global corridor width.
2. **Overlap is free, by construction.** The buffers are **unioned**, so two coincident opposing
   lanes occupy one corridor and are charged once, while separating them widens the union and costs
   more. No separate gap rule is needed and none exists.
3. **Parcels are not buildings.** `Block.parcels` and `Block.building_points` are distinct; the cost
   axis is counted over buildings.

- [ ] **Step 3: Wire into `main()`**

```python
    _write_page(methodology_dir / "permeability.md", _render_partial("permeability"),
                depth=1, url_depth=2, title="Permeability")
    _write_page(methodology_dir / "displacement.md", _render_partial("displacement"),
                depth=1, url_depth=2, title="Displacement")
```

- [ ] **Step 4: Add to nav**

```yaml
      - methodology/permeability.md
      - methodology/displacement.md
```

- [ ] **Step 5: Verify the defect is gone**

Run: `grep -rn "parcels a road set displaces" docs/`
Expected: no output.
Run: `grep -n "buildings" docs/methodology/displacement.md`
Expected: the corrected definition present.

- [ ] **Step 6: Generate and build**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0.

- [ ] **Step 7: Commit**

```bash
git add docs/_partials/ scripts/gen_site_pages.py mkdocs.yml
git commit -m "docs: permeability and displacement sections -- displacement is over buildings"
```

---

### Task 7: Renest the methods pages

The riskiest mechanical change: `_write_page`'s `depth`/`url_depth` both shift, and the generator
emits one hard-coded relative link that breaks.

**Files:**
- Modify: `scripts/gen_site_pages.py:478` (the `../benchmark.md` link), `main()`
- Modify: `mkdocs.yml` (nav paths, `exclude_docs` path)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `METHODS`, `_write_page` (existing).
- Produces: pages at `docs/methodology/methods/`, served at `<base>/methodology/methods/<slug>/`.

- [ ] **Step 1: Change the output directory and depths in `main()`**

Replace the `methods_dir` block:

```python
    methods_dir = methodology_dir / "methods"
    methods_dir.mkdir(parents=True, exist_ok=True)
    # methodology/methods/index.md serves at <base>/methodology/methods/ (2 segments);
    # methodology/methods/peel.md at <base>/methodology/methods/peel/ (3). Source depth is 2 for
    # both. Getting url_depth wrong 404s every figure on the page.
    _write_page(methods_dir / "index.md", gen_methods_overview(), depth=2, url_depth=2,
                title="The methods")
    for m in METHODS:
        _write_page(methods_dir / f"{m.slug}.md", gen_method_section(m), depth=2, url_depth=3,
                    title=m.display_title)
```

Delete the old `shutil.rmtree(DOCS / "methods")` block — Task 5's `rmtree(methodology_dir)` already
clears it, and `docs/methods/` no longer exists.

- [ ] **Step 2: Fix the hard-coded cross-page link**

`scripts/gen_site_pages.py:478` emits `[Cape Town benchmark](../benchmark.md)` into every method
page. From `methodology/methods/<slug>.md` the Results page is now two levels up. Task 8 moves it to
`results/frontier.md`; write the final target now:

```python
    parts.append("From the [Cape Town benchmark](../../results/frontier.md) — the 12-block, "
```

This link is dead until Task 8 lands. Run Tasks 7 and 8 back to back, and do not run `--strict`
between them.

- [ ] **Step 3: Update nav**

Replace the whole `- Methods:` block with a nested subsection under Methodology, keeping the existing
comment about paths-only entries (labels come from the generated front matter) and about the list
being manual:

```yaml
      - Methods:
          - methodology/methods/index.md
          - methodology/methods/peel.md
          - methodology/methods/clearance.md
          - methodology/methods/clearance_looped.md
          - methodology/methods/greedy_arterial_buildable.md
          - methodology/methods/cycle_native.md
          - methodology/methods/topology.md
          - methodology/methods/resistance_lp.md
          - methodology/methods/greedy_arterial_access_displacement.md
          - methodology/methods/osm_footpaths.md
          - methodology/methods/euclidean_grid.md
```

- [ ] **Step 4: Update `exclude_docs` — the key Task 2 guards**

Change `methods/dream_come_true.md` to `methodology/methods/dream_come_true.md`. Keep the
STALE-KEY warning comment; add a line noting the guard test now enforces it.

- [ ] **Step 5: Update `.gitignore`**

Remove `docs/methods/` if present; `docs/methodology/` (Task 5) already covers the new location.

- [ ] **Step 6: Verify the guard catches a mismatch**

Before running the generator, temporarily leave `exclude_docs` at the **old** path.
Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: `test_unpublished_methods_are_excluded_from_the_build` still passes — the regex matches the
slug regardless of prefix — but `test_every_published_site_method_is_in_the_mkdocs_nav` passes too.
This is the known limit of a text-regex guard. Instead verify by build:
run the generator and `mkdocs build --strict`, and confirm `site/methodology/methods/dream_come_true/`
**exists** when the exclude path is stale and **does not exist** once it is corrected.

```bash
python3 scripts/gen_site_pages.py
pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build
ls site/methodology/methods/ | grep dream || echo "correctly excluded"
```

- [ ] **Step 7: Verify figures resolve**

Run: `grep -o 'src="[^"]*assets[^"]*"' docs/methodology/methods/peel.md | head -3`
Expected: paths beginning `src="../../../assets/` (url_depth 3).

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_site_pages.py mkdocs.yml .gitignore
git commit -m "site: renest the method pages under methodology/, with their depths and links"
```

---

### Task 8: `benchmark.md` → `results/frontier.md`

**Files:**
- Modify: `scripts/gen_site_pages.py` (`main()`)
- Modify: `mkdocs.yml` (nav)
- Modify: `docs/background.md` (its `benchmark.md` link)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `gen_benchmark_section()` (existing, unchanged).
- Produces: `docs/results/frontier.md`, served at `<base>/results/frontier/`.

- [ ] **Step 1: Write to the new location**

In `main()`, replace the `benchmark.md` line:

```python
    results_dir = DOCS / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # results/frontier.md serves at <base>/results/frontier/ -- source depth 1, url_depth 2.
    _write_page(results_dir / "frontier.md", gen_benchmark_section(), depth=1, url_depth=2,
                title="Frontier benchmark")
```

Update the closing `print(...)` to name the pages actually written.

- [ ] **Step 2: Update nav**

```yaml
  - Results:
      - results/frontier.md
```

- [ ] **Step 3: Fix the inbound link**

In `docs/background.md`, change `[Results](benchmark.md)` to `[Results](results/frontier.md)`.

- [ ] **Step 4: Update `.gitignore`**

Replace `docs/benchmark.md` with `docs/results/`.

- [ ] **Step 5: Generate and build strictly**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0 — this is the first strict build since Task 7, so it also validates the
`../../results/frontier.md` link emitted into every method page.

- [ ] **Step 6: Verify no stale path survives**

Run: `grep -rn "benchmark\.md" docs/ scripts/ mkdocs.yml --include=*.md --include=*.py --include=*.yml`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_site_pages.py mkdocs.yml docs/background.md .gitignore
git commit -m "site: benchmark.md becomes results/frontier.md, with every inbound link"
```

---

### Task 9: The screen bake-off Results page

The strongest evidence artifact in the repo, currently invisible on the site.

**Files:**
- Create: `docs/_partials/bakeoff.md`
- Modify: `scripts/gen_site_pages.py` (`_bakeoff_figures`, `_bakeoff_floors_table` producers; `main()`)
- Modify: `mkdocs.yml` (nav)

**Interfaces:**
- Consumes: `_render_partial`, `_screen_table`, `_copy_asset`, `_figure`, `_table` (existing).
- Produces:
  - `_bakeoff_figures() -> str` — the three committed PNGs (`precision_recall.png`, `city_map.png`,
    `settlements.png`), each captioned, copied into `docs/assets/bakeoff/`. Marker `BAKEOFFFIGS`.
  - `_bakeoff_floors_table() -> str` — the two shipped screens at their absolute floors, with pool
    size, precision and recall from `screen_comparison.csv`. Marker `BAKEOFFFLOORS`.

- [ ] **Step 1: Write the producers**

```python
def _bakeoff_figures() -> str:
    """The three committed bake-off figures, in narrative order. Each is emitted only if present."""
    wanted = [("precision_recall.png", "Precision and recall across the candidate screens."),
              ("city_map.png", "Where the shipped screen and its predecessor disagree, city-wide."),
              ("settlements.png",
               "The four settlements of sharpest disagreement. Green sits inside the gold "
               "settlement outlines; red sits outside them.")]
    out: list[str] = []
    for name, caption in wanted:
        url = _copy_asset(BAKEOFF / name, "bakeoff")
        if url:
            out.append(_figure(url, caption, caption))
    return "\n\n".join(out)


def _bakeoff_floors_table() -> str:
    """Both shipped screens at their absolute floors: pool size, precision, recall."""
    rows = [r for r in _read_csv(BAKEOFF / "screen_comparison.csv") if r.get("floor")]
    if not rows:
        return ""
    body = [[r["metric"], _num(float(r["floor_n"])), _pct(float(r["floor_prec"])),
             _pct(float(r["floor_recall"]))] for r in rows]
    return _table(["screen at its floor", "blocks", "precision", "recall"], body)
```

Register both in `MARKERS`.

- [ ] **Step 2: Write `docs/_partials/bakeoff.md`**

Required content:

1. **What this grades.** Every other result grades reblocking *methods*; this grades the **screen**
   that decides which blocks get reblocked at all — a stage that went unvalidated until 2026-08-08.
2. **The ground truth.** The City of Cape Town's own informal-structure survey, digitised from
   February 2018 aerial photography at 1:200, published via University of Edinburgh DataShare
   ([doi:10.7488/ds/2758](https://doi.org/10.7488/ds/2758)). The file carries no settlement-name
   field, so extents are clustered from the structures themselves. A block counts as informal when
   at least 30% of its area falls inside one.
   **All counts stay out of this prose** — see step 3.
3. `<!-- SCREENTABLE -->` — the ranking.
4. `<!-- BAKEOFFFLOORS -->` — the two shipped screens head to head.
5. **The honest headline.** Even the better screen is right about one block in four; the *previous*
   default selected three non-settlement blocks for every settlement one. Screening is hard, and the
   page should say so rather than bury it.
6. `<!-- BAKEOFFFIGS -->`
7. **Caveats**, carried over from `examples/screen-bakeoff/README.md`: Cape Town only (no equivalent
   published layer exists for Nairobi); the 30% cover threshold is a choice, though the metric
   *ordering* was verified stable from 10% to 90%; 2018 structures against blocks built from later
   data; and a single Open Buildings feature beats every metric here at the cost of a polygon
   download this screen does not need.

- [ ] **Step 3: Add the survey counts as a marker, not prose**

The dwelling count, settlement count, and informal-block count are numbers, so they cannot be typed.
Add:

```python
def _bakeoff_scale() -> str:
    """Survey scale, from the ground-truth artifact rather than prose. Returns an empty string when
    the artifact is absent, so the sentence around it must read naturally without it."""
    path = BAKEOFF / "ground_truth.json"
    if not path.exists():
        return ""
    g = json.loads(path.read_text(encoding="utf-8"))
    return (f"{_num(g['structures'])} dwelling polygons, clustered into {_num(g['settlements'])} "
            f"settlements, marking {_num(g['informal_blocks'])} of {_num(g['total_blocks'])} "
            f"Cape Town blocks informal")
```

Register as `BAKEOFFSCALE`. This requires `scripts/gen_screen_bakeoff.py` to emit
`examples/screen-bakeoff/ground_truth.json` with those four keys — add that write alongside its
existing `screen_comparison.csv` write, and regenerate:

```bash
pixi run python -m scripts.gen_screen_bakeoff
```

- [ ] **Step 4: Wire into `main()` and nav**

```python
    _write_page(results_dir / "bakeoff.md", _render_partial("bakeoff"),
                depth=1, url_depth=2, title="Screen bake-off")
```

```yaml
      - results/bakeoff.md
```

- [ ] **Step 5: Generate, build, verify**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0.
Run: `grep -c "117,336\|0.817" docs/results/bakeoff.md`
Expected: at least 1 — the numbers rendered from artifacts.
Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: PASS — the marker/producer closed-set tests cover the four new markers.

- [ ] **Step 6: Commit**

```bash
git add docs/_partials/bakeoff.md scripts/gen_site_pages.py scripts/gen_screen_bakeoff.py \
        examples/screen-bakeoff/ mkdocs.yml
git commit -m "docs: publish the screen bake-off, with its scale read from the ground truth"
```

---

### Task 10: The Nairobi Results page

**Files:**
- Create: `docs/_partials/nairobi.md`
- Modify: `scripts/gen_site_pages.py` (`_nairobi_table` producer; `main()`)
- Modify: `mkdocs.yml` (nav)

**Interfaces:**
- Consumes: `_render_partial`, `_read_csv`, `_table`, `_copy_asset`, `_figure`.
- Produces: `_nairobi_table() -> str` — one row per variant directory under `examples/nairobi/`,
  reading `meta.json` for the metric name and region size and `lens_permeability.csv` for whether an
  `osm_footpaths` baseline exists. Marker `NAIROBITABLE`.

- [ ] **Step 1: Write the producer**

```python
NAIROBI = ROOT / "examples" / "nairobi"


def _nairobi_table() -> str:
    """One row per Nairobi variant, read from its own artifacts. Absent variants simply do not
    appear -- a partial checkout yields a shorter table, never a placeholder row."""
    rows: list[list[str]] = []
    for d in sorted(p for p in NAIROBI.iterdir() if p.is_dir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        methods = {r["method"] for r in _read_csv(d / "lens_permeability.csv")}
        rows.append([f"[{d.name}](https://github.com/jmendoza167/REU_Slum_Optimization/"
                     f"tree/main/examples/nairobi/{d.name})",
                     str(meta.get("metric", "—")),
                     _num(meta.get("region_members", 0)),
                     "yes" if "osm_footpaths" in methods else "—"])
    if not rows:
        return ""
    return _table(["variant", "metric", "blocks in region", "OSM baseline"], rows)
```

- [ ] **Step 2: Write `docs/_partials/nairobi.md`**

Required content:

1. **The same pipeline, a second country.** Kenya kblock data clipped to the Nairobi metro bbox plus
   Open Buildings (`data=nairobi_full`), the same three composable `BlockMetric` variants.
2. `<!-- NAIROBITABLE -->`
3. **Shipped as-is, and why that is the honest framing.** Two things do not transfer from Cape Town:
   **region sizes** (Nairobi's blocks are bimodal — giant and tiny — so a Cape-Town-tuned building
   budget yields wildly different regions, and no single budget fixes both), and **OSM coverage**
   (the `density_compactness` region has essentially no mapped footpaths, so its frontier grades
   only the synthesized methods). The screens and metric behaviour carry over cleanly; the
   region-growth tuning and OSM coverage are what differ.
4. **No ground truth exists here.** Link to `bakeoff.md` and state plainly that the precision/recall
   validation is Cape Town only.

- [ ] **Step 3: Wire into `main()` and nav**

```python
    _write_page(results_dir / "nairobi.md", _render_partial("nairobi"),
                depth=1, url_depth=2, title="Second city: Nairobi")
```

```yaml
      - results/nairobi.md
```

- [ ] **Step 4: Generate, build, verify**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0.
Run: `grep -c "multiblock_depth" docs/results/nairobi.md`
Expected: at least 1.

- [ ] **Step 5: Commit**

```bash
git add docs/_partials/nairobi.md scripts/gen_site_pages.py mkdocs.yml
git commit -m "docs: publish the Nairobi second-city results"
```

---

### Task 11: The Reproduce page

**Files:**
- Create: `docs/_partials/reproduce.md`
- Modify: `scripts/gen_site_pages.py` (`_repro_commands` producer; `main()`)
- Modify: `mkdocs.yml` (nav)

**Interfaces:**
- Consumes: `_render_partial`, `_table`.
- Produces: `_repro_commands() -> str` — a table of every example flagship and the exact command
  that regenerates it, read from each `meta.json`'s `command` field. Marker `REPROCOMMANDS`.

- [ ] **Step 1: Write the producer**

```python
EXAMPLES = ROOT / "examples"


def _repro_commands() -> str:
    """Every flagship's regeneration command, read from its own meta.json. Never typed here: a
    command that drifts from the artifact it claims to produce is the same defect as a stale
    number."""
    rows: list[list[str]] = []
    for meta_path in sorted(EXAMPLES.glob("*/meta.json")) + sorted(
            EXAMPLES.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cmd = meta.get("command")
        if not cmd:
            continue
        rows.append([meta_path.parent.relative_to(EXAMPLES).as_posix(), f"`{cmd}`"])
    if not rows:
        return ""
    return _table(["flagship", "command"], rows)
```

- [ ] **Step 2: Write `docs/_partials/reproduce.md`**

Required content:

1. **Setup** — `git clone --recurse-submodules`, install pixi, `pixi install`. (Copy the shape from
   `README.md`'s Setup section; these are commands, not numbers, so they may be written inline.)
2. **One block, no whole-city pass** — the `block_ids` quickstart, with the note that the brackets
   must be quoted so the shell does not glob them.
3. **A whole city** — the `capetown_full` screen-and-reblock command, noting that the first run
   downloads and caches the metro under `~/.cache/reblock` and later runs are instant.
4. `<!-- REPROCOMMANDS -->` — every figure on this site, and the command behind it.
5. **How the site itself rebuilds** — `python3 scripts/gen_site_pages.py` then `mkdocs build
   --strict`, and the note that generated pages are not committed.

- [ ] **Step 3: Wire into `main()` and nav**

```python
    # reproduce.md serves at <base>/reproduce/ -- source depth 0, url_depth 1.
    _write_page(DOCS / "reproduce.md", _render_partial("reproduce"), depth=0, url_depth=1,
                title="Reproduce")
```

```yaml
  - Reproduce: reproduce.md
```

Add `docs/reproduce.md` to `.gitignore`.

- [ ] **Step 4: Generate, build, verify**

Run: `python3 scripts/gen_site_pages.py && pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict`
Expected: exit 0.
Run: `grep -c "gen_example" docs/reproduce.md`
Expected: at least 1 — commands came from `meta.json`.

- [ ] **Step 5: Commit**

```bash
git add docs/_partials/reproduce.md scripts/gen_site_pages.py mkdocs.yml .gitignore
git commit -m "docs: a Reproduce page whose commands are read from the artifacts they build"
```

---

### Task 12: Home cards, final nav, and the whole-site link audit

**Files:**
- Modify: `docs/_partials/intro.md` (the five cards)
- Modify: `mkdocs.yml` (final nav order)
- Modify: `tests/test_gen_site_pages.py`

**Interfaces:**
- Consumes: every page created above.
- Produces: nothing.

- [ ] **Step 1: Rewrite the Home cards**

In `docs/_partials/intro.md`, the "Start here" card grid currently links to `background.md`,
`methodology.md`, `methods/index.md`, `benchmark.md`, `team.md`. Three of those paths no longer
exist. Replace with cards for: Background, Methodology, Methods
(`methodology/methods/index.md`), Results (`results/frontier.md`), Reproduce, Team & References.
Keep the `<!-- METHODCOUNT -->` marker in the Methods card.

Also update the closing line — it currently claims every number is machine-generated from artifacts
"committed in the repository", which is now true of more pages than it was; leave the claim, it is
still accurate.

- [ ] **Step 2: Set the final nav**

```yaml
nav:
  - Home: index.md
  - Background: background.md
  - Methodology:
      - methodology/index.md
      - methodology/screening.md
      - methodology/permeability.md
      - methodology/displacement.md
      - Methods:
          - methodology/methods/index.md
          - methodology/methods/peel.md
          - methodology/methods/clearance.md
          - methodology/methods/clearance_looped.md
          - methodology/methods/greedy_arterial_buildable.md
          - methodology/methods/cycle_native.md
          - methodology/methods/topology.md
          - methodology/methods/resistance_lp.md
          - methodology/methods/greedy_arterial_access_displacement.md
          - methodology/methods/osm_footpaths.md
          - methodology/methods/euclidean_grid.md
  - Results:
      - results/frontier.md
      - results/bakeoff.md
      - results/nairobi.md
  - Reproduce: reproduce.md
  - Team & References: team.md
```

No `Explore` entry — that page arrives with piece E.

- [ ] **Step 3: Add a link-integrity test**

```python
def test_no_partial_links_to_a_retired_path() -> None:
    """methodology.md and benchmark.md are gone; a link to either 404s. mkdocs --strict catches
    this at build time, but only in CI -- this fails in the unit suite."""
    retired = ("](methodology.md)", "](benchmark.md)", "](methods/index.md)")
    offenders: list[str] = []
    for name, text in _partials().items():
        for path in retired:
            if path in text:
                offenders.append(f"{name}: {path}")
    assert not offenders, f"links to retired paths: {offenders}"
```

- [ ] **Step 4: Run the test and fault-inject**

Run: `pixi run pytest tests/test_gen_site_pages.py -v`
Expected: all PASS.
Add `[x](benchmark.md)` to `docs/_partials/intro.md`, re-run.
Expected: FAIL listing `intro.md: ](benchmark.md)`. Remove it.

- [ ] **Step 5: Full clean build**

```bash
rm -rf site docs/index.md docs/methodology docs/results docs/reproduce.md
python3 scripts/gen_site_pages.py
pixi exec --spec mkdocs-material==9.7.7 --spec mkdocs-glightbox==0.5.2 -- mkdocs build --strict
```
Expected: exit 0, zero warnings.

- [ ] **Step 6: Verify all four defects are dead**

```bash
grep -rn "screening heuristic" docs/ || echo "defect 1 clear"
grep -rn "density × compactness.*heuristic" docs/background.md || echo "defect 2 clear"
grep -rn "parcels a road set displaces" docs/ || echo "defect 3 clear"
grep -rn "Seven road-generation" docs/ || echo "defect 4 clear"
```
Expected: all four "clear" lines.

- [ ] **Step 7: Run the full suite**

Run: `pixi run check`
Expected: lint, typecheck, and tests all pass.

- [ ] **Step 8: Commit**

```bash
git add docs/_partials/intro.md mkdocs.yml tests/test_gen_site_pages.py
git commit -m "site: the new nav, the rewritten Home cards, and a retired-path link guard"
```

---

## Self-Review

**Spec coverage.** §Why defects 1–4 → Tasks 5, 4, 6, 1. §1 IA → Tasks 5, 6, 7, 8, 9, 10, 11, 12
(`Explore` deferred to piece E, as the spec sequences it). §6 page dispositions → every row has a
task; `team.md` and `metrics-north-star.md` are explicitly unchanged. §6 hazard 1 (`exclude_docs`
stale key) → Task 2 + Task 7 step 6. §6 hazard 2 (URL breakage) → Tasks 7 and 8 take the move, with
inbound links fixed in the same commits and a guard in Task 12.

**Known gap, deliberate.** The spec's §Open Questions item on Nairobi's screening tier is a piece-D
decision and has no task here. The Khayelitsha item is Task 4 and must be *measured*, not assumed —
if step 1 reports a settlement other than Khayelitsha, step 2 says to write what is true.

**Sequencing constraint.** Tasks 7 and 8 must run back to back. Task 7 emits
`../../results/frontier.md` into every method page and Task 8 creates that file; a `--strict` build
between them fails by design.
