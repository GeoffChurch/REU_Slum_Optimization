"""The pin lives in ONE place. Two bakers previously each declared their own, so changing one
silently desynchronised the other -- and the widget would then describe a different block than the
caption beside it."""
import ast
from pathlib import Path


def _module_level_names(src: str) -> set[str]:
    """Every name bound by a module-level `ast.Assign` or `ast.AnnAssign`.

    A literal substring scan (`'VARIANT = "' in text`) has false negatives: it misses
    single-quoted (`VARIANT = 'x'`) and annotated (`VARIANT: str = "x"`) re-declarations alike,
    and neither form is caught anywhere else in the gate (`ruff check .`'s selected rules have no
    quote-style check, and `pixi run check` does not run `ruff format`). Parsing the module and
    reading the AST's own binding targets is quoting- and annotation-independent."""
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _bakers() -> list[Path]:
    """Every script that loads the pinned block, DERIVED rather than listed.

    The listed form named two of the three bakers and missed `gen_frontier_bundle.py` -- the one
    added by the branch that introduced it -- so injecting a `VARIANT`/`METHOD` re-declaration there
    left the suite green (final review, I3). That is the hand-maintained-closed-list failure mode
    this file's own docstring exists to guard against, applied to the guard itself. Membership is
    now the property that matters: a script is a baker exactly when it imports the shared loader, so
    the next baker is covered by writing that import.
    """
    found = [p for p in sorted(Path("scripts").glob("*.py"))
             if "scripts._example_block" in p.read_text(encoding="utf-8")]
    assert len(found) >= 3, f"expected at least the three known bakers, found {found}"
    return found


def test_pin_is_declared_once() -> None:
    """No baker may re-declare the variant or method; they import them."""
    from scripts._example_block import PINNED_METHOD, PINNED_VARIANT

    assert PINNED_VARIANT == "explore"
    assert PINNED_METHOD == "clearance_looped"
    for src in _bakers():
        bound = _module_level_names(src.read_text(encoding="utf-8"))
        assert "VARIANT" not in bound, f"{src.name} still declares its own VARIANT"
        assert "METHOD" not in bound, f"{src.name} still declares its own METHOD"


def test_example_method_names_includes_osm_footpaths() -> None:
    """`conf/example/explore.yaml` declares its methods; `osm_footpaths` -- the real as-built
    informal network, injected from a committed OSM snapshot exactly as
    scripts/gen_example.py:175-182 injects it -- is one more, and the reference the whole
    comparison is measured against, not a competitor. A loader that only reads the declared list
    silently returns five; this is the guard against exactly that.

    It is not hypothetical, and the failure is silent in the direction that matters. The snapshot
    is named for the BLOCK, so when `multiblock_depth_density`'s seed moved to
    `ZAF.9.3.1_1_5810` its committed `desire_lines_ZAF.9.3.1_1_38528.geojson` stopped matching and
    that example quietly dropped from six methods to five -- visible only as a count in a
    regeneration log, against Nairobi's six on the same line.

    Cheap by construction, unlike `load_example_block`: `example_method_names` reads a yaml and
    stats one file, it does not propose (solve) anything, so this needs no cache and no `slow`
    marker."""
    from scripts._example_block import example_method_names

    names = example_method_names()
    assert set(names) == {
        "clearance_looped", "euclidean_grid", "cycle_native", "cycle_native_betweenness_contrast",
        "greedy_arterial_access_displacement", "osm_footpaths",
    }
