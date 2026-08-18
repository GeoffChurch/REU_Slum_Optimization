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


def test_pin_is_declared_once() -> None:
    """No baker may re-declare the variant or method; they import them."""
    from scripts._example_block import PINNED_METHOD, PINNED_VARIANT

    assert PINNED_VARIANT == "method_comparison"
    assert PINNED_METHOD == "clearance"
    for baker in ("gen_perm_graph.py", "gen_web_bundle.py"):
        src = Path("scripts") / baker
        bound = _module_level_names(src.read_text(encoding="utf-8"))
        assert "VARIANT" not in bound, f"{baker} still declares its own VARIANT"
        assert "METHOD" not in bound, f"{baker} still declares its own METHOD"


def test_example_method_names_includes_osm_footpaths() -> None:
    """`conf/example/method_comparison.yaml:31` declares seven methods; `osm_footpaths` -- the
    real as-built informal network, injected from a committed OSM snapshot exactly as
    scripts/gen_example.py:175-182 injects it -- is the eighth, and the reference the whole
    comparison is measured against, not a competitor. A loader that only reads the declared list
    silently returns seven; this is the guard against exactly that.

    Cheap by construction, unlike `load_example_block`: `example_method_names` reads a yaml and
    stats one file, it does not propose (solve) anything, so this needs no cache and no `slow`
    marker."""
    from scripts._example_block import example_method_names

    names = example_method_names()
    assert set(names) == {
        "topology", "clearance", "clearance_looped", "cycle_native", "resistance_lp",
        "euclidean_grid", "greedy_arterial_access_displacement", "osm_footpaths",
    }
