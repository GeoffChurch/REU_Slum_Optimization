"""The pin lives in ONE place. Two bakers previously each declared their own, so changing one
silently desynchronised the other -- and the widget would then describe a different block than the
caption beside it."""
from pathlib import Path


def test_pin_is_declared_once() -> None:
    """No baker may re-declare the variant or method; they import them."""
    from scripts._example_block import PINNED_METHOD, PINNED_VARIANT

    assert PINNED_VARIANT == "method_comparison"
    assert PINNED_METHOD == "clearance"
    for baker in ("gen_perm_graph.py", "gen_web_bundle.py"):
        src = Path("scripts") / baker
        text = src.read_text(encoding="utf-8")
        assert 'VARIANT = "' not in text, f"{baker} still declares its own VARIANT"
        assert 'METHOD = "' not in text, f"{baker} still declares its own METHOD"
