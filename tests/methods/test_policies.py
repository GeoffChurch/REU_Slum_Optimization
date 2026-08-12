from __future__ import annotations

from reblock.methods.arterial.policies import CandidatePolicySpec, Faithful, Fixed, Grow


def test_specs_are_their_own_identity_and_distinct() -> None:
    # All three pairwise comparisons (not just two of the three), plus reflexive equality for
    # every class -- not just Grow. The brief's own two-assertion version leaves Grow vs Faithful
    # unchecked, which a copy-paste transcription bug (e.g. Faithful.identity accidentally
    # returning Grow(), plausible given three near-identical `return self` property blocks
    # stacked in the file) would pass silently: Fixed().identity != Faithful().identity still
    # holds (Fixed vs Grow are correctly distinct), so the collision between Grow and Faithful
    # goes undetected -- a real defect (two different policies silently sharing a cache key).
    assert Grow().identity == Grow().identity
    assert Fixed().identity == Fixed().identity
    assert Faithful().identity == Faithful().identity
    assert Grow().identity != Fixed().identity
    assert Grow().identity != Faithful().identity
    assert Fixed().identity != Faithful().identity


def test_there_is_no_string_factory_left() -> None:
    """_make_policy resolved a closed set of three by string, with a ValueError fallback that a
    typo reached at runtime instead of at type-check time."""
    from reblock.methods.arterial import policies
    assert not hasattr(policies, "_make_policy")


def test_specs_satisfy_the_protocol() -> None:
    """All three implementations conform to CandidatePolicySpec; non-conformers don't."""
    assert isinstance(Grow(), CandidatePolicySpec)
    assert isinstance(Fixed(), CandidatePolicySpec)
    assert isinstance(Faithful(), CandidatePolicySpec)
    assert not isinstance(object(), CandidatePolicySpec)
