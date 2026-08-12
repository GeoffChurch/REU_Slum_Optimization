from __future__ import annotations

from reblock.methods.arterial.policies import CandidatePolicySpec, Faithful, Fixed, Grow


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
