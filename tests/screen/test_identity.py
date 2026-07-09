from reblock.contracts import Region
from reblock.screen.identity import IdentityScreen


class _StubSource:
    def region(self) -> Region:  # satisfies Source structurally; unused by IdentityScreen
        raise NotImplementedError


def test_identity_passthrough_returns_configured_block_ids() -> None:
    assert IdentityScreen(["a", "b"]).select(_StubSource()) == ["a", "b"]


def test_identity_default_is_none_meaning_all() -> None:
    assert IdentityScreen().select(_StubSource()) is None


def test_identity_copies_the_list_defensively() -> None:
    src = ["a", "b"]
    out = IdentityScreen(src).select(_StubSource())
    assert out == src and out is not src
