from geopandas import GeoDataFrame

from reblock.contracts import BBox, Region
from reblock.screen.identity import IdentityScreen


class _StubSource:
    # satisfies Source structurally; none of the three are used by IdentityScreen
    def region(self) -> Region:
        raise NotImplementedError

    def block_geometries(self, bbox: BBox | None = None) -> GeoDataFrame:
        raise NotImplementedError

    def building_points(self, bbox: BBox | None = None) -> GeoDataFrame:
        raise NotImplementedError


def test_identity_passthrough_returns_configured_block_ids() -> None:
    assert IdentityScreen(["a", "b"]).select(_StubSource()) == ["a", "b"]


def test_identity_default_is_none_meaning_all() -> None:
    assert IdentityScreen().select(_StubSource()) is None


def test_identity_copies_the_list_defensively() -> None:
    src = ["a", "b"]
    out = IdentityScreen(src).select(_StubSource())
    assert out == src and out is not src
