"""The building sets of the road-transparency experiment (BACKLOG 5; the owner approved it
2026-10-12), as `reblock.transparency.Transparency` strategies.

  ClearedSet        the buildings one stored roadless clearing clears, at factor f; every other
                    building 1.
  MatchedRandomSet  the control: a random set matched to such a clearing. For each building the
                    clearing holds in a stratum of the block's footprint areas, it draws one from
                    the same stratum, uniformly without replacement, from every building of the
                    stratum (the clearing's own included); seeded per (seed, block), at factor f.
                    The stratification is a Strategy: AreaMass (strata of equal total area, the
                    experiment's, at 20) or CountQuantiles (deciles at 10). Each draw records its
                    area over the clearing's and its Jaccard overlap with the clearing (`Draw`).

Each is one block's: `Clearing.read` reads that block's stored row ONCE, where the experiment is
configured, and the set raises on any other block. The identities cover what was read -- the
cleared positions themselves, not the row's path -- and this module's own source, which no reblock
import closure reaches. The rows are not in git and are rewritten in place (the free-first runs
rewrote 12 blocks' on 2026-10-11), so a key on the path alone would hand back networks built from
the old rows.

The guards: transparency_check.py `sets`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import config_identity, source_hash
from reblock.transparency import price_tag

ROADLESS = Path(__file__).resolve().parent
# This module's own code, in every identity below: it is in no reblock closure.
CODE = source_hash(Path(__file__))


@dataclass(frozen=True, kw_only=True)
class StoredRows:
    """One kind of stored clearing: a directory under research/roadless of `<block>_p2.parquet`
    rows (one row each) and the column holding the cleared positions."""

    rows: Path
    column: str


# The stored clearings at D 0.05 (area population, J_2, uni): SIMP's default and the cheap preset,
# as road_rescore.STORED's `simp_default` and `cheap` read them.
SIMP_DEFAULT = StoredRows(
    rows=Path("relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2/D0.05"),
    column="simp_cleared")
CHEAP = StoredRows(rows=Path("polish_rows/uni/S0.01cat.P64w8/D0.05"), column="cleared")


@dataclass(frozen=True, kw_only=True)
class Clearing:
    """The buildings one stored clearing clears on one block: positions in `block.buildings` row
    order (`Footprints` row order, the order the optimizers index), sorted and distinct."""

    block_id: str
    cleared: tuple[int, ...]

    @classmethod
    def read(cls, rows: StoredRows, block_id: str) -> Clearing:
        df = pd.read_parquet(ROADLESS / rows.rows / f"{block_id}_p2.parquet")
        if len(df) != 1 or df["block"].iloc[0] != block_id:
            raise ValueError(f"{rows.rows}: expected one row of {block_id}, got "
                             f"{df['block'].tolist()}")
        cleared = np.asarray(df[rows.column].iloc[0], dtype=np.int64)
        if (cleared < 0).any() or len(np.unique(cleared)) != len(cleared):
            raise ValueError(f"{rows.rows} {block_id}: positions must be distinct and >= 0")
        return cls(block_id=block_id, cleared=tuple(int(i) for i in np.sort(cleared)))

    @property
    def digest(self) -> str:
        return hashlib.sha256(np.asarray(self.cleared, dtype=np.int64).tobytes()).hexdigest()

    def positions(self, block: Block) -> NDArray[np.intp]:
        """The positions, for `block` only."""
        if block.block_id != self.block_id:
            raise ValueError(f"the clearing of {self.block_id} was asked for {block.block_id}")
        pos = np.asarray(self.cleared, dtype=np.intp)
        if len(pos) and pos[-1] >= len(block.buildings):
            raise ValueError(f"{self.block_id}: position {pos[-1]} of {len(block.buildings)} "
                             "buildings")
        return pos


def _factor_checked(f: float) -> None:
    if not 0.0 < f <= 1.0:
        raise ValueError(f"a factor must lie in (0, 1]; got {f}")


def _with_factor(block: Block, members: NDArray[np.intp], f: float) -> NDArray[np.float64]:
    out = np.ones(len(block.buildings), dtype=np.float64)
    out[members] = f
    return out


@dataclass(frozen=True, kw_only=True)
class ClearedSet:
    clearing: Clearing
    f: float

    def __post_init__(self) -> None:
        _factor_checked(self.f)

    @property
    def identity(self) -> Hashable:
        return ("ClearedSet", CODE, self.clearing.block_id, self.clearing.digest, self.f)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def members(self, block: Block) -> NDArray[np.intp]:
        return self.clearing.positions(block)

    def factors(self, block: Block) -> NDArray[np.float64]:
        return _with_factor(block, self.members(block), self.f)


class Stratification(Protocol):
    """How a block's buildings are cut into strata by footprint area, the matched random set
    drawing as many from each stratum as the clearing holds there."""

    @property
    def identity(self) -> Hashable: ...

    def assign(self, area: NDArray[np.float64]) -> NDArray[np.intp]:
        """Each building's stratum, from 0, given every building's footprint area."""
        ...


def _strata_checked(strata: int) -> None:
    if strata < 1:
        raise ValueError(f"strata must be >= 1; got {strata}")


@dataclass(frozen=True, kw_only=True)
class CountQuantiles:
    """Strata of equal COUNT: quantile bins of footprint area (deciles at 10). It matches the area
    share in distribution only: where a clearing takes a block's largest buildings, the top bin
    spans a factor 25 - 120 in area and a draw from it is far smaller (deciles on the 59 blocks:
    |area ratio - 1| median 0.036, p90 0.200, worst 1.39)."""

    strata: int

    def __post_init__(self) -> None:
        _strata_checked(self.strata)

    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def assign(self, area: NDArray[np.float64]) -> NDArray[np.intp]:
        edges = np.quantile(area, np.arange(1, self.strata) / self.strata)
        return np.asarray(np.searchsorted(edges, area, side="right"), dtype=np.intp)


@dataclass(frozen=True, kw_only=True)
class AreaMass:
    """Strata of equal total AREA: the buildings sorted by footprint area and cut where their
    running total crosses each 1 / strata of the block's footprint area, each building in the
    stratum that holds its middle. A block's largest buildings make strata of their own, so the
    clearing's largest are matched by others as large; where the clearing holds a stratum's only
    buildings, the random set holds them too (its overlap with the clearing, `Draw.jaccard`). The
    experiment's stratification, at 20 strata (owner's ruling, 2026-10-12)."""

    strata: int

    def __post_init__(self) -> None:
        _strata_checked(self.strata)

    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def assign(self, area: NDArray[np.float64]) -> NDArray[np.intp]:
        order = np.argsort(area, kind="stable")
        ranked = area[order]
        middle = np.cumsum(ranked) - ranked / 2.0
        out = np.empty(len(area), dtype=np.intp)
        out[order] = np.floor(middle / ranked.sum() * self.strata).astype(np.intp)
        return out


def _block_key(block_id: str) -> int:
    """A stable integer per block (Python's own string hash is salted per process)."""
    return int.from_bytes(hashlib.sha256(block_id.encode()).digest()[:8], "little")


class Draw(NamedTuple):
    """One random set, and what the experiment records of it: its footprint area over the
    clearing's (1 is a perfect match) and its Jaccard overlap with the clearing,
    |R & C| / |R | C|."""

    members: NDArray[np.intp]
    area_ratio: float
    jaccard: float


@dataclass(frozen=True, kw_only=True)
class MatchedRandomSet:
    clearing: Clearing
    stratification: Stratification
    seed: int
    f: float

    def __post_init__(self) -> None:
        _factor_checked(self.f)

    @property
    def identity(self) -> Hashable:
        return ("MatchedRandomSet", CODE, self.clearing.block_id, self.clearing.digest,
                self.stratification.identity, self.seed, self.f)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def draw(self, block: Block) -> Draw:
        """For each building the clearing holds in a stratum, one from the same stratum, uniformly
        without replacement, seeded per (seed, block)."""
        area = shapely.area(np.asarray(block.buildings.outlines))
        stratum = self.stratification.assign(area)
        cleared = self.clearing.positions(block)
        rng = np.random.default_rng([self.seed, _block_key(block.block_id)])
        members = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(stratum == s), size=int((stratum[cleared] == s).sum()),
                       replace=False)
            for s in np.unique(stratum[cleared])])).astype(np.intp)
        both = len(np.intersect1d(members, cleared))
        return Draw(members=members, area_ratio=float(area[members].sum() / area[cleared].sum()),
                    jaccard=both / (len(members) + len(cleared) - both))

    def members(self, block: Block) -> NDArray[np.intp]:
        return self.draw(block).members

    def factors(self, block: Block) -> NDArray[np.float64]:
        return _with_factor(block, self.members(block), self.f)
