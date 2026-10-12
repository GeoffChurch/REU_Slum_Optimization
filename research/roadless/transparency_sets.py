"""The building sets of the road-transparency experiment (BACKLOG 5; the owner approved it
2026-10-12), as `reblock.transparency.Transparency` strategies.

  ClearedSet        the buildings one stored roadless clearing clears, at factor f; every other
                    building 1.
  MatchedRandomSet  the control: a random set matched to such a clearing. From each stratum of
                    the block's footprint areas (`strata` quantile bins, deciles at 10) it draws,
                    uniformly without replacement, as many buildings as the clearing has there:
                    the count matches exactly, the area share in distribution (on the 59 blocks
                    with stored clearings its median draw is 3.6% off the clearing's, 90% of draws
                    within 20%, but up to 139% where a clearing takes a block's largest buildings
                    and the top decile spans a factor 25 - 120 in area). Seeded per (seed, block),
                    at factor f; it draws from every building of a stratum, the clearing's own
                    included.

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

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import source_hash

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

    def members(self, block: Block) -> NDArray[np.intp]:
        return self.clearing.positions(block)

    def factors(self, block: Block) -> NDArray[np.float64]:
        return _with_factor(block, self.members(block), self.f)


def area_strata(block: Block, strata: int) -> NDArray[np.intp]:
    """Each building's stratum, 0 .. strata - 1: its quantile bin of footprint area over the
    block."""
    area = shapely.area(np.asarray(block.buildings.outlines))
    edges = np.quantile(area, np.arange(1, strata) / strata)
    return np.asarray(np.searchsorted(edges, area, side="right"), dtype=np.intp)


def _block_key(block_id: str) -> int:
    """A stable integer per block (Python's own string hash is salted per process)."""
    return int.from_bytes(hashlib.sha256(block_id.encode()).digest()[:8], "little")


@dataclass(frozen=True, kw_only=True)
class MatchedRandomSet:
    clearing: Clearing
    strata: int
    seed: int
    f: float

    def __post_init__(self) -> None:
        _factor_checked(self.f)
        if self.strata < 1:
            raise ValueError(f"strata must be >= 1; got {self.strata}")

    @property
    def identity(self) -> Hashable:
        return ("MatchedRandomSet", CODE, self.clearing.block_id, self.clearing.digest,
                self.strata, self.seed, self.f)

    def members(self, block: Block) -> NDArray[np.intp]:
        stratum = area_strata(block, self.strata)
        cleared = self.clearing.positions(block)
        rng = np.random.default_rng([self.seed, _block_key(block.block_id)])
        drawn = [rng.choice(np.flatnonzero(stratum == s), size=int((stratum[cleared] == s).sum()),
                            replace=False)
                 for s in range(self.strata)]
        return np.sort(np.concatenate(drawn)).astype(np.intp)

    def factors(self, block: Block) -> NDArray[np.float64]:
        return _with_factor(block, self.members(block), self.f)
