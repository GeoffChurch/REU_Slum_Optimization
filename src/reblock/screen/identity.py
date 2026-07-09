"""IdentityScreen: the passthrough Screen (run()'s default). Selects nothing of its
own -- returns the configured block_ids (or None => all blocks), so a run with no
real screen behaves exactly as a plain reblock."""
from __future__ import annotations

from reblock.contracts import Source


class IdentityScreen:
    def __init__(self, block_ids: list[str] | None = None) -> None:
        self.block_ids = list(block_ids) if block_ids is not None else None

    def select(self, source: Source) -> list[str] | None:
        del source   # a passthrough needs no data
        return list(self.block_ids) if self.block_ids is not None else None
