"""The memory ladders of a wave's pools (docs/superpowers/specs/2026-10-21-lease-pool-design.md,
3.4): a class is a bundle (cards, CPUs, memory GB), what one worker holds, so one class on one
node is one demand line and one pool (GeoffChurch/mycooc's lesson 10).

A ladder is a wave preset's: a few rungs of memory, one bundle of cards and CPUs, and the lease's
length and margins. A job's class is the smallest rung holding its estimated peak (`rung_of`); an
OOM moves it a rung up, and past the top rung it goes to the long lane (pool_queue). The presets
(LADDERS) are chosen by name where a wave is configured and recorded whole in its queue's header.

- A CPU class holds a whole core, 2 CPUs (the plan's Ruling 3: ai's CR_CORE_MEMORY holds both of a
  core's threads, so a 1-CPU class's demand line would never match what its workers hold); jobs
  stay single-threaded (cluster_env.sh).
- The steward's bundle (STEWARD, Ruling 11) is no rung's, so its unit never joins a pool's line.
- Leases are the preset's length (Ruling 9): 120 minutes in both production presets, the active
  tier's 2 h; a worker claims only a job whose estimate fits the walltime it has left less the
  margin, and a job estimated past the lease less the margin is the long lane's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from cluster_submit import Bundle

#: The steward's lease (Ruling 11): orion, 2 CPUs (a whole core) and 3 GB, a bundle no rung has.
STEWARD = Bundle(cards=0, cpus=2, mem_gb=3)


def class_name(cards: int, mem_gb: int) -> str:
    """A class's name: `g16` (one card, 16 GB), `m12` (CPU only, 12 GB)."""
    if cards not in (0, 1):
        raise ValueError(f"a class has 0 or 1 cards, not {cards!r}")
    return f"{'g' if cards else 'm'}{mem_gb}"


@dataclass(frozen=True, kw_only=True)
class Ladder:
    """A wave preset's ladder: every worker holds `cards` (0 or 1) and `cpus`, and one of `rungs`
    (GB, ascending) of memory; a lease runs `lease_minutes`, and a worker claims only a job that
    fits the walltime it has left less `margin_minutes`. A job is failed after `strikes` attempts
    that made no progress (the plan's Ruling 6); an idle worker retires after `idle_minutes`,
    polling every `poll_seconds`. A job's estimate is its stored peak times `stored_mem` (its
    stored wall times `stored_time`), or the family's fit times `fit_mem`. `about` says what the
    preset optimizes and what it gives up."""

    name: str
    cards: int
    cpus: int
    rungs: tuple[int, ...]
    lease_minutes: float
    margin_minutes: float
    strikes: int
    idle_minutes: float
    poll_seconds: float
    stored_mem: float
    fit_mem: float
    stored_time: float
    about: str

    def __post_init__(self) -> None:
        if self.cards not in (0, 1):
            raise ValueError(f"{self.name}: a worker holds 0 or 1 cards, not {self.cards!r}")
        r = self.rungs
        if not r or any(g <= 0 for g in r) or any(a >= b for a, b in pairwise(r)):
            raise ValueError(f"{self.name}: rungs {r} are not positive and ascending")
        if not self.margin_minutes < self.lease_minutes:
            raise ValueError(f"{self.name}: a margin of {self.margin_minutes} minutes is not "
                             f"below the lease's {self.lease_minutes}")
        steward = [g for g in r if self.bundle(g) == STEWARD]
        if steward:
            raise ValueError(f"{self.name}: the rung of {steward[0]} GB is the steward's bundle "
                             f"({STEWARD.text()}), so its unit would join a pool's line")

    def bundle(self, mem_gb: int) -> Bundle:
        """What a worker of the rung `mem_gb` holds; a size that is no rung is refused."""
        if mem_gb not in self.rungs:
            raise ValueError(f"{self.name}: {mem_gb} GB is not a rung of {self.rungs}")
        return Bundle(cards=self.cards, cpus=self.cpus, mem_gb=mem_gb)


def rung_of(gb: float, ladder: Ladder) -> int | None:
    """The smallest rung of `ladder` holding `gb`; None past the top (the long lane's)."""
    if not math.isfinite(gb) or gb < 0:
        raise ValueError(f"{gb!r} GB is not an amount of memory")
    return next((r for r in ladder.rungs if gb <= r), None)


#: The presets, by name: the command line's choice, resolved once (no default).
LADDERS: dict[str, Ladder] = {lad.name: lad for lad in (
    Ladder(name="cpu-score", cards=0, cpus=2, rungs=(4, 12, 24, 56, 88), lease_minutes=120,
           margin_minutes=10, strikes=3, idle_minutes=10, poll_seconds=60, stored_mem=1.2,
           fit_mem=1.5, stored_time=1.25,
           about="CPU work (the lineup's scoring and proposals): a whole core and one of 4, "
                 "12, 24, 56 or 88 GB a worker (Task 7's classes and a 4-GB rung, "
                 "which 174 of the 565 lineup networks fit), in 2-hour leases. Optimizes how "
                 "many jobs share a node and how soon a pool steps aside (jobs no longer than "
                 "110 minutes, small ones filling a lease's tail); gives up the jobs past 88 GB "
                 "or 110 minutes, which run in the long lane, unshared and sized per job."),
    Ladder(name="gpu-presets", cards=1, cpus=4, rungs=(16, 32, 64, 96), lease_minutes=120,
           margin_minutes=10, strikes=3, idle_minutes=10, poll_seconds=60, stored_mem=1.2,
           fit_mem=1.5, stored_time=1.25,
           about="GPU work (the presets' one-card timings): one card, 4 CPUs and one of 16, 32, "
                 "64 or 96 GB of host memory a worker (the largest GPU peak stored is 7.6 GB, and "
                 "every usable card holds 32 GB, so classes vary host memory only), in 2-hour "
                 "leases. Optimizes a card's sharing (one process on it, released at a job "
                 "boundary); gives up the jobs past 96 GB or 110 minutes, which run in the long "
                 "lane, unshared and sized per job."),
    Ladder(name="smoke", cards=0, cpus=2, rungs=(2, 4), lease_minutes=20, margin_minutes=2,
           strikes=3, idle_minutes=3, poll_seconds=15, stored_mem=1.2, fit_mem=1.5,
           stored_time=1.25,
           about="The smoke waves (Task 10): a whole core and 2 or 4 GB a worker, in 20-minute "
                 "leases with a 2-minute margin, polling every 15 s, so a walltime kill, an OOM's "
                 "promotion and a lease's end are all seen within minutes. Optimizes how soon "
                 "the pool's machinery is seen on the cluster; gives up real work, whose jobs "
                 "neither its classes nor its leases hold."),
)}
