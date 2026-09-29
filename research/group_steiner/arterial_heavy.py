"""greedy_arterial on the largest blocks, the way the original study ran it: one block at a time in the MAIN
process with a 16-worker candidate pool (pool workers cannot fork, so the lineup pool runs it serially, ~4.5x
slower). `workers` and the engine's threads are exempt from the cache key, so the proposal is the same.

    PYTHONPATH=<repo> pixi run python arterial_heavy.py <block> [<block> ...]
"""
from __future__ import annotations

import dataclasses
import sys
import time

import large_blocks as LB
from reblock.derivations import propose

LB.OUT.mkdir(exist_ok=True)
LB.build()
name = "greedy_arterial_access_displacement"
for bid in sys.argv[1:]:
    city, block = LB._BLOCKS[bid]
    m = dataclasses.replace(LB._ARMS[city][name], workers=16)
    t0 = time.time()
    roads = propose(m, block).roads
    LB._write(LB.score(bid, name, roads, time.time() - t0))
    print(f"{time.strftime('%H:%M:%S')} {bid} {name} {time.time() - t0:.0f}s", flush=True)
