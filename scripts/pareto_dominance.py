"""Pairwise Pareto-dominance across the method lineup, from the committed example frontiers.

    pixi run python -m scripts.pareto_dominance

Reads every `examples/**/frontier_permeability.csv` and asks, for each ordered pair (A, B): does A's
achievable set cover B's whole curve? A dominates B iff at every displacement B samples, A can buy
at least as much permeability for no more displacement -- i.e. `env_A(d) >= perm_B(d)` for every
sampled `d`, where `env_A(d) = max{perm_A(d') : d' <= d}` is A's best-so-far permeability (the
curves are prefixes of one road set in drainage order, so the running max IS the achievable set).

TWO SAMPLING ARTIFACTS corrupt that test if you do not correct for them. Both were live when this
was first run, and both change the answer:

  * LOW END. `compare_report` subsamples each method's road list to 21 points, so a method with
    more roads gets finer resolution near zero. Scoring B's early samples against an A that has no
    sample there yet reads A as permeability 0 and manufactures a loss. It manufactured a -0.237
    "loss" for cycle_native against euclidean_grid on nairobi/density_compactness -- a region where
    cycle_native's curve is ABOVE the grid's at every displacement it actually reports. Fix: start
    the comparison at `max(first positive sample of A, of B)`.

  * HIGH END. A method whose own budget truncates it early cannot cover a rival's tail, and that
    reads as non-dominance even when the truncation is a config artifact rather than a property of
    the method. `cycle_native` was truncated by a hardcoded `range(60)` until 2026-08-13, which
    held it to 2/7 against euclidean_grid; unbound, it covers 6/7. There is no fix for this in the
    analysis -- the only fix is to not truncate. `--overlap` reports the narrower "where both
    operate" question as a cross-check, which is what makes a truncation-driven verdict visible.

The origin (0, 0) is dropped throughout: every method passes through it, so including it makes the
worst-case margin read as an exact tie for every pair.
"""
from __future__ import annotations

import argparse
import csv
from bisect import bisect_right
from pathlib import Path

from reblock.method_labels import friendly_method_name

ROOT = Path(__file__).resolve().parent.parent
REGIONS = [
    ("capetown/depth", "examples/multiblock_depth"),
    ("capetown/depth_density", "examples/multiblock_depth_density"),
    ("capetown/density_compactness", "examples/multiblock_density_compactness"),
    ("nairobi/depth", "examples/nairobi/multiblock_depth"),
    ("nairobi/depth_density", "examples/nairobi/multiblock_depth_density"),
    ("nairobi/density_compactness", "examples/nairobi/multiblock_density_compactness"),
    ("one-block", "examples/method-comparison"),
]
MIN_PTS = 4       # below this the "comparison" is a couple of points and says nothing

Curve = list[tuple[float, float]]


def load(path: Path) -> dict[str, Curve]:
    out: dict[str, Curve] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["method"], []).append(
                (float(row["displacement"]), float(row["permeability"])))
    return {m: sorted(pts) for m, pts in out.items()}


def envelope(pts: Curve) -> tuple[list[float], list[float]]:
    """Best-so-far permeability as displacement grows -- the achievable set, not the raw samples."""
    ds: list[float] = []
    ps: list[float] = []
    best = -1.0
    for d, p in pts:
        best = max(best, p)
        if ds and ds[-1] == d:
            ps[-1] = best
        else:
            ds.append(d)
            ps.append(best)
    return ds, ps


def env_at(ds: list[float], ps: list[float], d: float) -> float:
    i = bisect_right(ds, d + 1e-12) - 1
    return ps[i] if i >= 0 else 0.0


def margin(a: Curve, b: Curve, *, overlap_only: bool) -> tuple[float, float, int] | None:
    """(worst, mean, n) of `env_A(d) - perm_B(d)`. worst >= 0 means A covers B. None if too few
    comparable points. `overlap_only` also caps at min(max displacement), answering the narrower
    "where both operate" question -- see the module docstring on the HIGH END artifact."""
    lo = max(min(d for d, _ in a if d > 0), min(d for d, _ in b if d > 0))
    hi = min(max(d for d, _ in a), max(d for d, _ in b)) if overlap_only else max(d for d, _ in b)
    pts = [(d, p) for d, p in b if lo - 1e-12 <= d <= hi + 1e-12]
    if len(pts) < MIN_PTS:
        return None
    ds, ps = envelope(a)
    gaps = [env_at(ds, ps, d) - p for d, p in pts]
    return min(gaps), sum(gaps) / len(gaps), len(pts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap", action="store_true",
                    help="restrict to the range where both methods operate (cross-check for a "
                         "verdict driven by one method's truncation rather than its quality)")
    args = ap.parse_args()

    tally: dict[tuple[str, str], list[tuple[str, float]]] = {}
    reach: list[str] = []
    for label, rel in REGIONS:
        curves = load(ROOT / rel / "frontier_permeability.csv")
        reach.append(f"{label:<28} " + "  ".join(
            f"{friendly_method_name(m)} {max(d for d, _ in curves[m]):.3f}"
            for m in sorted(curves, key=lambda m: -max(d for d, _ in curves[m]))))
        for a in sorted(curves):
            for b in sorted(curves):
                if a == b:
                    continue
                got = margin(curves[a], curves[b], overlap_only=args.overlap)
                if got is not None:
                    tally.setdefault((a, b), []).append((label, got[0]))

    print("TERMINAL DISPLACEMENT per method per region (where each method's own budget stops it)")
    print("\n".join(reach))
    scope = "where both operate" if args.overlap else "over B's whole curve"
    print(f"\nA COVERS B ({scope}); worst margin is the permeability A has to spare at B's "
          f"worst point")
    for (a, b), rows in sorted(tally.items(), key=lambda kv: -sum(1 for r in kv[1] if r[1] >= 0)):
        wins = sum(1 for _, w in rows if w >= 0)
        if not wins:
            continue
        detail = " ".join(f"{lab}:{w:+.3f}" for lab, w in rows)
        print(f"  {friendly_method_name(a):>24} covers {friendly_method_name(b):<24} "
              f"{wins}/{len(rows)} | {detail}")


if __name__ == "__main__":
    main()
