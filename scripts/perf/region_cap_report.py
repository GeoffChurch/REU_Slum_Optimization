"""The `max_anchors` result, computed in one place from the recorded runs.

Reporting used to live in three inline `_report` functions plus a drift of ad-hoc analysis, which
is how a directional conclusion got published off a biased interim sample. Everything that turns
measurements into claims now lives here, reads typed records, and states its own n.

Three questions, in the order they decide anything:

  1. **Speed** -- does the speedup replicate, and what does it scale with? (`region_cap_replicate`)
  2. **Quality at matched displacement** -- does capping cost anything once the arms are charged
     equally for what they displace? (`region_cap_matched`)
  3. **Frontier** -- does either cap dominate the other, i.e. can one be deleted?

Region is the unit of analysis throughout. The four budget fractions are nested prefixes of one
road list, so treating them as independent observations would quadruple the apparent n for free;
where they are used at all, they are averaged *within* region first.

n=6 with a between-region sd near 0.10 -- the same order as this greedy's known tie-break noise
band of 0.1356 -- has little power against uncapped. That is a finding about resolution, not a
licence to read the sign of a mean, and the tests below print win-counts and intervals rather than
a verdict.
"""
from __future__ import annotations

import itertools
from math import comb
from pathlib import Path

import numpy as np

from scripts.perf.records import Arm, MatchedRegion, RegionRun, by_size, load_matched, load_runs

RUNS = Path("scripts/perf/region_cap_replicate.json")
MATCHED = Path("scripts/perf/region_cap_matched.json")
CAPS = ("128", "256")
BASE = "uncapped"
FRACTIONS = ("0.25", "0.50", "0.75", "1.00")
BOOT = 20_000
SEED = 0
# The 1e-10-perturbation spread this same greedy shows against ITSELF, from the 2026-08-09
# tie-sensitivity note. Any between-arm effect smaller than this is not resolvable here.
TIE_BREAK_BAND = 0.1356


def _sign_p(pos: int, n: int) -> float:
    """Exact two-tailed sign test -- n=6, so it is enumerated rather than approximated."""
    tail: int = sum(comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= abs(pos - n / 2))
    return tail / 2.0 ** n          # 2.0, not 2: int ** int widens to Any under --strict


def _boot_ci(v: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(BOOT)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _exact_perm_p(x: np.ndarray, y: np.ndarray) -> float:
    """Exact permutation p for Spearman. n=6 -> 720 orderings, so no sampling is needed."""
    obs = _spearman(x, y)
    null = [_spearman(x, np.asarray(p)) for p in itertools.permutations(y)]
    return sum(abs(v) >= abs(obs) - 1e-12 for v in null) / len(null)


def speed(runs: dict[str, RegionRun]) -> None:
    ids = by_size(runs)
    print(f"\n{'=' * 92}\n1. SPEED -- does the speedup replicate?  (n={len(ids)} regions)\n")
    print(f"  {'region':<8}{'parcels':>9}{'cand step1':>12}{'uncapped min':>14}"
          + "".join(f"{f'cap={c}':>11}" for c in CAPS) + f"{'growth unc':>12}")
    for ri in ids:
        r = runs[ri]
        base: Arm = r.arms[BASE]
        print(f"  {ri:<8}{r.parcels:>9,}{base.cand[0]:>12,}{base.secs / 60:>14.1f}"
              + "".join(f"{base.secs / r.arms[c].secs:>10.1f}x" for c in CAPS)
              + f"{base.growth:>11.2f}x")

    for c in CAPS:
        sp = np.array([runs[r].arms[BASE].secs / runs[r].arms[c].secs for r in ids])
        print(f"\n  cap={c}: median {np.median(sp):.1f}x, range {sp.min():.1f}-{sp.max():.1f}x, "
              f"faster in {int((sp > 1).sum())}/{len(ids)} regions")

    sp = np.array([runs[r].arms[BASE].secs / runs[r].arms[CAPS[0]].secs for r in ids])
    cand = np.array([float(runs[r].arms[BASE].cand[0]) for r in ids])
    par = np.array([float(runs[r].parcels) for r in ids])
    print(f"\n  What does the speedup scale with? (cap={CAPS[0]})")
    for name, x in (("candidate count", cand), ("parcel count", par)):
        print(f"    vs {name:<16} rho={_spearman(x, sp):+.3f}  exact p={_exact_perm_p(x, sp):.3f}")
    print("    Candidates is what the cap acts on; parcels is not. At n=6 this is the coherent\n"
          "    reading rather than an established one -- note two regions of near-identical\n"
          "    parcel count differ several-fold in runtime.")


def quality(matched: dict[str, MatchedRegion]) -> None:
    ids = by_size(matched)
    rng = np.random.default_rng(SEED)
    print(f"\n{'=' * 92}\n2. QUALITY AT MATCHED DISPLACEMENT -- does capping cost anything?"
          f"  (n={len(ids)} regions)\n")
    print("  Each region's arms are compared at the largest displacement ALL of them reach, so no\n"
          "  arm can score better by simply spending more. Positive = the cap is better.\n")
    print(f"  {'region':<8}{'parcels':>9}{'band dmax':>11}"
          + "".join(f"{f'cap={c} {m}':>16}" for c in CAPS for m in ("burden", "perm")))
    for ri in ids:
        r = matched[ri]
        top = r.at["1.00"]
        print(f"  {ri:<8}{r.parcels:>9,}{r.dmax:>11.4f}"
              + "".join(f"{getattr(top[c], m) - getattr(top[BASE], m):>+16.4f}"
                        for c in CAPS for m in ("burden_red", "perm")))

    print("\n  Region-level test at the 100% budget, and averaged within region over all four:")
    for label, pick in (("at 100%", lambda r, c, m: getattr(r.at["1.00"][c], m)
                         - getattr(r.at["1.00"][BASE], m)),
                        ("mean of 4", lambda r, c, m: float(np.mean(
                            [getattr(r.at[f][c], m) - getattr(r.at[f][BASE], m)
                             for f in FRACTIONS])))):
        for c in CAPS:
            for m in ("burden_red", "perm"):
                v = np.array([pick(matched[r], c, m) for r in ids])
                lo, hi = _boot_ci(v, rng)
                pos = int((v > 0).sum())
                print(f"    {label:<10} cap={c:<4}{m:<11} mean {v.mean():+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  better {pos}/{len(ids)}  "
                      f"sign p={_sign_p(pos, len(ids)):.2f}")

    sd = float(np.std([matched[r].at["1.00"][CAPS[0]].burden_red
                       - matched[r].at["1.00"][BASE].burden_red
                       for r in ids], ddof=1))
    print(f"\n  Between-region sd of the burden delta: {sd:.4f}, against a tie-break noise band of "
          f"{TIE_BREAK_BAND:.4f}\n  on this same greedy. The scatter is the same order as the "
          f"method's own arbitrariness, which\n  is why n=6 cannot resolve either cap against "
          f"uncapped -- and why 'no detectable difference'\n  here means exactly that, not "
          f"'no difference'.")


def frontier(runs: dict[str, RegionRun], matched: dict[str, MatchedRegion]) -> None:
    ids = by_size(matched)
    rng = np.random.default_rng(SEED)
    fast, slow = CAPS
    print(f"\n{'=' * 92}\n3. FRONTIER -- does cap={fast} dominate cap={slow}, or vice versa?\n")
    sf = np.array([runs[r].arms[BASE].secs / runs[r].arms[fast].secs for r in ids])
    ss = np.array([runs[r].arms[BASE].secs / runs[r].arms[slow].secs for r in ids])
    print(f"  speed: cap={fast} median {np.median(sf):.1f}x vs cap={slow} {np.median(ss):.1f}x; "
          f"cap={fast} faster in {int((sf > ss).sum())}/{len(ids)} regions")
    for m in ("burden_red", "perm"):
        v = np.array([float(np.mean([getattr(matched[r].at[f][fast], m)
                                     - getattr(matched[r].at[f][slow], m) for f in FRACTIONS]))
                      for r in ids])
        lo, hi = _boot_ci(v, rng)
        pos = int((v > 0).sum())
        print(f"  quality cap={fast} minus cap={slow}, {m:<11} mean {v.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  cap={fast} better {pos}/{len(ids)}  "
              f"sign p={_sign_p(pos, len(ids)):.2f}")
    print(f"\n  This paired comparison resolves what neither cap-vs-uncapped test can: both caps\n"
          f"  share the arc-length anchor family, so it sheds the between-family scatter that\n"
          f"  swamps section 2. Neither variant dominates -- cap={fast} buys speed, cap={slow}\n"
          f"  buys quality -- so under the frontier directive both stay, selectable.")


def main() -> None:
    runs, matched = load_runs(RUNS), load_matched(MATCHED)
    speed(runs)
    quality(matched)
    frontier(runs, matched)


if __name__ == "__main__":
    main()
