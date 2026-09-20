"""Score the AXIS-SPLIT judging experiment: can a judge see density and depth separately?

The screen ranks on `Product(DepthProxy, Density)` -- one number fusing two quantities. A ground
truth that also fuses them can never test whether the fusion is right, so this experiment asks
three judges the same 23 blocks under three questions: packing only, burial only, and the
combined `RULE.md` question. Each judge's output is scored against a COMPUTED quantity, so the
experiment needs no new human labels.

## The result to read is the 3x3 matrix, not the diagonal

A judge scoring well against its own quantity proves little on its own: density and depth
correlate at Spearman 0.580 across the control sample, so a judge that silently reports "how
informal does this look" would score respectably on both. What matters is DISCRIMINANT validity --
the density judge must track density MORE than it tracks depth, and the depth judge the reverse.
If the off-diagonal matches the diagonal, the two questions elicited one judgement wearing two
hats, and splitting the label set buys nothing.

The sample is drawn balanced on a 2x2 of (density, depth) precisely so the matrix can separate
them: within it the two quantities correlate at 0.304 against the corpus's 0.580.

## Why the sample is not random, and what that forbids

It is a DESIGNED sample -- 5 per cell, plus the four deferred blocks that motivated the split.
That is correct for measuring agreement between a judge and a quantity, and it is wrong for
estimating any population rate. Nothing here may be weighted up to the 18,309-block pool; for
that see `score_agent_run.py`, whose control rows are a stratified random draw.

    pixi run python -m scripts.score_axis_run
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

RUNS = Path("data/adjudication/axis_runs")
TRUTH = Path("data/adjudication/axis_truth.csv")

# What each judge was asked, and the computed quantity it is meant to track. The third entry is
# the fused question against the fused metric -- the baseline the split has to beat.
# judge -> (run file, its column, the quantity it was ASKED about, the quantity it must NOT
# track more than). The `@` suffix is the rendering's measured resolution, median over the 23
# blocks, as drawn -- not as fetched. Both resolutions are kept because the pair IS the
# ablation: `masked_mpp035/` preserves the images the first three judges saw, and `masked/`
# holds the same blocks re-rendered after the caption was found to be reporting the tile's
# resolution rather than the picture's.
JUDGES = {
    "density@0.46": ("control_masked_density.csv", "packing", "density_ha", "true_depth"),
    "depth@0.46": ("control_masked_depth.csv", "burial", "true_depth", "density_ha"),
    "both@0.46": ("control_masked_both.csv", "label", "dd_proxy", "true_depth"),
    "density@0.22": ("control_masked_density_hires.csv", "packing", "density_ha", "true_depth"),
    "depth@0.22": ("control_masked_depth_hires.csv", "burial", "true_depth", "density_ha"),
}
QUANTITIES = ["density_ha", "true_depth", "dd_proxy"]

# `RULE.md` teaches by worked example and names these two with their labels, so for a judge told
# to read it they are lookups rather than judgements. Same exclusion as `score_agent_run.py`.
LEAKED_BY_RULE = {"ZAF.9.3.1_1_63818", "ZAF.9.3.1_1_38988"}

# A channel nobody enumerated until a judge disclosed it. An agent's environment carries the
# repo's recent commit SUBJECTS, and on the day this ran the last four subjects were
# "defer 40094...", "defer 39399...", "defer 40760..." -- naming three of the 23 blocks and
# saying, of each, that a human had found it too ambiguous to label. That is not the answer, but
# it is information about the answer, and a judge that reads it is no longer judging blind.
#
# The combined judge reported the contamination itself, on item 19, unprompted. Treat that as the
# only reason we know: the density judge ran in the same environment and did not mention it, which
# is evidence about disclosure, not about exposure.
#
# Not a static property of the corpus -- it is a fact about WHEN a run executed, so it lives with
# the experiment rather than in `score_agent_run.py`. Re-running after these commits age out of
# the window needs this set re-derived, not reused.
LEAKED_BY_GIT = {"ZAF.9.3.1_1_39399", "ZAF.9.3.1_1_40760", "ZAF.9.3.1_1_40094"}
# Blocks whose computed density is an average over land uses that should not be averaged: a shack
# strip and a playing field, houses and a water-treatment plant. For these the TRUTH value is the
# defective number, so judge-truth agreement measures nothing -- `39399`'s block mean is 50.9/ha
# against 86/ha in its edge strip and 21/ha in its interior. Identified independently by the
# density judge as its three hardest items, and by hand before the run; the two lists agree.
MIXED_LAND_USE = {"ZAF.9.3.1_1_39399", "ZAF.9.3.1_1_53959", "ZAF.9.3.1_1_21708",
                  "ZAF.9.3.1_1_34552"}

ORDINAL = {"no-dense-informal": 0, "some-dense-informal": 1, "all-dense-informal": 2}


def numeric(col: pd.Series, name: str) -> pd.Series:
    """A judge's answers as numbers, with `unclear` dropped rather than coerced.

    `unclear` is an abstention: `RULE.md` promises it costs nothing, and mapping it to a number
    would silently make it an opinion. Dropping it here is what keeps that promise.
    """
    if name == "label":
        return col.map(ORDINAL)
    return pd.to_numeric(col.where(col != "unclear"), errors="coerce")


def main() -> int:
    if not TRUTH.exists():
        print(f"missing {TRUTH} -- run the axis computation first", file=sys.stderr)
        return 1
    truth = pd.read_csv(TRUTH)

    found = {k: v for k, v in JUDGES.items() if (RUNS / v[0]).exists()}
    if not found:
        print(f"no runs in {RUNS}", file=sys.stderr)
        return 1

    for scope, drop in (("all blocks", LEAKED_BY_RULE),
                        ("git-leak excluded", LEAKED_BY_RULE | LEAKED_BY_GIT),
                        ("homogeneous only", LEAKED_BY_RULE | LEAKED_BY_GIT | MIXED_LAND_USE)):
        print(f"\n== {scope} ==")
        report(found, truth, drop)
    return 0


def report(found: dict, truth: pd.DataFrame, drop: set[str]) -> None:
    print(f"{'judge':13s} {'n':>3s} {'abst':>5s}  " + "".join(f"{q:>13s}" for q in QUANTITIES))
    matrix = {}
    for judge, (fname, col, own, _other) in found.items():
        run = pd.read_csv(RUNS / fname)
        run = run[~run.block_id.isin(drop)]
        d = run.merge(truth, on="block_id", how="inner")
        vals = numeric(d[col], col)
        keep = vals.notna()
        row = []
        for q in QUANTITIES:
            rho = (spearmanr(vals[keep], d.loc[keep, q]).statistic
                   if keep.sum() > 2 else float("nan"))
            row.append(rho)
        matrix[judge] = dict(zip(QUANTITIES, row, strict=True))
        star = ["*" if q == own else " " for q in QUANTITIES]
        print(f"{judge:13s} {keep.sum():3d} {(~keep).sum():5d}  "
              + "".join(f"{v:>12.3f}{s}" for v, s in zip(row, star, strict=True)))
    print("  * = the quantity that judge was asked about")

    print("\ndiscriminant validity -- does each judge track its OWN quantity"
          " more than the other's?")
    for judge in matrix:
        _f, _c, own, other = JUDGES[judge]
        if own == "dd_proxy":
            continue          # the fused judge has no "other axis" to separate from
        a, b = matrix[judge][own], matrix[judge][other]
        verdict = "SEPARATES" if a > b else "DOES NOT SEPARATE"
        print(f"  {judge:13s} own {own:11s} {a:+.3f}   vs other {other:11s} {b:+.3f}"
              f"   -> {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
