"""Re-score the screens on HAND labels, once the worksheet's `verdict` column is filled.

The worksheet is the minimal set that can order the screens: `precision@k` differences are
(1/k) * the sum over two screens' symmetric difference, so only contested blocks matter, and
every block in any screen's top-k under either count source is in the file.

Reports three things, in order of what they settle:

1. **How often the survey and the adjudicator disagree**, and in which direction. This is the
   direct test of whether the February 2018 ground truth misses settlements newer than itself.
2. **precision@k per screen under both label sets**, so a change in the ordering is visible as
   a change rather than asserted.
3. **Pairwise margins in blocks**, because "0.80 vs 0.73" at k=15 is one block and should not
   be read as a separation.

`verdict` values are the four in `data/adjudication/RULE.md`: `all-dense-informal`,
`some-dense-informal`, `no-dense-informal`, `unclear`. Anything else is reported and REFUSED --
a typo silently scored as "not informal" would bias every number here toward the survey, which
is the failure under investigation.

`all-` and `some-` collapse to one class here, by the rule's own instruction: the question is
whether the block contains fabric reblocking would serve, and a block that is half township and
half shacks has it. The split is descriptive and nothing below depends on it.

    pixi run python -m scripts.score_adjudicated [worksheet.csv]
"""
from __future__ import annotations

import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path("data/adjudication/screen_top15_worksheet.csv")
VALID = {"all-dense-informal", "some-dense-informal", "no-dense-informal", "unclear"}
POSITIVE = {"all-dense-informal", "some-dense-informal"}    # collapse, per RULE.md
DECIDED = VALID - {"unclear"}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    rows = list(csv.DictReader(path.open()))
    filled = [r for r in rows if r["verdict"].strip()]
    print(f"{path}: {len(rows)} blocks, {len(filled)} adjudicated")
    if not filled:
        print("\nNothing to score yet -- fill the `verdict` column (informal/formal/unclear).")
        return 0

    bad = sorted({r["verdict"].strip() for r in filled} - VALID)
    if bad:
        print(f"\nREFUSING: unrecognised verdict(s) {bad}. Allowed: {sorted(VALID)}.")
        print("A typo scored as 'not informal' would bias every number below toward the survey.")
        return 1

    def collapsed(v: str) -> str:
        return "informal" if v in POSITIVE else "formal" if v != "unclear" else "unclear"

    agree = Counter((r["survey_label"], collapsed(r["verdict"].strip())) for r in filled)
    print("\n1. survey vs adjudicator")
    for (surv_label, hand_label), n in sorted(agree.items()):
        flag = "" if surv_label == hand_label else "   <- DISAGREE"
        print(f"   survey={surv_label:8s} hand={hand_label:8s} {n:4d}{flag}")
    missed = agree[("formal", "informal")]
    over = agree[("informal", "formal")]
    print(f"   survey MISSED {missed} settlement(s); over-called {over}. "
          f"{'Recall-limited, as predicted.' if missed > over else ''}")
    split = Counter(r["verdict"].strip() for r in filled)
    print("   raw verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(split.items()))
          + "   (all-/some- collapse; the split is descriptive, not calibrated)")

    rank_cols = [c for c in rows[0] if c.startswith(("kb_", "ob_")) and c not in
                 ("kb_count", "ob_count", "ob_over_kb", "ob_n", "ob_median_m2")]
    hand = {r["block_id"]: collapsed(r["verdict"].strip()) for r in filled}
    surv = {r["block_id"]: r["survey_label"] for r in rows}

    print("\n2. precision@k per screen (unclear verdicts excluded from both numerator and k)")
    print(f"   {'screen':34s} {'survey':>8} {'hand':>8} {'n scored':>9}")
    prec: dict[str, float] = {}
    for col in sorted(rank_cols):
        ids = [r["block_id"] for r in rows if r[col].strip()]
        scored = [b for b in ids if hand.get(b) in ("informal", "formal")]
        if not scored:
            continue
        p_h = sum(hand[b] == "informal" for b in scored) / len(scored)
        p_s = sum(surv[b] == "informal" for b in scored) / len(scored)
        prec[col] = p_h
        print(f"   {col:34s} {p_s:8.3f} {p_h:8.3f} {len(scored):9d}")

    print("\n3. pairwise margins, in BLOCKS (a 1-block margin is not a separation)")
    for a, c in itertools.combinations(sorted(prec), 2):
        if a.split("_", 1)[0] != c.split("_", 1)[0]:
            continue                                    # compare within a count source
        ia = {r["block_id"] for r in rows if r[a].strip()}
        ic = {r["block_id"] for r in rows if r[c].strip()}
        only_a = [b for b in ia - ic if hand.get(b) in ("informal", "formal")]
        only_c = [b for b in ic - ia if hand.get(b) in ("informal", "formal")]
        d = sum(hand[b] == "informal" for b in only_a) - sum(hand[b] == "informal" for b in only_c)
        verdict = "SEPARATED" if abs(d) >= 3 else "too close to call" if abs(d) <= 1 else "weak"
        print(f"   {a:30s} vs {c:30s} {d:+3d} blocks  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
