"""Score an AGENT labelling run against the two label sets that already exist.

A run is `data/adjudication/runs/<sheet>_<rendering>_<protocol>.csv` with columns
`block_id,label,notes`. Runs are standalone by construction: nothing here writes to a `verdict`
column, and the human sheets are opened read-only. A judge that could edit the answer key is not
a judge.

## What can and cannot be measured, and on which sheet

The two sheets answer different halves, because of how each was drawn:

- **`screen_top15_worksheet.csv`** is top-k by construction. Of 35 adjudicated blocks, 34 are
  informal and 1 is not, so it measures **sensitivity** and nothing else. An agent that answers
  "informal" unconditionally scores 34/35 here. Reporting accuracy on this sheet would be
  reporting the base rate.
- **`control_sample.csv`** is a stratified random draw, 109 of 140 blocks survey-formal, and it
  is the only source of **negatives**. Its truth column is the SURVEY, which is recall-limited --
  so an agent-informal/survey-formal cell is ambiguous between a survey miss and an agent false
  positive, and is reported as a QUEUE for human adjudication, never as an error rate.

## The population estimate

Control rows carry `weight` = N_stratum / n_sampled_stratum, so
`sum(weight * indicator)` is a Horvitz-Thompson total over the 18,309-block pool. It is reported
as an UPPER BOUND on survey misses until the queue is human-adjudicated, because every agent
false positive inflates it. The bound is still worth having: it is the first number in this
project attached to a denominator.

    pixi run python -m scripts.score_agent_run                    # every run found
    pixi run python -m scripts.score_agent_run runs/foo.csv       # one
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

RUNS = Path("data/adjudication/runs")
WORKSHEET = Path("data/adjudication/screen_top15_worksheet.csv")
CONTROL = Path("data/adjudication/control_sample.csv")

# RULE.md teaches by worked example, and it names these two blocks together with their labels.
# Every judge is told to read it, so for an AGENT run these two are lookups rather than judgements
# and are excluded from agent scoring. They stay in the rule because a human adjudicator needs the
# worked examples and because one rule read by both judge kinds is the point of the file -- the
# contamination is a property of the scoring, not of the document. It mattered: `_63818` is the
# only human-NEGATIVE among the 35 adjudicated blocks, so leaving it in reported a specificity of
# 1/1 that was really a successful grep.
LEAKED_BY_RULE = {"ZAF.9.3.1_1_63818", "ZAF.9.3.1_1_38988"}

VALID = {"all-dense-informal", "some-dense-informal", "no-dense-informal", "unclear"}
POSITIVE = {"all-dense-informal", "some-dense-informal"}      # collapse, per RULE.md
DECIDED = VALID - {"unclear"}


def positive(label: str) -> bool | None:
    """True/False, or None for `unclear` -- which is excluded from both numerator and denominator
    exactly as `score_adjudicated.py` does, so an honest abstention costs nothing."""
    return None if label == "unclear" else label in POSITIVE


def load_run(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    labels = {r["block_id"].strip(): r["label"].strip() for r in rows if r["block_id"].strip()}
    bad = sorted({v for v in labels.values()} - VALID)
    if bad:
        raise SystemExit(
            f"{path}: unrecognised label(s) {bad}. The four in data/adjudication/RULE.md are the "
            f"only ones scored -- a typo silently read as 'not informal' would bias every number "
            f"here toward the survey, which is the thing under investigation.")
    return labels


def score(path: Path) -> None:
    labels = load_run(path)
    sheet = CONTROL if "control" in path.stem else WORKSHEET
    rows = {r["block_id"]: r for r in csv.DictReader(sheet.open())}
    print(f"\n{'=' * 78}\n{path.name}   ({len(labels)} labels, sheet={sheet.name})\n{'=' * 78}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(Counter(labels.values()).items())))

    covered = [b for b in labels if b in rows]
    if len(covered) != len(labels):
        print(f"  !! {len(labels) - len(covered)} labelled block(s) are not on {sheet.name}")

    if sheet is WORKSHEET:
        leaked = sorted(set(covered) & LEAKED_BY_RULE)
        if leaked:
            print(f"  excluding {len(leaked)} block(s) named with their labels in RULE.md: "
                  f"{', '.join(leaked)}")
        covered = [b for b in covered if b not in LEAKED_BY_RULE]
        pairs = [(positive(labels[b]), positive(rows[b]["verdict"].strip()))
                 for b in covered if rows[b]["verdict"].strip()]
        pairs = [(a, h) for a, h in pairs if a is not None and h is not None]
        pos = [(a, h) for a, h in pairs if h]
        print(f"\n  vs HUMAN verdicts ({len(pairs)} comparable)")
        print(f"    sensitivity  {sum(1 for a, _ in pos if a)}/{len(pos)}"
              f"   <- all this sheet can measure; it has ~1 negative")
        neg = [(a, h) for a, h in pairs if not h]
        if neg:
            print(f"    on its {len(neg)} negative(s): {sum(1 for a, _ in neg if not a)} correct")
        return

    # control sample: the survey is the only truth column, and it is recall-limited
    cells: Counter[tuple[bool, bool]] = Counter()
    queue: list[tuple[float, str, str, str]] = []
    for b in covered:
        a = positive(labels[b])
        if a is None:
            continue
        s = rows[b]["survey_label"] == "informal"
        cells[(a, s)] += 1
        if a != s:
            queue.append((float(rows[b]["weight"]), b, rows[b]["stratum"], labels[b]))

    print(f"\n  vs SURVEY labels ({sum(cells.values())} decided)")
    print(f"    agent informal, survey informal   {cells[(True, True)]:>4}")
    print(f"    agent informal, survey formal     {cells[(True, False)]:>4}   <- candidate MISSES")
    print(f"    agent formal,   survey informal   {cells[(False, True)]:>4}"
          f"   <- candidate agent misses")
    print(f"    agent formal,   survey formal     {cells[(False, False)]:>4}")

    ht = sum(w for w, _, _, lab in queue if positive(lab))
    print(f"\n  Horvitz-Thompson UPPER BOUND on survey misses in the 18,309-block pool: "
          f"{ht:,.0f} blocks")
    print("    Upper, not an estimate: every agent false positive inflates it. Adjudicate the")
    print("    queue below to turn it into a rate.")

    if queue:
        queue.sort(reverse=True)
        print(f"\n  adjudication queue ({len(queue)} blocks, highest leverage first)")
        for w, b, stratum, lab in queue[:20]:
            print(f"    {w:>7.1f}  {b:<24} {stratum:<10} agent={lab}")
        if len(queue) > 20:
            print(f"    ... and {len(queue) - 20} more")


def main() -> int:
    paths = ([Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1
             else sorted(RUNS.glob("*.csv")))
    if not paths:
        print(f"no runs in {RUNS}/ -- nothing to score")
        return 0
    for p in paths:
        score(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
