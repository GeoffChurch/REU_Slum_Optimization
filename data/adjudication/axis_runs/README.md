# The axis-split judging experiment

The screen ranks on `Product(DepthProxy, Density)` — one number fusing two quantities. A ground
truth that also fuses them can never test whether the fusion is right. So three judges answer the
same 23 blocks under three questions:

| run | question | scored against |
|---|---|---|
| `control_masked_density.csv` | how tightly packed are the structures? (0–3) | `density_ha` |
| `control_masked_depth.csv` | how buried is the most buried structure? (0–3) | `true_depth` |
| `control_masked_both.csv` | `RULE.md`'s single question (4 labels) | `dd_proxy` |

Each judge is scored against a **computed** quantity, so the experiment needs no new human labels.
Truth is `../axis_truth.csv`; score with `pixi run python -m scripts.score_axis_run`.

**Read the 3×3 matrix, not the diagonal.** Density and depth correlate at Spearman 0.580 across
the control sample, so a judge silently reporting "how informal does this look" would score
respectably on both. The question is discriminant validity: does each judge track its own quantity
*more* than the other's? If not, the two questions elicited one judgement wearing two hats.

**The sample is designed, not random** — 5 per cell of a 2×2 on (density, depth), plus the four
`DEFERRED` blocks that motivated the split. Within it the two quantities correlate at 0.304
against the corpus's 0.580, which is the decorrelation that gives the matrix its power. Correct
for measuring judge-quantity agreement; **wrong for any population rate**, and nothing here may be
weighted up to the 18,309-block pool.

**Rendering is `masked/` at 0.35 m/px**, not the 0.10 m/px the fetcher now produces. Deliberate:
it matches `runs/control_masked_sequence_sonnet.csv`, so this experiment is comparable to the
existing combined-judge run. Re-running at 0.10 is a resolution ablation worth doing separately.

**Leakage control.** The combined judge got an extract of `RULE.md`'s judging sections only.
Everything from `## Provenance` onward is process, and the axis tables there name **four of these
23 blocks with their measured density and depth** — a judge reading the full file would be doing
lookups. `LEAKED_BY_RULE` additionally excludes the two blocks the rule names as worked examples.
