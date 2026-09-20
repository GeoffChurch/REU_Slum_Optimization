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

## The resolution ablation

The first three runs were scored against images captioned 0.35 m/px. They were not: the caption
reported the fetched TILE's resolution, while matplotlib rendered into a ~655 px axes, so the
pictures delivered a median **0.46 m/px** and two of the 23 were past `RULE.md`'s 1.0 m/px "too
coarse for shacks" line with neither flagged. "Depth is invisible from satellite" had therefore
been measured under worse conditions than it claimed, which is a live alternative explanation.

So the density and depth judges were re-run on the same 23 blocks re-rendered honestly — median
**0.22 m/px**, worst 0.70, nothing past the coarseness line. `masked_mpp035/` preserves the
original images so the first three runs stay reproducible.

| judge | ρ(density) | ρ(depth) | ρ(own) − ρ(other), 95% CI | separates? |
|---|---|---|---|---|
| density @0.46 | **0.854** | 0.043 | +0.811 [+0.339, +1.279] | yes |
| density @0.22 | **0.771** | 0.018 | +0.753 [+0.317, +1.168] | yes |
| depth @0.46 | 0.566 | **0.604** | +0.038 [−0.476, +0.534] | no |
| depth @0.22 | **0.728** | 0.489 | −0.239 [−0.674, +0.174] | no |

**Resolution is not the explanation.** At 2× finer imagery the depth judge got *worse*, not
better: its correlation with true depth fell 0.604 → 0.489 while its correlation with density rose
0.566 → 0.728. The recommendation stands — **label density, compute depth.**

### The part that does not depend on the leakage

|  | self-agreement across resolutions | best correlate |
|---|---|---|
| density | 0.918 | density |
| depth | 0.743 | **density (0.728), not depth (0.489)** |

The depth judge is reproducible and reproducibly measures the wrong thing. Its agreement with
*itself* (0.743) exceeds its agreement with true depth (0.489–0.604), and what it actually tracks
is packing. That is a direct measurement, not an inference from the comparison, so it survives the
contamination below.

### Contamination: a re-test cannot be blind in a repo that has published its own result

Commit **subjects** are injected into every agent's context by the harness `gitStatus` reminder,
and one of ours read `data(adjudication): density is visually separable, depth is not` — the prior
conclusion, stated outright. **Both** re-run judges disclosed it unprompted and said they
disregarded it; neither could prove it had no effect, and nor can we.

The reading was fixed BEFORE the result was seen, so it is not post-hoc: a re-run that still found
no separation would be confounded with conformity and count as weak evidence, while one that found
separation would contradict the leak and count as strong. The result landed on the weak branch.

Two things nonetheless argue against conformity driving it. The leak says density *does* separate,
yet the density judge's gap moved slightly **down** (+0.811 → +0.753), against the leak's
direction. And the depth judge's failure deepened with better data rather than staying flat, which
conformity does not predict. Suggestive, not decisive.

**The general lesson is procedural: commit the result after the replication, or run the
replication somewhere the commit log is not visible.** No prompt can fix this — the reminder is
injected before the agent begins.
