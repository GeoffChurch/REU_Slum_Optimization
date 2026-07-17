# Does clearance fall into the settlement's desire-lines when repulsion is high?

**Question (owner, 2026-07-14):** in the multiblock region, long fairly-straight informal routes
crisscross the main block. Does `clearance` fall into these when `repulsion` is high?

## Ground truth: the LOW-DENSITY corridors (not betweenness)

First attempt defined desire-lines as high-**betweenness** corridors of the parcel-boundary graph.
**Rejected** (owner): betweenness concentrates where the network *pinches* = the DENSEST fabric, the
opposite of the visibly-low-density routes. The routes the eye sees are **linear low-density
channels** (buildings absent/sparse along a line). Correct extraction (block `ZAF.9.3.1_1_5810`):

- building **density** field (smoothed point histogram);
- **Frangi vesselness** on the density *valleys* — the tool for curvilinear structure in point noise;
  suppresses isotropic speckle, lights up the channel network. Fine scales (3-8 m) → the full web;
  large scales (5-12 m) + top-quantile → the WIDE salient corridors (the owner's "cross" + a far-left
  nexus splitting into 3 right-radiating rays, middle ray = the cross's horizontal → far-right
  vertical). Segment-linking by good-continuation connects fragments; Hough recovers the straight
  crossing lines that linking can't (crossing ≠ collinear). Scripts: `scratchpad/desireline_v*.py`.

Detection of the exact perceived network asymptotes (auto-tuning trades a missed corridor for a false
one); the validated density-valley **field** is used directly as ground truth for the measurement
below — no fragile line extraction.

## Measurement: repulsion sweep vs the low-density field (depth 3, block 5810)

| repulsion | roads | density %ile under road | frac road on wide corridor |
|---|---|---|---|
| −3 (straight/aspirational) | 234 | 0.58 | 0.16 |
| 0 | 241 | 0.56 | 0.17 |
| +3 | 228 | 0.45 | 0.26 |
| +6 (gap-following/buildable) | 226 | **0.42** | **0.28** |

## Answer: yes, monotonically — but partial

1. **Confirmed.** As repulsion rises, the **density under the roads crosses the median** (0.58 →
   0.42): from *above*-median density (bulldozing through the fabric) to *below*-median (in the
   gaps). Corridor-following nearly **doubles** (0.16 → 0.28). Both monotonic.
2. **Only partial** (42nd pctl, 28% on corridors — not 10th/70%): clearance must reach *every* deep
   parcel, so most road length is capillaries into the fabric, not the few wide desire-lines.
3. **Mechanism.** `repulsion` shifts road **geometry** into the low-density gaps + cuts displacement;
   it does NOT change that clearance solves **coverage**, not through-traffic. To make roads *follow*
   the desire-lines you'd add a corridor/through-traffic term to the objective — unpursued follow-up.

Reusable by-product: a density-valley Frangi-vesselness extractor for the informal circulation
network from building points (`scratchpad/desireline_v7/v8/v11.py`).
