# Donor quality is predictable from cheap features; donor–recipient MATCHING is not (2026-08-02)

Cheap rotation/translation/scale-invariant descriptors of a **donor block alone** predict how well
that donor transplants — for unseen donors and unseen recipients — roughly **6× better than GW
distance does**. Adding the recipient, or the recipient–donor difference, adds nothing.

If that holds up, the retrieval programme has been solving the wrong problem. You do not need to
find the nearest donor to a given recipient. You need to know which blocks are good donors at all,
which is a **per-block score computable once over the corpus**, with no pairwise GW, no n², and no
index.

## The measurement

`scratchpad/ot/predict_fidelity.py` and `predict_fidelity_checks.py`, on the 500 pairs already in
`data/benchmarks/gw_pair_matrix.parquet`. No new GW fits.

Descriptors are the vetted ones from `outline_vs_fabric.py` — a 64-point outline EDM spectrum and a
k-NN parcel-packing profile — so every invariance is structural and nothing has to be learned.

Scoring is **within-recipient Spearman on held-out recipients**, because retrieval ranks donors *for
one recipient*: a model that only learned "some recipients are easier" would look good on pooled R²
and be useless. Cross-validation is GroupKFold by recipient, so no recipient's rows straddle folds.

    model                            median rho    better than chance
    gbr    (recipient+donor+diff)      +0.261          17/20
    ridge  (recipient+donor+diff)      +0.195          18/20
    GW distance alone (incumbent)      +0.035          11/20

## Which part of the pair carries it — the ablation that matters

    recipient + donor + diff   +0.195
    donor only                 +0.218      <- as good as the full pair
    difference only            +0.169
    recipient only                nan      <- constant within a recipient, as it must be

**The signal is donor quality, not matching.** Donor-alone matches the full pair. The pairwise
difference does carry something (+0.169), but it adds nothing on top of the donor's own descriptor.

## Why it is not an artifact

* **Permutation null.** Shuffling `perm_gap` *within* each recipient destroys donor signal while
  preserving every recipient's marginal: null median +0.018, 95th percentile +0.130, max +0.174 over
  30 shuffles. Real +0.195 exceeds **30/30** draws.
* **Not memorisation.** GroupKFold holds out recipients, not donors, so a repeated donor could in
  principle be memorised. It cannot be here: **365 of 429 donors (85%) appear exactly once**, and no
  donor appears more than 3 times. A test-fold donor is almost never in training, so the model is
  generalising from descriptors to unseen donors.
* **Ranking, not level.** Spearman within recipient cannot be inflated by getting recipient
  baselines right.

## Why this matters for the programme

GW distance does not predict fidelity — β = −0.18 within recipient, 95% interval [−11.2, +5.4] on
the 500-pair resample. The response so far was to treat that as a power problem. This says something
different: **the pairwise question may simply be the wrong one.** Donor goodness is largely a
property of the donor, and it is legible to two cheap descriptors.

Concretely, this removes the motivation for the Phase 2 retrieval index (masked-NCC FFT over
feature vectors) rather than merely deferring it. An index answers "which donor is nearest this
recipient"; the measurement says that question has little to do with transplant quality.

## What it does not establish

* **n is small.** 20 recipients × 25 donors, Cape Town only, one fidelity measure (`perm_gap`).
* **The margin is modest.** The permutation null's maximum (+0.174) sits close under the real
  +0.195. This wants a bigger pool before anything is built on it.
* **It does not say WHICH descriptor features carry it**, so "good donor" has no interpretation yet
  — only a prediction. That interpretation is the interesting part and is not done.
* It says nothing about consensus transplant, which already beats single-donor on 100% of blocks and
  works for a different reason.

## Next, cheapest first

1. **Re-run at scale.** The census/shortlist chain has 65,364 qualified blocks provisioned; this used
   449. Same script, bigger pool, and the permutation margin either widens or it does not.
2. **Hold donors out explicitly** (GroupKFold on donor, or on both), which is only strictly needed
   once the pool has repeated donors.
3. **Interpret the score** — which spectrum components move it, and does "good donor" reduce to
   something already measured (interior footpath length, depth, parcel count)? If it does, the
   descriptor is unnecessary and the finding is simpler still.
4. Only then, if a per-block donor-quality score holds up, wire it in as a donor FILTER — which is
   what the consensus reblocker would consume, and it needs no retrieval at all.
