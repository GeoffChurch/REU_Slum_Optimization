# Donor quality is predictable — and it is mostly SIZE (2026-08-02)

> **CORRECTED same day.** The first version compared the learned descriptor against GW distance
> (+0.218 vs +0.035) and called it a 6× win. That is the wrong baseline. Against six trivial donor
> numbers the descriptor scores +0.218 vs **+0.190** — a margin this design cannot resolve. The
> durable findings are that donor quality is predictable at all, that it is NOT matching, and that
> it is NOT permeability; the learned feature map is not established as necessary. See
> [What "good donor" actually is](#what-good-donor-actually-is).

Descriptors of a **donor block alone** predict how well that donor transplants — for unseen donors
and unseen recipients — where GW distance does not. Adding the recipient, or the recipient–donor
difference, adds nothing.

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

## What "good donor" actually is

Asked whether donor quality is just permeability. It is not — and the honest answer deflates the
result above.

    within-recipient rho with perm_gap        vs the model's own prediction (spearman)
      descriptor model      +0.218
      n_parcels             -0.229               area_m2               -0.485
      compactness A/P^2     +0.188               compactness A/P^2     +0.344
      area_m2               -0.157               interior_footpath_m   -0.309
      own permeability P0   -0.134               n_parcels             -0.192
      donor_depth           -0.106               own permeability P0   -0.159
      interior_footpath_m   +0.071               donor_depth           -0.073
      GW distance           +0.035

**Permeability is among the weakest predictors** (-0.134). What predicts transplant quality is
donor SIZE, negatively: `n_parcels` alone reaches |rho| 0.229, matching the whole descriptor model,
and the model's own prediction correlates -0.485 with donor area and +0.344 with compactness. Scale-
invariant descriptors can still track size, because larger blocks in this corpus have systematically
different outline shapes.

So "good donor" is largely **small and compact** — a rule in two numbers.

The ablation that decides whether the spectra are worth anything:

    trivial only (n, area, perim, compactness, P0/parcel, footpath_m)   +0.190
    descriptor only                                                    +0.218
    both                                                               +0.241

The descriptor's +0.028 over six trivial numbers is well inside this design's noise. **It is not
established that the learned feature map is needed at all.**

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


## REPLICATED AT 5x SCALE (2026-08-03) — the effect is real, `n_parcels` was not

2,500 Gauteng pairs over **100 recipients** (`gw_pair_matrix_zaf_scaled.parquet`, 33 min, zero
skips), same protocol. Generating it first required fixing `pair_matrix`, which had been silently
skipping 75% of pairs since the width refactor.

    median within-recipient rho, held out          by RECIPIENT   by DONOR
      n_parcels only                                  -0.018       -0.008
      n_parcels + compactness                         +0.122       +0.099
      five trivial numbers                            +0.171       +0.137
      descriptor spectra                              +0.127       +0.104
      trivial + descriptor                            +0.172       +0.178

    permutation null (within-recipient shuffle):  median +0.003, 95th +0.032, max +0.038

**The null is now tight** — max +0.038 against Cape Town's +0.174 — so at 100 recipients the design
finally resolves what 20 could not. Three conclusions change or firm up:

1. **The effect is real and decisive.** +0.17 against a null that never exceeds +0.04.
2. **`n_parcels` alone is DEAD (-0.018).** The Cape Town reading that a single size number matched
   the whole model does not replicate — that was a 20-recipient artifact, and this note's own
   "good donor is mostly small and compact" headline was too strong.
3. **The descriptor still does not beat the trivial features** (+0.127 vs +0.171 by recipient), and
   still adds nothing on top of them when recipients are held out. It does add under donor-held-out
   CV (+0.178 vs +0.137), which is the one place the spectra have ever earned anything.
4. **It survives holding DONORS out** (+0.137 trivial), so it generalises to unseen donors — worth
   checking here because donors repeat up to 7x in this pool, unlike Cape Town's 85%-singleton set.

### And permeability, at scale

    within-recipient spearman with perm_gap
      perimeter_m                      -0.174
      area_m2                          -0.172
      own_permeability_P0_per_parcel   -0.161
      compactness_A_over_P2            +0.123
      real_gw_dist                     -0.115
      n_parcels                        -0.055
      donor_depth                      -0.055

Donor permeability is **-0.161** — no longer near the bottom as it read on Cape Town, but part of an
undifferentiated cluster with perimeter and area at -0.17. So the honest answer to "is donor quality
highly correlated with permeability" is: **weakly and negatively, and not distinguishably more than
with donor SIZE**, which permeability-per-parcel partly tracks. Nothing here isolates permeability
as the mechanism.

**What a donor filter would use:** the five trivial numbers, not the spectra and not GW. That is
+0.17 from geometry every block already has.
