# Spectral / effective-resistance metric — investigation & prototype

**Date:** 2026-07-11 · **Status:** investigation + prototype (no `src/reblock/` changes) ·
**Question:** should grounded effective resistance replace or augment the current shortest-path
directness/E metric (`budget.network_efficiency`)?

**Verdict (short):** **Augment, don't replace.** Grounded effective resistance is the right form
for the north-star's *egress* piece ("Piece 2") — the thing directness/E structurally cannot
express — and it arrives with three properties the current metric lacks: reachability folded in,
**Rayleigh monotonicity for free**, and **exact, machine-precision, ~10–550× cheaper rank-1
marginals**. It is also **3–150× faster** to evaluate at our block sizes. The redundancy-aware
*internal-circulation* analog (a resistance version of E/directness) is a smaller, optional win.
Keep shortest-path directness/E for internal circulation; add grounded resistance as the egress
score. Smallest next step in the final section.

Prototype code (all in scratchpad, referenced by absolute path):
- `…/scratchpad/spectral_proto.py` — the metric library (graph build reuses `budget` internals,
  grounded Laplacian solve, internal-resistance analog, frozen-entry monotone sweep, rank-1 check).
- `…/scratchpad/driver_compare.py` — cross-method agreement table.
- `…/scratchpad/driver_perf.py` — monotonicity, rank-1, scaling.

(Full scratchpad root: `/tmp/claude-1641171234/-home-gchurchill-src-reblock/8e45cee5-1c58-4e08-b3fd-8166cdfa5f11/scratchpad/`.)

---

## 1. What I prototyped

**Same graph as the current metric.** I reuse `budget._road_street_graph` +
`_edge_lines`/`_line_entries`/`_split_graph`, so the resistance metric sees *exactly* the graph and
the parcel line-entry nodes that `network_efficiency` sees — the comparison is apples-to-apples.

**Grounded resistance-to-egress.** Edge conductance `c_e = 1/length_e`, so a single wire has
resistance = its length in metres (resistance distance is then in metres and *equals* shortest-path
length on a tree, and is strictly lower when loops exist). Ground set `S` = graph nodes within `tol`
of `block.streets` (the existing egress network, all at potential 0). Weighted Laplacian `L = D − C`;
reduced Laplacian `L_G` = drop `S` rows/cols (SPD on each component that reaches ground). Each
parcel's egress resistance:

```
R_i = (L_G^{-1})_{entry_i, entry_i}  +  leg_i          # leg_i = euclid(rep_i, entry_i), last-mile walk
```

added as a series resistor, exactly mirroring `_sampled_efficiency`'s door-to-door `walk + drive`.
Unreached parcel (no entry within `tol`, or entry not connected to ground) → `R = cap` (bbox
diagonal), analogous to `access_burden`'s unreached-depth cap. Block score = mean `R_i` (**lower =
better**); benefit = `R(∅) − R(roads)`. Solve: one SuperLU factorization of `L_G`
(`scipy.sparse.linalg.factorized`), then a back-substitution per distinct entry node for the diagonal.

**Also prototyped:** (a) the **internal-circulation resistance analog** `(E_R, directness_R)` — same
door-to-door structure as `network_efficiency` but with all-pairs *resistance* distance in place of
shortest-path netdist (redundancy-aware version of the *same* question directness answers);
(b) the **Kirchhoff index** `Kf = N·Σ_{k≥2} 1/λ_k` via dense `eigvalsh`; (c) a **Hutchinson trace
estimate** of the block's egress-resistance aggregate `Σ_i R_i`.

---

## 2. Agreement with the current directness / E metric

### 2a. On a tree, the resistance metric *reproduces* shortest-path — exactly

`DJI.3_1_1721` (98 parcels), dijkstra roads (a shortest-path forest): `E = 0.00278` and
`E_R = 0.00278` — **identical to 5 digits**; `directness = 0.0308` vs `directness_R = 0.0311` (1 %
gap = the block-perimeter street loop's redundancy credit). This is the theoretical anchor: with
`c = 1/length`, resistance distance = shortest-path distance on a tree, so the resistance analog is a
strict generalization of E/directness that only *adds* a redundancy credit — it never contradicts
shortest-path where there is no redundancy to credit.

### 2b. The redundancy credit is real and points where the north-star doc says it should

`E_R ≥ E` and `directness_R ≥ directness` in **every** row measured. The gap is the credit for loops,
and it is largest exactly for the meshed arterial road sets:

| block (par) | roadset | direct | direct_R | gap |
|---|---|---|---|---|
| DJI.3_1_1751 (21) | dijkstra (tree) | 0.106 | 0.124 | +17 % |
| DJI.3_1_1751 (21) | arterial (mesh) | 0.490 | 0.810 | **+65 %** |
| DJI.3_1_1129 (14) | arterial (mesh) | 0.614 | 0.790 | +29 % |

Shortest-path directness cannot see that arterial through-roads create loops that give alternative
routes; the resistance metric does. This is the doc's "rewards redundancy" claim, quantified.

### 2c. Grounded resistance vs directness — they answer *different* questions, and disagree where the doc predicts

Best non-empty road set by each metric (directness: higher better; grounded R: lower better):

| block (par) | best by grounded-R | best by directness | agree? |
|---|---|---|---|
| DJI.3_1_1808 (10, compact) | dijkstra | arterial | **DISAGREE** |
| DJI.3_1_1129 (14, all street-fronting) | dijkstra (=empty) | arterial | **DISAGREE** |
| DJI.3_1_1751 (21, deep) | arterial | arterial | **AGREE** |

The pattern is exactly the north-star doc's thesis. Grounded R is an **egress** metric; directness/E
are **internal-circulation** metrics. On compact/already-fronting blocks (1808, 1129) an interior
mesh improves internal circulation (directness ↑) but does nothing for — or slightly worsens — egress
(some parcels re-snap their entry to an interior road with a longer path to the street), so grounded
R prefers the lean dijkstra tree. On the *deep* block (1751) the arterial through-roads genuinely
shorten egress, and the two metrics **agree**. Through-roads win on deep/elongated regions — and
grounded R is what captures that, while directness captures the orthogonal internal-circulation axis.

### 2d. Surprising: raw directness/E *penalize* connecting deep parcels; grounded R credits it

`DJI.3_1_1721` (98 par): empty→dijkstra takes reachability 0.35→1.00 and grounded `R_capped`
**86.4 → 18.6** (a 4.6× egress improvement), yet `directness` **drops 0.074 → 0.031** and `E` barely
moves. Adding shortest-path spurs to connect deep, previously-stranded parcels enlarges the reachable
all-pairs set with long, winding pairs, which *drags the directness mean down*. So the current metric
actively mis-scores the single most important reblocking action — getting stranded homes reachable.
Grounded R (with the unreached cap) folds reachability in and credits it correctly. This is the
clearest single argument that grounded R is the missing egress piece, not a nicer directness.

---

## 3. Compute cost (grounded-R vs `network_efficiency`)

Median-of-3 wall times, dijkstra roads, `k=40` sources for `network_efficiency`:

| block | par | nodes | edges | t_factor | t_diag(all entries) | t_1solve | **t_network_efficiency** |
|---|---|---|---|---|---|---|---|
| DJI.3_1_1808 | 10 | 17 | 16 | 0.0 ms | 0.0 ms | 0.01 ms | 1.4 ms |
| DJI.3_1_1751 | 21 | 35 | 30 | 0.0 ms | 0.1 ms | 0.01 ms | 2.4 ms |
| DJI.3_1_1721 | 98 | 222 | 195 | 0.1 ms | 1.2 ms | 0.02 ms | 14.4 ms |
| DJI.3_1_2010 | 123 | 269 | 230 | 0.1 ms | 1.9 ms | 0.02 ms | 15.2 ms |
| DJI.3_1_1789 | 405 | 900 | 839 | 0.3 ms | 13.9 ms | 0.04 ms | 50.7 ms |
| DJI.3_1_3182 | 649 | 1384 | 1313 | 0.5 ms | 31.7 ms | 0.06 ms | 85.6 ms |

- **Full per-parcel resistance** (factor + all entry-node diagonal solves): 3182 = 32 ms vs 86 ms →
  **2.7×**; 1789 = 14 ms vs 51 ms → **3.6×**; 2010 = 2 ms vs 15 ms → **7.6×** faster than
  `network_efficiency`.
- **Block aggregate `Σ_i R_i` via Hutchinson** (factor + ~32 stochastic solves): **~2 % relative
  error at 1–4 ms** (measured on 1789 and 3182) — vs the exact entry-node diagonal (14 ms / 32 ms in
  the table) that is ~**7–16×**, and vs `network_efficiency` ~**15–45×**, giving a single trustworthy
  navigability number at a fraction of the cost.
- **Which scales better?** The resistance solve. `network_efficiency` is `k` single-source Dijkstras,
  `O(k·(E + N log N))`; the resistance factorization is near-linear for these planar Laplacians and
  the speedup *grows* with N (27× → 157× for the single aggregate solve). The Laplacian factorization
  is essentially free at our sizes (≤0.5 ms even at 1384 nodes).

**Caveat on cheap aggregates (a correction to a plausible-but-wrong shortcut).** The naïve
single-solve `P = wᵀ L_G⁻¹ w` and the raw Kirchhoff index `Kf` are **extensive** (they grow with the
number of reached parcels / nodes), so they do **not** rank road sets correctly: on 1751, `P` calls
arterial *worse* while the per-parcel mean calls it *better*; raw `Kf` is dominated by node count
(arterial `Kf` 15884 vs dijkstra 1599 purely because arterial adds nodes). The trustworthy scores are
the **intensive** ones — per-parcel mean `R_i`, or normalized `Kf/\binom{N}{2}` (mean pairwise
resistance). Use the diagonal (exact) or Hutchinson (estimate of `Σ_i R_i`), not one bare solve.

---

## 4. Rank-1 marginal (Sherman–Morrison) — confirmed, and this is the strongest result

Adding one chord between two existing graph nodes is a rank-1 Laplacian update
`L_G' = L_G + c·uuᵀ`, `u = e_a − e_b`. Sherman–Morrison updates every resistance from one solve
`L_G⁻¹u`. Verified against a full re-factorize+re-solve:

| block | N_free | max\|R_rank1 − R_full\| | t_full_resolve | t_rank1 | speedup |
|---|---|---|---|---|---|
| DJI.3_1_1751 | 18 | 3.6e-15 | 0.4 ms | 0.04 ms | 10× |
| DJI.3_1_1721 | 165 | 7.1e-14 | 3.1 ms | 0.07 ms | 42× |
| DJI.3_1_1789 | 790 | 4.3e-14 | 170.9 ms | 0.31 ms | **547×** |

**Exact to machine precision, and 10–547× faster, the gap widening with graph size.** This is the
property that would let a resistance-based greedy arterial factor `L_G` **once** and score every
candidate chord by a rank-1 lookup, instead of the current greedy's full sampled all-pairs re-score
(~15–85 ms) *per candidate per step*. (Caveat: a real arterial that *subdivides* edges / injects new
entry nodes is a low-rank, not strictly rank-1, update — but still a cheap Woodbury update of rank =
#new nodes, and the between-existing-anchors continuations are exactly rank-1.) Relatedly,
**Rayleigh monotonicity is confirmed empirically**: with entries frozen against the full road set
(the `_efficiency_factory` pattern — *required*, since per-call entry churn breaks monotonicity for
resistance just as it does for E/directness), `R_capped` is monotone non-increasing over drainage
prefixes: 1751 `40.4→35.2→29.8→24.6→22.1→14.8→7.7`; 1721 `124.4→…→18.6`.

---

## 5. GPU suitability — honest assessment

**Per-block, per-solve: no.** These are *small* graphs (17–1384 nodes; nnz a few thousand). A CPU
direct solve factors in ≤0.5 ms and each back-substitution is ~10–100 µs. A GPU sparse CG on a
~1000-node system is overhead-bound — every SpMV is a tiny kernel launch (~10–30 µs) and you need
dozens of iterations, plus host↔device transfer, so the GPU loses to a sub-millisecond CPU direct
solve. Small-graph overhead is decisive here.

**Batched: yes, and specifically two shapes are good GPU targets.**
1. **Batch across blocks/sources** — the DJI sample alone is ~300 blocks; region-wide it is thousands.
   Stack many independent small `L_G` block-diagonally (or use a batched small-dense Cholesky:
   cuSOLVER/MAGMA batched) and solve them in one launch. The arithmetic per system is small but the
   batch width is huge, so batch parallelism (not per-solve speed) is the win.
2. **Batch the greedy's rank-1 candidate scoring** — scoring `K` candidate chords is `K` rank-1
   updates = applying `L_G⁻¹` to `K` columns = one **SpMM / dense GEMM**. That is the GPU's best case.
   A resistance greedy that evaluates all candidates as a batched matmul is the single most compelling
   GPU opportunity in this whole design.

**Bottom line:** GPU is an optimization for *scale* (many blocks × many candidates), not a
prerequisite — a well-tuned CPU direct-solve + rank-1 pipeline already puts per-block cost in the
low-single-digit-ms range. Reach for GPU only once batched across the region and the greedy inner
loop; don't GPU a single small block.

---

## 6. Recommendation & smallest next step

**Adopt grounded effective resistance as the north-star "Piece 2" (egress/accessibility) metric —
as an augmentation, not a replacement.** Justification, all measured above:

1. It expresses the quantity directness/E *cannot* — egress cost with reachability folded in — and it
   correctly credits the highest-value action (connecting stranded deep parcels) that raw directness/E
   actively penalize (§2d).
2. It reduces to shortest-path on trees (§2a) and only adds a redundancy credit (§2b), so it never
   contradicts the current metric where there is no redundancy to reward.
3. It comes with **Rayleigh monotonicity for free** (§4) — the cost-benefit monotonicity the current
   `_efficiency_factory` has to engineer with frozen entries.
4. It is **3–150× cheaper** to evaluate (§3) and has **exact, ~10–550× cheaper rank-1 marginals**
   (§4), directly attacking the greedy arterial's dominant re-scoring cost.

Keep shortest-path directness/E as the internal-circulation lens (they and grounded R answer
different questions; §2c). The redundancy-aware internal analog `(E_R, directness_R)` is a nice-to-have
that could later *replace* E/directness, but that is a smaller win and not urgent.

**Smallest next step:** add a `resistance_benefit` factory to `budget.py` mirroring
`_efficiency_factory` — freeze line entries against the full road set (needed for monotonicity, §4),
ground the street nodes, and score `benefit = (R_capped(∅) − R_capped(prefix)) / R_capped(∅)` from
one SuperLU factorization per prefix (with the entry-node diagonal, or a ~32-sample Hutchinson
aggregate for speed). Wire it as an `Eval` alongside the existing curves and sanity-check that its
cost-benefit AUC ranks a handful of DJI blocks sensibly against `access`. That is a self-contained,
~1-file addition that reuses all the existing graph/entry machinery; only after it proves out is it
worth building the rank-1 greedy or any GPU batching.

### Surprising findings (recap)
- Raw directness/E **decrease** when you connect stranded deep parcels — the current metric mis-scores
  the single most important reblocking move (§2d).
- The "one cheap linear solve" framing needs a caveat: the bare `wᵀL⁻¹w` (and raw `Kf`) are extensive
  and **rank road sets wrongly**; you need the intensive per-parcel diagonal (or a Hutchinson estimate
  of `Σ R_i`), which is still far cheaper than the current metric (§3).
- Rank-1 marginals are exact to **machine precision** (~1e-14), not merely approximate (§4).
