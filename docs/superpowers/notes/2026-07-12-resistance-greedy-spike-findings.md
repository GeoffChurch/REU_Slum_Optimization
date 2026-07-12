# Resistance-objective greedy reblocker — spike findings (2026-07-12)

**Status:** exploratory spike (scratchpad `resistance_greedy_spike.py`), NOT productionized. Validates
the approach before a spec→plan→SDD build.

## Goal
Break the arterial reblocker's per-step scaling wall ("Wall 2": each greedy step scores hundreds of
candidates with an independent shortest-path, so per-step cost ∝ parcel count and ~4000-parcel
regions are intractable) AND optimize a coverage/egress objective (more municipally relevant than
directness). Idea: score all candidates' marginal reduction in mean grounded egress-resistance via a
batched Sherman-Morrison `L⁻¹B` GEMM instead of per-candidate shortest paths.

## Design (research-grounded)
- **Substrate = the Voronoi dual (parcel-adjacency graph).** Parcels are Voronoi cells; their dual is
  the parcel-adjacency graph — already the substrate of the existing access-depth metric. Nodes =
  parcels, edges = adjacency (conductance 1/rep-distance), ground = street-fronting parcels (BFS
  layer 1). Canonical connectivity, no arbitrary k. (User's suggestion; better than kNN centroid
  connectors here because it *is* the access-depth substrate.)
- **Action = node-grounding.** A road "grounds" the parcels it fronts (resistance 0). Decision
  variable = which parcels to ground. Per the research this matters: **node-grounding is submodular
  → greedy has the (1−1/e) guarantee**; edge-addition to minimize effective resistance is NOT
  submodular (Summers et al. *retracted* their claim; Achterberg–Kooij 2025) — only monotone (Rayleigh).
- **Objective:** `J = (1/n) tr(L_g⁻¹)` over free parcels (grounded = 0). `L_g` = grounded Dirichlet
  Laplacian (SPD, true inverse). Grounding parcel j drops `tr(L_g⁻¹)` by exactly
  `Δ(j) = ‖L_g⁻¹ e_j‖² / (L_g⁻¹)_jj` (principal-submatrix inverse identity) — so J is monotone by
  construction, and all frontier candidates' Δ come from ONE batched solve `X = L_g⁻¹ E_frontier`.
- Restricting candidates to the frontier (free parcels adjacent to a grounded one) keeps roads
  connected to the street automatically.

## Spike results (real Cape Town blocks, cached capetown_full)

| block | parcels | shortest-path arterial | resistance-greedy | resistance ↓ | throughput |
|---|---|---|---|---|---|
| ZAF.9.3.1_1_21719 | 479 | — | 60 grounded, 0.25 s | 39% | ~41k cand/s |
| ZAF.9.3.1_1_38528 | 2017 | 106 s for **4** roads | 300 grounded, 15 s | 51% | ~13k cand/s |
| 6-block region | 3425 | **timed out (15 min)** | 400 grounded, 55 s | 43% | ~8k cand/s |

- **Wall 2 broken.** The 3425-parcel region (arterial couldn't finish in 15 min) → 55 s. Per-step
  scoring of ~1157 candidates via the batched GEMM is ~138 ms vs the arterial's minutes/step.
- **Monotone** resistance reduction verified every step (Δ>0 by construction).
- Corridors: at small budgets, thin tree-like corridors from the street into the interior (sensible).

## Caveat / open problem
Node-grounding = **area-clearing**, not thin road centerlines. At larger budgets the greedy grounds
whole deep *cores* (radial blobs), because grounding a cluster center reduces all surrounded parcels'
resistance most. It correctly finds *where* roads are most needed, but as clearing-zones (high
displacement), not lines. To emit buildable roads: (a) extract centerlines (skeleton/medial axis of
the grounded set — deep cores → denser road grid, which is actually correct); or (b) add a
displacement/road-length penalty; or (c) an edge-addition reformulation (thin roads, but
non-submodular + needs road geometry).

## (a) Grading vs the incumbents — the important negative result

Extracted road centerlines (spanning tree of the grounded corridor, street-connected) and graded on
the four lenses vs dijkstra/mesh on the 2017-parcel block, **at equal road budget** (AUC capped at
the resistance-greedy's own road density — the fair "who wins per meter" comparison):

| lens (AUC @ equal budget) | resistance-greedy | dijkstra | mesh |
|---|---|---|---|
| access | 0.722 | **0.743** | 0.734 |
| resistance | 0.127 | **0.261** | 0.245 |
| directness | **0.007** | 0.002 | 0.002 |

**The resistance-greedy is dominated by dijkstra**, including on the *resistance* lens it supposedly
optimizes (0.127 vs 0.261), and dijkstra is far faster (0.6 s vs 22 s on this block). Two root causes:

1. **Substrate mismatch.** The Voronoi-dual (parcel-adjacency) substrate optimizes *access-DEPTH*
   (hop-resistance), NOT the eval's *network-geometry* egress resistance (street/road segments +
   entry/leg). So the builder minimizes a proxy that is competitive on access (0.722 vs 0.743) but
   is not the graded metric. To win the resistance lens, the builder must optimize on the
   network-geometry graph the eval uses — a different, harder formulation.
2. **dijkstra is a strong, fast baseline.** It has no scaling wall (0.6 s on 2017 parcels — only the
   per-candidate-shortest-path *arterial* does), and its frontage-following spanning network is
   already good egress geometry. The resistance-greedy's rep-to-rep tree (thin, sensible-looking,
   6495 m vs dijkstra's 15452 m) is not buildable (crosses parcels) and geometrically worse egress.

**Verdict:** the batched-marginal MECHANISM is real and fast at scale, but the prototype as
formulated does NOT beat dijkstra on the eval metrics. The speed win is only over the arterial, which
dijkstra already beats. **Do not productionize as-is.** The honest options: (i) reformulate to
optimize on the network-geometry graph (align builder + eval) + frontage-snap the roads, then
re-test vs dijkstra — more work, uncertain payoff; (ii) accept dijkstra as the coverage/egress
baseline; (iii) reframe the resistance marginal as an *analysis* tool (it cleanly identifies where
roads are most needed) rather than a builder.

## (a-reformulated) Network-geometry edge-upgrade (v2) — the crux experiment

Rebuilt the greedy to optimize on a network-geometry substrate: adjacency edges carry weak "walk"
conductance, street = ground, a road = **upgrading** an edge to road-conductance (ROAD_RATIO=20×,
one physical constant). Pure rank-1 (fixed node set/sparsity), frontier-restricted batched solve.
`resistance_greedy_v2.py`.

| lens @ equal budget | v2 (479-parcel) | dijkstra | | v2 (2017-parcel) | dijkstra |
|---|---|---|---|---|---|
| access | **0.642** | 0.632 | | 0.659 | **0.709** |
| resistance | 0.159 | **0.207** | | 0.094 | **0.224** |
| directness | **0.012** | 0.001 | | **0.005** | 0.002 |

v2 is a clear improvement over v1 (edge-upgrade aligns with the eval), and **wins on the small
block** — but **loses on the big deep block** (the regime that matters), on both access and the
resistance lens it optimizes, and is ~40× slower (25 s vs 0.6 s).

## Final verdict
Across **two formulations** (v1 node-grounding on the Voronoi dual; v2 edge-upgrade on the
network-geometry graph), the resistance-objective greedy **does not beat dijkstra** on deep blocks.
Root cause: the eval's egress metrics reward frontage-following short paths — *exactly* what
dijkstra's shortest-boundary-path heuristic already delivers, at high per-metre efficiency and 0.6 s.
The resistance objective's theoretical edge (rewarding redundancy) does not overcome dijkstra's
geometric per-metre efficiency. The one untried lever — **frontage-snapping** the roads — is
dijkstra's own mechanism, so adopting it would be "resistance-greedy + dijkstra's trick," not a win
for resistance-optimization per se.

**Recommendation: accept dijkstra as the coverage/egress baseline; do NOT productionize the
resistance-greedy as a builder.** Salvageable value: (1) the batched rank-1 marginal *mechanism* is
a real, fast technique (breaks the per-candidate-shortest-path arterial's wall — but dijkstra has no
such wall); (2) the resistance *marginal* is a valid **analysis/diagnostic** (where roads help most),
distinct from a builder. The arterial keeps its niche (directness/navigability), where it uniquely wins.

## Further speedup available
The spike **refactorizes `L_g` every step** (O(n^1.5)), which is why per-step grows (4→50→138 ms).
Rank-1 Sherman-Morrison *maintenance* of `L_g⁻¹` between accepted groundings (Predari–Angriman–
Meyerhenke 2023) makes per-step ~constant; JL/Laplacian-solver approximation of the diagonal
(Spielman–Srivastava 2011; Angriman et al. 2020) scales further.

## Key citations
Ghosh, Boyd & Saberi 2008 (SIAM Review); Klein & Randić 1993; Chandra et al. 1989 (commute-time =
resistance); Summers et al. 2015 (+2017 retraction) & Achterberg–Kooij 2025 (non-submodularity of
edge-addition); Clark–Bushnell–Poovendran 2014, Fitch–Leonard 2016 (grounded-Laplacian leader
selection = submodular node-grounding); Predari–Angriman–Meyerhenke 2023 (rank-1 maintenance);
Spielman–Srivastava 2011, Angriman et al. 2020 (diagonal approximation); Sheffi 1985, Qi et al. 2013
(centroid connectors); **Brelsford, Martin, Hand & Bettencourt 2018 (Science Advances) & Brelsford,
Martin & Bettencourt 2019 (EPB)** — deployed reblocking, binary topological access; our
resistance-graded objective is a refinement.
