# FFT homogenization of the roadless escape metric: a feasibility study

2026-10-10. Repo context: GeoffChurch/reblock, branch research/roadless (lifted.py, common.py, NOTES.md, BACKLOG.md), read only. Everything ran on the CPU, one thread per process, at most 16 worker processes; nothing ran on the GPU or the cluster. One unintended side effect: matplotlib runs from the repo venv wrote four PIL `.pyc` caches into `.venv`. I deleted them; the repo's git status is unchanged.

## 0. What was homogenized, and what "error" means

- **Fabric.** Block ZAF.9.3.1_1_5810: 6,619 footprints, 576,948 m² inside the outline. It is rasterized exactly as the metric does it (`lifted.UniformMesh(0.5, offset=lifted.OFFSET)`): 2,816 × 3,116 cells, 2,313,038 of them inside, each with an open fraction from 4 × 4 sub-samples. Demand is the metric's own (`lifted.demand`, area population, 2 stranded homes). Gated blocks 22422 and 30848 were extracted the same way.
- **Model homogenized: a scalar 'uni'.**
  - One potential per cell.
  - A face between 4-neighbours conducts c = min(o_a, o_b), the uni rule for an axis step.
  - Ground is the metric's ground cells.
  - Unknowns are the open cells in a 4-connected component that holds ground (`Grid.reach`'s rule).
  - The real metric is the lifted K = 8 heading graph with turning length 3 m; Section 3.2 measures how close the scalar model is to it.
- **Reference ("fine").** The scalar model solved directly at 0.5 m: 1,795,357 unknowns, pyamg smoothed aggregation with CG to rtol 1e-10, 12–15 s and 1.4 GB on one core. Every two-scale error below is measured against this solve of the *same* model, so it is homogenization error only.
- **Metric.** J₂ = Σ wᵢuᵢ², where uᵢ is the injection-weighted mean potential over home i's ring cells (as in `Scorer.home_u_of`). Lens A = 1 − (J/J₀)^½.

## 1. Which FFT schemes handle zero-conductance inclusions

Buildings conduct nothing, so the contrast is infinite. The literature separates two questions: the discretization, which comes first, and the solver.

### 1.1 Discretization

| Discretization | With pores | Same as our finite volumes? |
|---|---|---|
| Moulinec–Suquet trigonometric collocation (original) | Numerically unstable for porous materials (Schneider 2020; Donval & Schneider 2025) | no |
| Fourier–Galerkin (Vondřejc, Zeman & Marek 2014; Brisard & Dormieux 2012) | "Usually lead to numerical instabilities" with pores; no linear convergence in general (Donval & Schneider 2025; Schneider 2020) | no |
| Rotated scheme (Willot 2015); conductivity version (Willot, Abdallah & Pellegrini 2014) | Stable; iteration count tends to a finite value as contrast goes to infinity (their abstract) | a rotated finite-difference stencil: close, not identical |
| Staggered grid (Schneider, Ospald & Kabel 2016) | Stable; the basic scheme's linear convergence survives pores | **yes, exactly**: for a scalar conductivity it is the 5-point finite-volume scheme with face conductances |
| Voxel finite elements with an FFT preconditioner (Zeman et al. 2017; Leuschner & Fritzen 2018; Ladecký et al. 2023) | Stable | nodal, not cell-centred |
| Composite voxels (Kabel, Merkert & Schneider 2015) | Laminate-averaged interface voxels | relevant to our partly open cells |
| Porous-conductive reformulation (To & Bonnet 2020) | Reformulates on the solid skeleton's field linked to pore-boundary values, to avoid the infinite-contrast trouble | no |

The prototype uses the staggered grid. Its Green operator is the inverse periodic 5-point Laplacian, with FFT symbol 4 sin²(πk_x/n_x) + 4 sin²(πk_y/n_y). Every scheme therefore solves the same linear system as the sparse direct reference.

### 1.2 Solvers

| Scheme | On this discretization | With pores |
|---|---|---|
| Basic (Moulinec & Suquet 1994, 1998) | Preconditioned Richardson: u += L⁻¹r / c0 | Converges linearly under a geometric regularity condition on the pores. The rate depends on an interface constant: narrow necks make it worse (Schneider 2020). |
| Eyre–Milton (Eyre & Milton 1999) | Peaceman–Rachford splitting (Schneider, Wicht & Böhlke 2019) | Only non-expansive; convergence not guaranteed (Donval & Schneider 2025) |
| Augmented Lagrangian (Michel, Moulinec & Suquet 2001) | Douglas–Rachford, i.e. Eyre–Milton damped by ½ | Damped Eyre–Milton converges linearly for any damping in (0, 1), with the best estimate at ½ (Donval & Schneider 2025) |
| Monchiet–Bonnet polarization (2012); comparison by Moulinec & Silva (2014) | the family containing both of the above | as above |
| Conjugate gradients (Zeman et al. 2010; Brisard & Dormieux 2010; Gélébart & Mondon-Cancel 2013) | CG preconditioned by the reference Green operator | Converges. My inference from standard CG theory, not a cited result: its rate depends on the square root of the conditioning that limits the basic scheme. |
| Fast gradient, Barzilai–Borwein, Anderson (Schneider 2017; Wicht et al. 2021) | accelerations of the above | as their base scheme |

Reviews: Schneider 2021; Lucarini, Upadhyay & Segurado 2022; Mishra, Vondřejc & Zeman 2016.

### 1.3 Measured on 5810's fabric

Periodic windows, stopping at ‖Au − b‖/‖b‖ < 1e-8, one core. c0 was tuned by a sweep (`schemes_c0.py`). The basic scheme used 0.55; at 0.5, the stability edge, it needed 1,539 iterations. The augmented Lagrangian and Eyre–Milton used 0.15.

| window | L (m) | CG | basic | aug. Lagrangian | Eyre–Milton |
|---|---|---|---|---|---|
| A dense | 50 | 53 it, 0.02 s | 340 it, 0.10 s | 164 it, 0.09 s | 155 it, 0.10 s |
| A dense | 100 | 72 it, 0.10 s | 666 it, 0.80 s | 178 it, 0.42 s | 166 it, 0.39 s |
| A dense | 200 | 118 it, 0.76 s | 1,454 it, 8.88 s | 396 it, 4.79 s | 260 it, 3.22 s |
| C median | 50 | 45 it, 0.02 s | 216 it, 0.07 s | 108 it, 0.06 s | 118 it, 0.07 s |
| C median | 100 | 60 it, 0.08 s | 602 it, 0.78 s | 154 it, 0.41 s | 226 it, 0.55 s |
| C median | 200 | 88 it, 0.59 s | 1,058 it, 5.87 s | 265 it, 3.06 s | 210 it, 2.48 s |
| D sparse | 50 | 6 it, 0.00 s | 76 it, 0.02 s | 95 it, 0.05 s | 65 it, 0.04 s |
| D sparse | 100 | 63 it, 0.08 s | 862 it, 1.09 s | 230 it, 0.57 s | 121 it, 0.29 s |
| D sparse | 200 | 78 it, 0.61 s | 814 it, 4.66 s | 204 it, 2.77 s | 141 it, 1.75 s |

- All four schemes converged everywhere. Largest relative tensor difference from the direct solve: CG 4.5e-11, Eyre–Milton 1.4e-10, augmented Lagrangian 2.4e-9, basic 5.9e-9.
- CG is 3–12× faster in wall time than the next best scheme.
- The basic scheme's iteration count grows with window size (340 → 666 → 1,454 on window A), roughly the square of CG's growth (53 → 72 → 118). A larger window contains a narrower worst neck.
- Undamped Eyre–Milton converged here, but it is not guaranteed to with pores; it is not the scheme to rely on.
- Infinite contrast did not bite in practice, because 5810's open space is 75% of the area and percolates.

## 2. Prototype: effective tensors of real fabric

### 2.1 Discretization and validation (`fftk.py`, `test_fftk.py`)

- The unknown is a periodic fluctuation u.
- Faces: c_x = min(o_ij, o_i,j+1) and c_y = min(o_ij, o_i+1,j).
- The system is A u = −DᵀC E, and σ_ij = ⟨q_i⟩ under E = e_j.
- Checks:
  - Open space gives σ = I exactly.
  - A laminate gives σ_xx equal to the open share and σ_yy = 0.
- Reference: a SuperLU solve of the same periodic matrix, with one pinned cell per conducting component.
- Agreement is at most 1.2e-10 relative on all 20 windows.
- Cells outside the block count as open street.

### 2.2 Tensors by window

Four centres were chosen:
- **A:** densest by building count at 200 m.
- **B:** dense, south-west.
- **C:** the median window.
- **D:** sparsest, the institutional compound.

Angles are of the major axis, counter-clockwise from east.

| win | L (m) | bldg/ha | open | sxx | syy | sxy | maj/min | axis (°) | CG it | CG s | direct s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 25 | 128 | 0.67 | 0.432 | 0.218 | −0.006 | 1.98 | 178 | 50 | 0.007 | 0.008 |
| A | 50 | 176 | 0.63 | 0.330 | 0.298 | −0.014 | 1.14 | 160 | 53 | 0.022 | 0.030 |
| A | 100 | 162 | 0.67 | 0.368 | 0.403 | −0.009 | 1.11 | 103 | 72 | 0.130 | 0.146 |
| A | 200 | 168 | 0.64 | 0.325 | 0.336 | −0.005 | 1.05 | 112 | 118 | 1.21 | 0.63 |
| A | 300* | 139 | 0.68 | 0.356 | 0.386 | 0.000 | 1.08 | 89 | 148 | 3.81 | 2.15 |
| B | 25 | 112 | 0.80 | 0.462 | 0.637 | −0.023 | 1.39 | 97 | 47 | 0.007 | 0.009 |
| B | 50 | 136 | 0.77 | 0.489 | 0.546 | −0.036 | 1.19 | 116 | 61 | 0.025 | 0.038 |
| B | 100 | 136 | 0.77 | 0.477 | 0.546 | −0.038 | 1.22 | 114 | 72 | 0.124 | 0.195 |
| B | 200 | 120 | 0.79 | 0.528 | 0.572 | −0.014 | 1.10 | 106 | 80 | 0.80 | 1.21 |
| B | 300 | 123 | 0.78 | 0.523 | 0.561 | −0.010 | 1.08 | 103 | 91 | 2.38 | 3.63 |
| C | 25 | 224 | 0.68 | 0.406 | 0.425 | +0.044 | 1.24 | 51 | 38 | 0.005 | 0.007 |
| C | 50 | 128 | 0.78 | 0.535 | 0.563 | +0.006 | 1.06 | 78 | 45 | 0.017 | 0.035 |
| C | 100 | 128 | 0.76 | 0.508 | 0.549 | −0.011 | 1.09 | 104 | 60 | 0.102 | 0.184 |
| C | 200 | 132 | 0.76 | 0.488 | 0.527 | −0.014 | 1.10 | 108 | 88 | 0.87 | 1.03 |
| C | 300* | 122 | 0.78 | 0.519 | 0.564 | −0.012 | 1.10 | 104 | 92 | 2.39 | 3.57 |
| D | 25 | 0 | 1.00 | 1.000 | 1.000 | 0 | 1.00 | – | 1 | 0.000 | 0.014 |
| D | 50 | 0 | 1.00 | 0.999 | 1.000 | 0 | 1.00 | – | 6 | 0.003 | 0.094 |
| D | 100 | 43 | 0.89 | 0.498 | 0.798 | −0.013 | 1.61 | 92 | 63 | 0.108 | 0.320 |
| D | 200 | 85 | 0.79 | 0.480 | 0.570 | −0.025 | 1.22 | 105 | 78 | 0.79 | 1.28 |
| D | 300* | 96 | 0.77 | 0.505 | 0.526 | −0.016 | 1.08 | 118 | 98 | 2.52 | 3.68 |

\* = 94–99% inside the block; outside cells counted as open.

Timings are one core on a shared machine, so read them as ±50%. At these sizes a sparse direct solve of the same window costs about the same as the FFT. In 2D the FFT buys matrix-free memory and easy batching, not speed.

### 2.3 Variation across 5810 (`tiles.py`: every fully-inside tile, 4,417 tiles in 8 s on 8 workers)

| tile | n | mean σ_iso | sd | p10 / p50 / p90 | anisotropy p50 / p90 | blocked one way | CG it p50 / max |
|---|---|---|---|---|---|---|---|
| 12.5 m | 3,430 | 0.524 | 0.270 | 0.17 / 0.51 / 0.94 | 1.27 / 3.62 | 264 | 36 / 93 |
| 25 m | 790 | 0.493 | 0.204 | 0.24 / 0.48 / 0.76 | 1.22 / 2.00 | 18 | 47 / 98 |
| 50 m | 164 | 0.493 | 0.139 | 0.32 / 0.50 / 0.67 | 1.13 / 1.39 | 0 | 54 / 81 |
| 100 m | 33 | 0.489 | 0.116 | 0.34 / 0.50 / 0.61 | 1.10 / 1.20 | 0 | 67 / 98 |

σ_iso = (sxx + syy) / 2.

- **Magnitude.** The densest 200 m window (A: 168 buildings/ha, 64% open) conducts 0.33; B and C (120–132/ha, 76–79% open) conduct 0.51–0.55. The effective value sits well below the open fraction, because the min-rule faces and the tortuosity of the lanes set it, not the built share alone.
- **Anisotropy.** It is strong at the scale of a few buildings and mild at neighbourhood scale (table above).
- **Direction follows the lanes.**
  - The lane axis of each tile was measured independently of the solve: the low-variation eigenvector of the structure tensor of the open fraction.
  - On anisotropic tiles the conductivity's major axis lies near the lane axis (table below).
  - At 50–100 m both axes cluster at 90–105°, a north–south grain across the whole block (84 of 164 lane axes fall in 90–105°). So the shuffled baseline falls too: at that scale the alignment is block-wide, not tile by tile.
  - `fig_tensor_field.png` shows the field and a 100 m zoom with both axes drawn.

| tiles | anisotropy ≥ | n | mean angle to lane axis | shuffled | within 22.5° |
|---|---|---|---|---|---|
| 12.5 m | 1.25 | 1,574 | 22.1° | 43.9° | 64% |
| 12.5 m | 2.0 | 479 | 18.6° | 43.3° | 71% |
| 25 m | 1.25 | 331 | 22.1° | 41.5° | 65% |
| 25 m | 2.0 | 66 | 15.4° | 43.4° | 76% |
| 50 m | 1.1 | 103 | 22.4° | 32.8° | 72% |
| 100 m | 1.0 | 33 | 17.3° | 21.5° | 76% |

### 2.4 Is there a representative volume?

Not in the strict sense, and that is a property of the fabric.

- **Between tiles.** The spread of σ_iso falls 0.270 → 0.204 → 0.139 → 0.116 for L = 12.5, 25, 50, 100 m. For stationary fabric with short correlations it would halve with each doubling of L; from 50 to 100 m it falls only 17%.
  - Neighbourhoods differ: A is 0.33, B and C are 0.51–0.55.
  - So the tensor is a field with real large-scale structure. That is good for a two-scale model and rules out a single representative volume.
- **Within a neighbourhood.** Nested windows settle to within about ±10% by 50–100 m: B 0.52 / 0.51 / 0.55 / 0.54 at 50, 100, 200, 300 m; C 0.55 / 0.53 / 0.51 / 0.54.
- **Boundary-condition dependence** (the classical test; Huet 1990, Kanit et al. 2003). Periodic and oversampled tensors were compared; oversampled means solving on a 2L window and averaging flux over the central L (Wen, Durlofsky & Edwards 2003). They differ by:
  - 0.048 at 25 m (8.1% median, 672 tiles);
  - 0.031 at 50 m (5.3%, 138 tiles).
- **Blocked windows.**
  - At 25 m, 61 of 23,773 moving windows conduct nothing at all (inside the compound's large buildings); the macro problem has no equation there.
  - The prototype adds 10⁻³ I to every element tensor, which moves J₂ by about 0.6%.
  - No window of 50 m or more was blocked.
- **Working answer.** 50 m is the smallest safe window. 100 m is where the interior error stops growing with depth (Section 3.3).

### 2.5 Cost per window (`bench.py`, one core, two loads, tolerance 1e-8)

| window | 25 m | 50 m | 100 m | 200 m | 400 m | 600 m |
|---|---|---|---|---|---|---|
| CG iterations | 51 | 54 | 63 | 88 | 123 | 139 |
| seconds | 0.010 | 0.026 | 0.175 | 1.37 | 8.8 | 21.8 |

Moving-window fields over 5810 carry three right-hand sides each: two loads plus the source corrector. On 8 workers they took:

| window | stride | windows | wall time | per window-core |
|---|---|---|---|---|
| 25 m | 5 m | 23,773 | 32 s | 11 ms |
| 50 m | 10 m | 6,119 | 32 s | 42 ms |
| 100 m | 10 m | 6,119 | 167–333 s (load-dependent) | 0.22 s |
| 200 m | 20 m | 1,612 | 409–434 s | 2.0 s |

## 3. The two-scale model

### 3.1 Formulation (`ts.py`)

- **Macro problem.** −div(σ(x) ∇U) = ρ in the region, with U = 0 on the exit. σ is the tensor field from the moving windows and ρ is the fine injection.
  - Discretized with bilinear (Q1) finite elements of side H, one tensor per element (bilinear interpolation from window centres).
  - Injection is loaded onto nodes with bilinear weights, so the total is exact.
  - U = 0 at nodes outside the block or on ground. Same pyamg CG solver.
  - Q1 handles the off-diagonal term with no special stencil.
- **Recovery** (first-order expansion; Bensoussan, Lions & Papanicolaou 1978; Allaire 1992): u(y) ≈ U(x) + χ(y)·∇U(x) + w(y).
  - χ are the periodic correctors the FFT already returns.
  - w is a local source corrector: A w = f − g on the window, with g the window's injection spread over each conducting component by open fraction. It represents a pocket's own congestion.
  - Each window keeps only its central s × s block, normalized to zero mean (an oversampling choice, as in Hou & Wu 1997).
  - A home's uᵢ is read exactly as the metric reads it.
- **Three treatments of windows that reach outside the block:**
  - *open-outside*: outside cells are open street.
  - *extrapolated*: windows less than 90% inside take the nearest interior window's tensor.
  - *inside-avg*: σ = ⟨q⟩_in ⟨e⟩_in⁻¹ over the window's inside cells.

### 3.2 Is the scalar model the right thing to homogenize?

The baseline of the lifted K = 8 'uni' metric was solved on the CPU (`lifted_ref.py`): 14,366,296 unknowns, 277 s, 11.7 GB. P = 1.89452e6, matching the stored P0 of 1,894,516.7.

- **Per home, scalar vs lifted:** Spearman 0.997. Lifted/scalar ratio median 0.80 (p10–p90 0.74–0.89). By depth band: 0.82, 0.81, 0.80, 0.80.
- **Lens A, two series against the lifted metric.**
  - The four 5810 clearings: scalar 0.181–0.268 against lifted 0.222–0.343, i.e. 0.04–0.08 lower. Same order: SIMP > greedy at both budgets.
  - Gated blocks: 22422 step 1, 0.371 vs 0.534; 30848 step 5, 0.537 vs 0.672.
- So the scalar model ranks homes like the real metric but does not reproduce its Lens A levels. Homogenizing lifted 'uni' itself would need K-component cell problems, which the FFT can handle with a K × K symbol per frequency; this was not built.
- *Speculation:* the sightline model's factors depend on straight runs of up to about 100 m, so it may have no scale separation at 50 m windows.

### 3.3 Two-scale J₂ on all of 5810 (`twoscale_run.py`, `twoscale.csv`)

Macro elements H = 2 m (142,840 unknowns, 1.4 s). Errors are against the fine scalar solve. Homes deeper than 100 m carry 81% of J₂; homes within 10 m of the exit carry 0.01%; the north-east strip carries 0.12%.

| L | near-exit tensor | J₂ err | P err | Spearman | median abs rel | median rel <10 m | 10–30 m | 30–100 m | >100 m | NE strip |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | open-outside | +12.9% | +6.2% | 0.9987 | 5.0% | +34% | +8.0% | +4.5% | +4.6% | +49% |
| 25 | extrapolated | +15.2% | +7.7% | 0.9982 | 6.0% | +81% | +21% | +6.2% | +5.2% | +87% |
| 25 | inside-avg | +14.3% | +7.1% | 0.9985 | 5.5% | +63% | +15% | +5.4% | +4.9% | +68% |
| 50 | open-outside | +4.0% | +1.9% | 0.9982 | 2.4% | +31% | +4.8% | +0.8% | +1.8% | +62% |
| 50 | extrapolated | +11.4% | +6.5% | 0.9969 | 5.3% | +97% | +33% | +6.4% | +4.0% | +239% |
| 50 | inside-avg | +8.4% | +4.7% | 0.9979 | 3.9% | +83% | +25% | +4.2% | +3.1% | +133% |
| 100 | open-outside | −6.1% | −3.4% | 0.9967 | 3.2% | +27% | −0.8% | −4.6% | −1.7% | +50% |
| 100 | extrapolated | +2.5% | +2.2% | 0.9972 | 2.7% | +94% | +35% | +3.4% | +1.7% | +139% |
| 100 | inside-avg | +4.8% | +3.5% | 0.9972 | 3.3% | +91% | +34% | +4.8% | +2.2% | +195% |
| 200 | open-outside | −20.1% | −11.4% | 0.9918 | 9.6% | +12% | −14% | −13% | −7.1% | −4% |
| 200 | extrapolated | −6.7% | −2.7% | 0.9939 | 4.2% | +50% | +15% | −0.1% | −1.0% | +65% |
| 200 | inside-avg | +0.4% | +1.4% | 0.9962 | 3.3% | +87% | +34% | +4.5% | +1.2% | +131% |

- **Macro resolution.** Coarser macro elements add error mostly at the boundary. At L = 50 m open-outside, J₂ error is +4.0% / +5.2% / +9.8% at H = 2 / 5 / 10 m, with 142,840 / 22,848 / 5,717 unknowns.
- **Correctors.** χ and w change J₂ by less than 0.3% in every variant. χ halves the median error of homes within 10 m of the exit (for example +31% → +20%).
- **Error decomposition** (median two-scale minus fine u, by distance to exit):

| distance band (m) | 0–10 | 10–30 | 30–60 | 60–100 | 100–150 | 150–200 | 200–300 |
|---|---|---|---|---|---|---|---|
| fine median u | 7.7 | 26.8 | 152 | 294 | 448 | 591 | 705 |
| L 25, open-outside | +1.9 | +2.0 | +5.4 | +12.3 | +21.3 | +26.5 | +31.7 |
| L 50, open-outside | +1.7 | +1.5 | +0.4 | +3.0 | +6.4 | +9.8 | +12.7 |
| L 100, extrapolated | +5.2 | +8.6 | +8.9 | +7.1 | +7.1 | +10.3 | +11.4 |
| L 200, inside-avg | +2.6 | +9.5 | +11.2 | +5.7 | +3.2 | +8.0 | +9.2 |
| L 200, open-outside | −0.8 | −2.1 | −16.1 | −32.1 | −38.9 | −37.7 | −39.0 |

- **What the decomposition shows.**
  - With 25–50 m windows the error grows with depth: the interior tensor is biased low, as if long lanes spanning more than one window were being missed.
  - With 100–200 m windows (extrapolated or inside-avg) the error is a roughly constant offset of +5 to +11 reached within about 30 m of the exit: the interior has converged.
  - That offset is a boundary layer. It is negligible for deep homes (1–2%) but triples the u of homes within 10 m of the exit (fine median 7.7).
  - Counting outside cells as open inflates σ in a band L/2 wide, which is why large windows with open-outside go strongly negative.
- See `fig_homes.png`.

### 3.4 Lens A of real clearings (`clearings_run.py`, `clearings.csv`)

Stored clearings from `clear_rows_*`, optimized under lifted 'uni'. Macro H = 2 m. Only the windows touching cleared buildings were recomputed.

| clearing | bldgs | lifted (stored) | fine scalar | L50 open | L50 inside-avg | L100 open | L100 inside-avg | L25 open |
|---|---|---|---|---|---|---|---|---|
| SIMP D 0.05 | 355 | 0.251 | 0.189 | 0.177 | 0.182 | 0.155 | 0.169 | 0.190 |
| SIMP D 0.10 | 680 | 0.343 | 0.268 | 0.258 | 0.264 | 0.228 | 0.246 | 0.276 |
| greedy D 0.05 | 203 | 0.222 | 0.181 | 0.177 | 0.181 | 0.152 | 0.166 | 0.190 |
| greedy D 0.10 | 524 | 0.320 | 0.257 | 0.253 | 0.259 | 0.225 | 0.242 | 0.269 |

- With 50 m windows inside-avg, Lens A is within 0.007 of the fine scalar value.
- 100 m windows underestimate by 0.015–0.04. Larger windows dilute a localized opening.
- The gap the design cares about collapses. SIMP minus greedy is 0.0074 fine at D 0.05 against 0.0009 two-scale (L50 inside-avg); at D 0.10 it is 0.0103 against 0.0058.
- Cost per evaluation: windows 5–8 s on 16 workers (2,200–3,400 windows) plus macro 1.4 s, against a fine scalar solve of 11.9 s. A block-wide clearing is not cheaper to evaluate two-scale.

### 3.5 Where scale separation fails

- **Gates** (`gated_run.py`, `gated.csv`, L = 50 m, H = 2 m):

| block | state | lifted | fine scalar | two-scale (open / inside-avg) |
|---|---|---|---|---|
| 22422 | J₀ error | – | – | −51% / −49% |
| 22422 | step 1 (1 bldg, D 0.01: the gate) | 0.534 | 0.371 | 0.148 / 0.144 |
| 22422 | step 2 (3 bldgs, D 0.08) | 0.785 | 0.567 | 0.390 / 0.375 |
| 22422 | step 5 (48 bldgs, D 0.10) | 0.815 | 0.610 | 0.436 / 0.426 |
| 30848 | J₀ error | – | – | −14% / +15% |
| 30848 | step 1 (48 bldgs, D 0.01) | 0.219 | 0.240 | 0.192 / 0.200 |
| 30848 | step 2 (78 bldgs, D 0.02) | 0.305 | 0.352 | 0.287 / 0.298 |
| 30848 | step 5 (256 bldgs, D 0.05) | 0.672 | 0.537 | 0.451 / 0.464 |

  - On 22422 the 20 worst homes hold 83% of the fine J₀; two-scale gives them only 0.39–0.40 of their fine J₀ share.
  - A pocket's penalty sits behind a sub-window gap that no 50 m tensor sees. The source corrector does not recover it either.
  - On 5810, homes with the largest local pocket penalty (top 5%) are underestimated by a median 4.2%, while other homes are overestimated by 2%. The Spearman correlation between pocket penalty and error is −0.38.
  - This is the opposite failure to the Galerkin coarse model in BACKLOG: homogenization erases gates by construction.
- **Boundary layer near the exit.** Section 3.3: the offset is fixed by the window treatment, not by the fabric; it gives +27% to +97% median error for homes within 10 m of the exit.
- **Homes near the sink.** As above. They carry less than 1% of J₂, so J₂ hides the problem.
- **Narrow parts of a domain.** The north-east strip is 60–90 m wide, about one window. Its error runs from −4% to +411%.
- **Small windows.** Some 25 m windows conduct nothing at all (61 of 23,773).

### 3.6 Where clearing pays (`tiles_clear.py`, `tiles_clear.csv`, `fig_tiles.png`)

All 164 fully-inside 50 m tiles of 5810, each under two clearing rules:
- **R20:** buildings centred in the tile, in random order, until 20% of the tile's footprint area is cleared.
- **STRIP:** every building touching a 4 m strip through the tile centre, along the macro gradient.

Each clearing's fine ΔJ₂ was compared with three predictions:
- **two-scale:** recompute the touched windows (median 69), then re-solve the macro problem.
- **first-order:** one macro adjoint, with ΔJ ≈ −Σ_e S_e : Δσ_e.
- **naive proxy:** cleared area × the tile's mean baseline u.

| | R20 | STRIP |
|---|---|---|
| fine gain, median / max (% of J₂) | 0.10 / 2.4 | 0.11 / 9.9 |
| Spearman, two-scale | 0.978 | 0.991 |
| Spearman, first-order map | 0.971 | 0.990 |
| Spearman, naive proxy | 0.04 | 0.22 |
| Spearman per displaced m², two-scale / proxy | 0.973 / −0.40 | 0.991 / −0.40 |
| top-10% overlap, two-scale / first-order / proxy | 94% / 88% / 19% | 94% / 94% / 19% |
| two-scale / fine gain: median (p10–p90) | 1.11 (0.97–1.37) | 1.08 (0.96–1.28) |

- Gain is largest where the macro flow concentrates (high ∇U·∇Λ), not where homes are deepest. That is why the depth proxy fails.
- Each evaluation costs about 3 s of window solves on one core plus a 1.4 s macro solve (or nothing, using the first-order map), against 12–15 s for a fine solve.
- Not tested: a fine-scale adjoint map. It would need an eps world for closed buildings, the "grey leak" NOTES describes at gates.

## 4. Running it on all of Cape Town

### 4.1 Problem size

Built from `research/roadless/block_sizes.parquet` (footprint tier, at least 10 buildings) joined to `~/.cache/reblock/blocks_capetown_full.parquet` areas. Caveat: that file gives 5810 834k m² against 577k inside its kblock outline, so these areas may be overstated by up to about 1.4×.

| | blocks | area | buildings |
|---|---|---|---|
| all footprint-tier blocks | 28,768 | 4,971 km² | 1.92M |
| built-up blocks (≥ 5 buildings/ha) | 26,008 | 829 km² | 1.71M |
| low-density blocks (< 5/ha) | 2,760 | 4,142 km² | 215k |

- Built-up blocks: median 0.017 km², p90 0.058, p99 0.30, maximum 5.9 km². Only 10 exceed 1.4 km² (25 km² in all).
- Under per-block exits (the current model), 90% of built-up blocks are about 1–5 window widths across. There is no scale separation and nothing for homogenization to do there.
- It becomes relevant for exits at region boundaries. 5810@major (NOTES, 2026-10-10) is 6.74 km², 12,663 buildings and 27M uniform cells.

### 4.2 Cost

CPU figures are measured and scaled linearly; the GPU figure is speculation.

| task | rate measured on 5810 | all built-up Cape Town (3.3e9 cells) |
|---|---|---|
| fine scalar solve | 12 s, 1.4 GB per 2.31M cells | ≈ 5 CPU-h. Blocks are independent. Largest block about 14 GB. 5810@major about 2.5 min and about 16 GB. |
| fine lifted 'uni', CPU | 277 s, 11.7 GB | ≈ 110 CPU-h. The largest block needs about 120 GB, more than one 48 GB card. |
| tensor field, 50 m windows at 25 m stride | 42 ms per window-core | 1.33M windows ≈ 16 CPU-h (about 20 min on 48 cores). At 10 m stride: about 97 CPU-h. |
| tensor field, 100 m windows at 25 m stride | 0.22 s per window-core | ≈ 81 CPU-h |
| macro solve | 143k unknowns in 1.4 s at H = 2 m | city as one domain: H = 5 m is 33M unknowns, est. about 6 min; H = 2 m is 207M, est. about 35 min and over 100 GB |
| GPU (*speculative, not measured*) | bandwidth arithmetic: about 0.5 ms per batched 50 m window | 1.33M windows ≈ 10 min on one card, roughly this 48-core machine. Not needed. |

Storing correctors at full resolution would take about 40 GB (float32). Recompute them on demand instead.

### 4.3 How design would use it

1. Compute the tensor field once for the city, about 16 CPU-hours.
2. Per region: one macro solve and one adjoint give the sensitivity S(x) = −∂J/∂σ (`fig_tiles.png`, left).
3. For each candidate area, the gain is −S : Δσ, where Δσ comes from recomputing the roughly 70 windows touching the candidate (about 3 s per core) or from a precomputed standard clearing per window.
4. Rank areas by gain per displaced m² (Spearman 0.97–0.99 on 5810's tiles), pool the city budget over the ranked areas, then run the fine optimizer (SIMP or greedy, lifted metric) only inside the selected areas.

Because homogenization cannot see gates, the gated pockets need a separate cheap detector, such as homes whose fine u greatly exceeds their neighbours'.

## 5. Prior work, and how this differs from the Galerkin coarse model

- **Wiki.** Searched with `wiki search --lex` for "homogenization coarse model Galerkin multiscale", "effective conductivity tensor upscaling", "FFT Fourier homogenization", "reblock permeability coarse grid resolution gate", "AMG multigrid coarse level solver" and "Cape Town city scale region boundary exits roadless", plus a grep of all pages for homogeniz, upscal, two-scale and multiscale. Nothing on this topic; the hits are mycooc, bookgen and Kirchhoff weighting. The vault has no reblock section.
- **BACKLOG, "Not pursued".** A Galerkin coarse model: the AMG coarse level used as the design model, keeping sub-cell gates. "Coarse runs were not cheap enough to pay even where they kept gates (`.C0.75.m2` 1.36x, dominated)". BACKLOG item 4(c) names FFT homogenization as "untested, and a model change".
- **NOTES on coarser grids.** They understate permeability, by a lot on gated blocks (h 1.0: 38616 −0.313). "h 1.0 loses 30848's gate ... a gate below the cell has no value in the coarse objective." NOTES has no other mention of homogenization, upscaling or FFT.
- **How they differ:**

| | Galerkin coarse model | FFT homogenization |
|---|---|---|
| what it is | algebraic: AMG aggregates of a few cells, at most a few times coarser | scale-separated: a smooth tensor field from 25–200 m windows; macro grid 2–10 m |
| unknowns on 5810 | – | 1.8M fine → 143k / 23k / 5.7k at 2 / 5 / 10 m |
| updates | operator-dependent, rebuilt per state | local and embarrassingly parallel |
| extras | – | an adjoint map comes with the macro solution |
| gates | keeps sub-cell gates (its purpose) | erases them by design |

  So homogenization cannot do the job the Galerkin model was for: a cheap inner-loop design model that keeps gates. It can do screening, and domains too large for a fine solve.

## 6. Recommendation

- **Not worth building as a design model.**
  - Under per-block exits the blocks are too small for scale separation.
  - The fine scalar solve already costs about 5 CPU-hours for the whole city.
  - Two-scale erases the gates the optimizers exploit (−51% J₀ on 22422) and compresses differences between good clearings (0.007 → 0.001).
- **Possibly worth building as a screening layer**, if exits move to region boundaries (BACKLOG 4c, 5810@major) or a city-wide pooled budget is wanted.
  - It ranks where clearing pays almost exactly like the fine model on 5810 (Spearman 0.97–0.99, 94% top-decile overlap).
  - With 100–200 m windows and an inside-avg or extrapolated near-exit tensor it gets J within about 0–5% in the interior.
  - Pair it with a gate detector.
- **First experiment that decides it: 5810@major.**
  - Build the region raster with the repo's `regions.py`.
  - Solve the fine scalar model on CPU at h 0.5 (27M cells; about 2.5 min and 16 GB, estimated).
  - Build the two-scale model (L 100 m, about 12 CPU-min).
  - Compare three things:
    - J₂;
    - Lens A of the two stored region clearings (cheap preset 0.2714 and SIMP's default 0.3014 under lifted, at D 0.05);
    - the 50 m tile ranking.
  - Also check whether restricting the fine optimizer to the map's top areas keeps most of the Lens A. That last check needs GPU runs.
  - **Decision rule (mine):** keep it if the tile Spearman stays at least 0.9 and the top-area-restricted optimizer keeps at least 90% of Lens A.
  - **Cost:** about a day of work and under one CPU-hour for the comparison.
- **Cost of building it for real** (*estimate*): about 1–2 weeks. That means productionizing `fftk.py` and `ts.py` behind a Strategy, adding a tensor-field cache keyed by fabric and per-window updates. Running it for the city would take about 16–100 CPU-hours.
- **Open questions:**
  - Homogenizing the lifted model (K-component cell problems).
  - Whether the sightline model has any scale separation (*speculation:* probably not).

## 7. References (DOIs checked against Crossref)

- Allaire G. 1992. Homogenization and two-scale convergence. SIAM J Math Anal 23:1482–1518. doi:10.1137/0523084
- Bendsøe M, Kikuchi N. 1988. Generating optimal topologies in structural design using a homogenization method. CMAME 71:197–224. doi:10.1016/0045-7825(88)90086-2
- Bensoussan A, Lions J-L, Papanicolaou G. 1978. Asymptotic Analysis for Periodic Structures. North-Holland; AMS Chelsea reprint 2011, doi:10.1090/chel/374
- Brisard S, Dormieux L. 2010. FFT-based methods for the mechanics of composites: a general variational framework. Comput Mater Sci 49:663–671. doi:10.1016/j.commatsci.2010.06.009
- Brisard S, Dormieux L. 2012. Combining Galerkin approximation techniques with the principle of Hashin and Shtrikman. CMAME 217–220:197–212. doi:10.1016/j.cma.2012.01.003
- Donval E, Schneider M. 2025. Convergence of damped polarization schemes for the FFT-based computational homogenization of inelastic media with pores. IJNME 126. doi:10.1002/nme.7632
- Durlofsky L. 1991. Numerical calculation of equivalent grid block permeability tensors for heterogeneous porous media. Water Resour Res 27:699–708. doi:10.1029/91WR00107
- E W, Engquist B. 2003. The heterogeneous multiscale methods. Commun Math Sci 1:87–132. doi:10.4310/CMS.2003.v1.n1.a8
- Eyre D, Milton G. 1999. A fast numerical scheme for computing the response of composites using grid refinement. Eur Phys J Appl Phys 6:41–47. doi:10.1051/epjap:1999150
- Gélébart L, Mondon-Cancel R. 2013. Comput Mater Sci 77:430–439. doi:10.1016/j.commatsci.2013.04.046 (DOI from memory, not checked against Crossref)
- Gérard-Varet D, Masmoudi N. 2012. Homogenization and boundary layers. Acta Math 209:133–178. doi:10.1007/s11511-012-0083-5
- Hou T, Wu X-H. 1997. A multiscale finite element method for elliptic problems in composite materials and porous media. J Comput Phys 134:169–189. doi:10.1006/jcph.1997.5682
- Huet C. 1990. Application of variational concepts to size effects in elastic heterogeneous bodies. JMPS 38:813–841. doi:10.1016/0022-5096(90)90041-2
- Kabel M, Merkert D, Schneider M. 2015. Use of composite voxels in FFT-based homogenization. CMAME 294:168–188. doi:10.1016/j.cma.2015.06.003
- Kanit T, Forest S, Galliet I, Mounoury V, Jeulin D. 2003. Determination of the size of the representative volume element for random composites. IJSS 40:3647–3679. doi:10.1016/S0020-7683(03)00143-4
- Ladecký M et al. 2023. An optimal preconditioned FFT-accelerated finite element solver for homogenization. Appl Math Comput 446:127835. doi:10.1016/j.amc.2023.127835
- Leuschner M, Fritzen F. 2018. Fourier-accelerated nodal solvers (FANS) for homogenization problems. Comput Mech 62:359–392. doi:10.1007/s00466-017-1501-5
- Lucarini S, Upadhyay M, Segurado J. 2022. FFT based approaches in micromechanics: fundamentals, methods and applications. MSMSE 30:023002. doi:10.1088/1361-651X/ac34e1
- Målqvist A, Peterseim D. 2014. Localization of elliptic multiscale problems. Math Comp 83:2583–2603. doi:10.1090/S0025-5718-2014-02868-8
- Michel J-C, Moulinec H, Suquet P. 2001. A computational scheme for linear and non-linear composites with arbitrary phase contrast. IJNME 52:139–160. doi:10.1002/nme.275
- Mishra N, Vondřejc J, Zeman J. 2016. A comparative study on low-memory iterative solvers for FFT-based homogenization of periodic media. J Comput Phys 321:151–168. doi:10.1016/j.jcp.2016.05.041
- Monchiet V, Bonnet G. 2012. A polarization-based FFT iterative scheme for computing the effective properties of elastic composites with arbitrary contrast. IJNME 89:1419–1436. doi:10.1002/nme.3295
- Moulinec H, Suquet P. 1994. A fast numerical method for computing the linear and nonlinear mechanical properties of composites. C R Acad Sci Paris II 318:1417–1423 (no DOI)
- Moulinec H, Suquet P. 1998. A numerical method for computing the overall response of nonlinear composites with complex microstructure. CMAME 157:69–94. doi:10.1016/S0045-7825(97)00218-1
- Moulinec H, Silva F. 2014. Comparison of three accelerated FFT-based schemes for computing the mechanical response of composite materials. IJNME 97:960–985. doi:10.1002/nme.4614
- Renard P, de Marsily G. 1997. Calculating equivalent permeability: a review. Adv Water Resour 20:253–278. doi:10.1016/S0309-1708(96)00050-4
- Schneider M. 2017. An FFT-based fast gradient method for elastic and inelastic unit cell homogenization problems. CMAME 315:846–866. doi:10.1016/j.cma.2016.11.004
- Schneider M. 2020. Lippmann-Schwinger solvers for the computational homogenization of materials with pores. IJNME 121:5017–5041. doi:10.1002/nme.6508
- Schneider M. 2021. A review of nonlinear FFT-based computational homogenization methods. Acta Mech 232:2051–2100. doi:10.1007/s00707-021-02962-1
- Schneider M, Ospald F, Kabel M. 2016. Computational homogenization of elasticity on a staggered grid. IJNME 105:693–720. doi:10.1002/nme.5008
- Schneider M, Wicht D, Böhlke T. 2019. On polarization-based schemes for the FFT-based computational homogenization of inelastic materials. Comput Mech 64:1073–1095. doi:10.1007/s00466-019-01694-3
- To Q-D, Bonnet G. 2020. FFT based numerical homogenization method for porous conductive materials. CMAME 368:113160. doi:10.1016/j.cma.2020.113160
- Vondřejc J, Zeman J, Marek I. 2014. An FFT-based Galerkin method for homogenization of periodic media. Comput Math Appl 68:156–173. doi:10.1016/j.camwa.2014.05.014
- Wen X-H, Durlofsky L, Edwards M. 2003. Use of border regions for improved permeability upscaling. Math Geol 35:521–547. doi:10.1023/A:1026230617943
- Wicht D, Schneider M, Böhlke T. 2021. Anderson-accelerated polarization schemes for FFT-based computational homogenization. IJNME 122:2287–2311. doi:10.1002/nme.6622
- Willot F. 2015. Fourier-based schemes for computing the mechanical response of composites with accurate local fields. C R Mécanique 343:232–245. doi:10.1016/j.crme.2014.12.005
- Willot F, Abdallah B, Pellegrini Y-P. 2014. Fourier-based schemes with modified Green operator for computing the electrical response of heterogeneous media with accurate local fields. IJNME 98:518–533. doi:10.1002/nme.4641
- Zeman J, Vondřejc J, Novák J, Marek I. 2010. Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients. J Comput Phys 229:8065–8071. doi:10.1016/j.jcp.2010.07.010
- Zeman J, de Geus T, Vondřejc J, Peerlings R, Geers M. 2017. A finite element perspective on nonlinear FFT-based micromechanical simulations. IJNME 111:903–926. doi:10.1002/nme.5481

## 8. Files

All in `/tmp/claude-1641171234/-home-gchurchill-src-reblock/fe8870c0-fbd9-4712-ac98-aebcb951b199/scratchpad/fft_homog/`. Run from the repo root with `PYTHONPATH=.` for the extract and lifted scripts; the others run from this directory with the repo venv's python.

- **Data extracts:**
  - `extract.py` → `fabric_5810.npz`
  - `extract2.py` → `clearings_5810.npz`
  - `extract_gated.py` → `fabric_22422.npz`, `fabric_30848.npz`
  - `lifted_ref.py` → `lifted_ref_5810.npz`
- **Core code:**
  - `fftk.py`: the schemes and the direct reference
  - `ts.py`: fine solve, tensor fields, Q1 macro model, correctors
  - `test_fftk.py`
- **Section 2:** `windows.py` (`windows.csv`, `schemes.csv`), `schemes_c0.py`, `tiles.py` (`tiles.csv`), `bench.py` (`bench.log`), `pick.py`
- **Section 3:**
  - `twoscale_run.py` → `twoscale.csv`, `homes_*.npz`, `fields_L*_s*.npz`
  - `clearings_run.py` → `clearings.csv`
  - `gated_run.py` → `gated.csv`
  - `tiles_clear.py` and `tiles_analyze.py` → `tiles_clear.csv`, `tiles_clear_summary.csv`, `sens_map.npy`
- **Figures:** `fig_tensor_field.png`, `fig_homes.png`, `fig_tiles.png`, from `fig_tensor.py`, `fig_homes.py`, `fig_tiles.py`
