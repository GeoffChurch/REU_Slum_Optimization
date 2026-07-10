# Compare methods at a budget (cost-benefit curves)

Rank the reblockers by *efficiency* — how much benefit each buys per meter of road, across the whole
budget range (not just at full build, where they converge). Grades every method on three lenses:
`access`, `efficiency` (network E), and `directness` (1/circuity).

```bash
pixi run python -m reblock.compare data=dji eval=kcomplexity \
  "block_ids=[[DJI.3_1_1808]]" methods=[dijkstra,mesh,greedy_arterial_buildable]
```

Writes, per metric, `auc_table_{metric}.csv` (mean AUC per method, higher = more benefit per meter)
and `curve_{metric}_{block}.png` (benefit vs road density, m/ha). `greedy_arterial_buildable` is the
navigability flagship — it leads on `directness` (≈10× dijkstra/mesh) at the cost of being slow and
trading a little access; `dijkstra` stays the fast access-first default.

**Directness — arterial pulls ahead:**

![directness curve](curve_directness.png)
