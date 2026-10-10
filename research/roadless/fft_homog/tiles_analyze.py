"""Summarize tiles_clear.csv: does the two-scale model rank tiles like the fine model?"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

t = pd.read_csv("tiles_clear.csv")
d = np.load("fabric_5810.npz")
J0 = float(np.load("tiles_base.npz")["J0f"])
out = []
for rule, g in t.groupby("rule"):
    g = g[g.n_bldg > 0]
    gain_f = -g.dJ_fine / J0
    row = dict(rule=rule, tiles=len(g), median_bldg=g.n_bldg.median(),
               median_cleared_m2=g.cleared_m2.median(),
               fine_gain_median_pct=100 * gain_f.median(), fine_gain_max_pct=100 * gain_f.max())
    for col in ("dJ_twoscale", "dJ_linear", "proxy"):
        row[f"spearman_{col}"] = spearmanr(g.dJ_fine, g[col]).statistic
        # per displaced m^2 (what a budgeted design ranks by)
        row[f"spearman_perm2_{col}"] = spearmanr(g.dJ_fine / g.cleared_m2,
                                                 g[col] / g.cleared_m2).statistic
        k = max(1, len(g) // 10)
        top_f = set(g.nsmallest(k, "dJ_fine").index)
        top_c = set(g.nsmallest(k, col).index)
        row[f"top10pct_overlap_{col}"] = len(top_f & top_c) / k
    row["ratio_twoscale_over_fine_median"] = float(np.median(g.dJ_twoscale / g.dJ_fine))
    row["ratio_linear_over_fine_median"] = float(np.median(g.dJ_linear / g.dJ_fine))
    row["ratio_twoscale_p10"], row["ratio_twoscale_p90"] = np.percentile(
        g.dJ_twoscale / g.dJ_fine, [10, 90])
    row["seconds_median"] = g.seconds.median()
    out.append(row)
r = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(r.T.to_string())
r.to_csv("tiles_clear_summary.csv", index=False)
