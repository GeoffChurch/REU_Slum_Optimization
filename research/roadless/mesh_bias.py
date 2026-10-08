"""What a mesh does to Lens A: resolution_check's rows (res_rows/<along>/), each mesh's score of
every fixed clearing minus the uniform h 0.5 score of the same clearing, per mesh over the blocks
that have both (NOTES, "What a coarser grid does to Lens A").

    PYTHONPATH=. uv run python research/roadless/mesh_bias.py <along> [<mesh> ...]

Columns: median, mean and worst difference; the share of clearings off by more than 0.05; the
blocks whose every clearing is within 0.01; the median over blocks of the Kendall tau (tau-b)
between the clearings' order on the mesh and at h 0.5 (scores rounded to ROUND decimals), and the
share of blocks whose order is exactly kept (tau 1). No meshes named: every one in the rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
EXACT = "h0.5"
ROUND = 8          # decimals a score is compared at for the order (exact scores: rtol 1e-5 on u)


def rows(along: str) -> pd.DataFrame:
    d = pd.concat(pd.read_parquet(f) for f in sorted((HERE / "res_rows" / along).glob("*.parquet")))
    return d.drop_duplicates(["block", "name", "mesh"], keep="last")


def table(d: pd.DataFrame, meshes: list[str]) -> pd.DataFrame:
    exact = d[d.mesh == EXACT].set_index(["block", "name"]).perm
    out = []
    for mesh in meshes:
        m = d[d.mesh == mesh].set_index(["block", "name"]).perm
        both = m.index.intersection(exact.index)
        diff = (m[both] - exact[both]).rename("d").reset_index()
        taus = []
        for bid, g in diff.groupby("block"):
            names = g.name.tolist()
            # rounded: identical clearings score alike only to ~1e-12 (the GPU's summation order),
            # and tau-b counts such a broken tie as a discordant pair
            a = exact.loc[bid][names].to_numpy().round(ROUND)
            b = m.loc[bid][names].to_numpy().round(ROUND)
            if len(set(a)) > 1 and len(set(b)) > 1:
                taus.append(kendalltau(a, b).statistic)
        per_block = diff.groupby("block").d.apply(lambda s: s.abs().max())
        out.append(dict(mesh=mesh, blocks=diff.block.nunique(), median=diff.d.median(),
                        mean=diff.d.mean(), worst=diff.d.loc[diff.d.abs().idxmax()],
                        off_005=float((diff.d.abs() > 0.05).mean()),
                        within_001=int((per_block <= 0.01).sum()),
                        tau_median=float(np.median(taus)) if taus else np.nan,
                        order_kept=float(np.mean(np.asarray(taus) > 1 - 1e-9)) if taus else np.nan))
    return pd.DataFrame(out)


if __name__ == "__main__":
    d = rows(sys.argv[1])
    meshes = sys.argv[2:] or sorted(set(d.mesh) - {EXACT})
    pd.set_option("display.width", 200)
    print(table(d, meshes).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
