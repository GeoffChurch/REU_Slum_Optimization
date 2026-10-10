"""Per-building SVG paths and each arm's cleared set, for the page; context layers apart."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = Path(sys.argv[1])
info = json.loads((OUT / "layers" / "info.json").read_text())
X0, Y1 = info["frame"][0], info["frame"][3]
[b] = common.build_blocks([info["block"]])
polys = np.asarray(b.buildings.outlines)


def d(poly):
    return " ".join("M" + " ".join(f"{x - X0:.1f} {Y1 - y:.1f}"
                                   for x, y in np.asarray(r.coords)[:-1]) + "Z"
                    for p in getattr(poly, "geoms", [poly]) for r in [p.exterior, *p.interiors])


c = pd.read_parquet(HERE / "fine_clearings.parquet")
c = {n: sorted(int(i) for i in x)
     for n, x in zip(c[c.block == info["block"]].name, c[c.block == info["block"]].cleared,
                     strict=True)}
sets = {"cheap": c["cheap"], "default": c["default"],
        "ss": sorted(int(i) for i in np.load(OUT / "raw_ss.npz")["cleared"])}
(OUT / "layers" / "buildings.json").write_text(json.dumps({"d": [d(p) for p in polys], **sets}))
v = json.loads((OUT / "layers" / "vectors.json").read_text())
keep = ("outline", "neighbours", "context_buildings", "osm_major", "osm_minor", "osm_path")
(OUT / "layers" / "context.json").write_text(json.dumps({k: v[k] for k in keep}))
S = {k: set(x) for k, x in sets.items()}
print({f"{a}&{b2}": len(S[a] & S[b2]) for a in S for b2 in S if a < b2},
      {k: len(x) for k, x in S.items()})
