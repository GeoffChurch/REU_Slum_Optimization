"""The sightline arm's cleared buildings as an SVG path in the page's frame."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = Path(sys.argv[1])
info = json.loads((OUT / "layers" / "info.json").read_text())
X0, Y1 = info["frame"][0], info["frame"][3]
[b] = common.build_blocks([info["block"]])
polys = np.asarray(b.buildings.outlines)
take = np.load(OUT / "raw_ss.npz")["cleared"]


def path(geoms):
    return " ".join("M" + " ".join(f"{x - X0:.1f} {Y1 - y:.1f}"
                                   for x, y in np.asarray(r.coords)) + "Z"
                    for gm in geoms for poly in getattr(gm, "geoms", [gm])
                    for r in [poly.exterior, *poly.interiors])


v = json.loads((OUT / "layers" / "vectors.json").read_text())
v["ss_cleared"] = path(polys[take])
(OUT / "layers" / "vectors.json").write_text(json.dumps(v))
print(len(take), "sightline cleared;", len(v["ss_cleared"]) // 1000, "k chars")
