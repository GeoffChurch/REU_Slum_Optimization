"""The no-buildings prior on the gap complex (phase 2b): the same homes' flow to the same edge with
every building gone (every footprint cleared on the fixed mesh), its crossing current per pixel as
export.py paints the page's layers (info.json's frame)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_maps  # noqa: E402
import metrics  # noqa: E402

OUT = Path(sys.argv[1])
info = json.loads((OUT / "layers" / "info.json").read_text())
[b] = common.build_blocks([info["block"]])
m = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU()).clearing(
    b, demand=common.PopulationDemand())
fr = gap_maps.Frame.around(gc.geometry(b), info["h"], 0.0)
st = m.state(np.ones(m.n))
f = gap_maps.flow_of(m, st, m.pop)
s = gap_maps.flow_image(m, st, f, fr).ravel() * info["h"]
np.savez_compressed(OUT / "raw_prior.npz", open_uni=np.nan_to_num(s).astype("float32"))
print("P open", float(m._rhs(st, m.pop) @ f.u), flush=True)
