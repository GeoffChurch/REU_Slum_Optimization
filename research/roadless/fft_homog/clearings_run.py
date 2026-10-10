"""Q3: Lens A of four stored clearings of 5810 (SIMP and the greedy, D 0.05 and 0.10, chosen under
the lifted uni metric): fine scalar vs two-scale scalar, with the lifted metric's stored value.

Writes clearings.csv.
"""
import sys

import numpy as np
import pandas as pd
import ts

d = np.load("fabric_5810.npz")
o, ins, gr, f = d["ff0"].astype(float), d["inside"], d["ground"], d["f"]
own, w, live = d["owner"], d["w"], ~d["stranded"]
C = np.load("clearings_5810.npz")
H = 0.5
K = 4          # macro element 2 m
fields = sys.argv[1:] or ["fields_L50_s10.npz"]

fs0 = ts.fine_solve(o, gr, f)
J0_f = ts.J(ts.home_u(fs0.u, f, own, w, live), w, live)
rows = []
FINE = {}
for key in ("simp_D0.05", "simp_D0.10", "greedy_D0.05", "greedy_D0.10"):
    fs = ts.fine_solve(C[f"ff_{key}"].astype(float), gr, f)
    FINE[key] = (ts.J(ts.home_u(fs.u, f, own, w, live), w, live), fs.seconds)
for path in fields:
    tf = ts.load_field(path)
    J0_t = {}
    for name, ia in (("open-outside", False), ("inside-avg", True)):
        m0 = ts.macro_solve(ts.fill(tf, None, ia), tf.s, K, ins, gr, f)
        U0 = ts.evaluate(m0, o.shape)[0]
        J0_t[name] = ts.J(ts.home_u(np.where(gr, 0, U0), f, own, w, live), w, live)
    for key in ("simp_D0.05", "simp_D0.10", "greedy_D0.05", "greedy_D0.10"):
        oc = C[f"ff_{key}"].astype(float)
        Jf, fsec = FINE[key]
        changed = oc != o
        tfc = ts.update_field(tf, oc, ins, f, changed, 16)
        for name, ia in (("open-outside", False), ("inside-avg", True)):
            mac = ts.macro_solve(ts.fill(tfc, None, ia), tf.s, K, ins, gr, f)
            U = ts.evaluate(mac, o.shape)[0]
            Jt = ts.J(ts.home_u(np.where(gr, 0, U), f, own, w, live), w, live)
            row = dict(field=path, fill=name, clearing=key, n_buildings=len(C[f"ids_{key}"]),
                       lensA_lifted_stored=float(C[f"perm_{key}"]),
                       lensA_fine_scalar=1 - np.sqrt(Jf / J0_f),
                       lensA_twoscale=1 - np.sqrt(Jt / J0_t[name]),
                       windows_recomputed=int(ts.windows_hit(tf, changed).sum()),
                       fine_s=fsec, update_s=tfc.seconds, macro_s=mac.seconds)
            rows.append(row)
            print(row, flush=True)
pd.DataFrame(rows).to_csv("clearings.csv", index=False)
