"""Gated blocks (22422, 30848): fine scalar vs two-scale J0 and Lens A of the greedy's first
steps (the gate is in step 1 of 22422). L 50 m windows, stride 10 m, 2 m macro elements.
Writes gated.csv."""
import numpy as np
import pandas as pd
import ts

H, K, L, S = 0.5, 4, 50, 10
rows = []
for bid in ("22422", "30848"):
    d = np.load(f"fabric_{bid}.npz")
    o, ins, gr, f = d["ff0"].astype(float), d["inside"], d["ground"], d["f"]
    own, w, live = d["owner"], d["w"], ~d["stranded"]
    fs0 = ts.fine_solve(o, gr, f)
    u0 = ts.home_u(fs0.u, f, own, w, live)
    J0f = ts.J(u0, w, live)
    tf = ts.tensor_field(o, ins, f, int(L / H), int(S / H), 6)
    base = {}
    for name, ia in (("open-outside", False), ("inside-avg", True)):
        m0 = ts.macro_solve(ts.fill(tf, None, ia), tf.s, K, ins, gr, f)
        ut0 = ts.home_u(np.where(gr, 0, ts.evaluate(m0, o.shape)[0]), f, own, w, live)
        base[name] = (ts.J(ut0, w, live), ut0)
        ok = live & (u0 > 0)
        top = np.argsort(-(w * u0 ** 2) * ok)[:20]
        rows.append(dict(block=bid, state="baseline", fill=name, n_cleared=0,
                         lensA_lifted=0.0, lensA_fine=0.0, lensA_twoscale=0.0,
                         J0_rel_err=base[name][0] / J0f - 1,
                         top20_homes_share_J0=float((w[top] * u0[top] ** 2).sum() / J0f),
                         top20_homes_twoscale_over_fine=float((w[top] * ut0[top] ** 2).sum()
                                                              / (w[top] * u0[top] ** 2).sum()),
                         windows=int(tf.have.sum()), field_s=tf.seconds, fine_s=fs0.seconds))
        print(rows[-1], flush=True)
    for step in (1, 2, 5):
        oc = d[f"ff_step{step}"].astype(float)
        fs = ts.fine_solve(oc, gr, f)
        Jf = ts.J(ts.home_u(fs.u, f, own, w, live), w, live)
        tfc = ts.update_field(tf, oc, ins, f, oc != o, 6)
        for name, ia in (("open-outside", False), ("inside-avg", True)):
            mac = ts.macro_solve(ts.fill(tfc, None, ia), tf.s, K, ins, gr, f)
            Jt = ts.J(ts.home_u(np.where(gr, 0, ts.evaluate(mac, o.shape)[0]), f, own, w, live),
                      w, live)
            rows.append(dict(block=bid, state=f"step{step}", fill=name,
                             n_cleared=int(d[f"n_step{step}"]), D=float(d[f"D_step{step}"]),
                             lensA_lifted=float(d[f"perm_step{step}"]),
                             lensA_fine=1 - np.sqrt(Jf / J0f),
                             lensA_twoscale=1 - np.sqrt(Jt / base[name][0])))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("gated.csv", index=False)
