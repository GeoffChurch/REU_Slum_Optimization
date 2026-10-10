"""Serial checks of region_tiles' pieces on a few region tiles: local window hits vs
ts.windows_hit, local dk vs ts.element_tensors, macro delta vs a fresh macro solve."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import region_common as rc
import region_tiles as rt
import ts
from region_macro import MacroBase
t = time.time()
rt.setup(("50",), fine=False)
G = rt.G
print("setup", round(time.time() - t), "s; tiles", len(G["tiles"]), flush=True)
fab, mac, tf = G["fab"], G["mac50"], G["tf50"]
rng = np.random.default_rng(1)
picks = [G["tiles"][i] for i in rng.choice(len(G["tiles"]), 6, replace=False)]
for (r0, c0) in picks:
    for rule in ("R20", "STRIP"):
        t = time.time()
        ids = rt.clearing(r0, c0, rule)
        if len(ids) == 0:
            print(r0, c0, rule, "empty"); continue
        oc = rc.open_buildings(fab, ids)
        box, r_lo, c_lo, m2 = rt.changed_of(oc)
        hit = rt.windows_hit(tf, box, r_lo, c_lo)
        ref = np.argwhere(ts.windows_hit(tf, oc != fab.o))
        assert sorted(map(tuple, ref.tolist())) == sorted(hit), (len(ref), len(hit))
        D, nwin = rt.dsigma("50", oc, box, r_lo, c_lo)
        t1 = time.time() - t
        # local dk == full element_tensors
        elems, dk = mac.local_dk(D)
        full = ts.element_tensors(D, tf.s, 4, mac.NI, mac.NJ).reshape(-1, 2, 2)
        nzf = np.flatnonzero(np.abs(full.reshape(-1, 4)).max(1) > 0)
        assert np.array_equal(np.sort(elems), nzf) and np.allclose(full[elems], dk, rtol=0, atol=1e-15)
        t = time.time()
        md = mac.delta(D)
        t2 = time.time() - t
        # fresh macro with the updated block tensors
        sb = ts.fill(tf, None, True) + D
        t = time.time()
        m1 = MacroBase(sb, tf.s, 4, fab)
        t3 = time.time() - t
        dJ_fresh = m1.J0 - mac.J0
        print(f"{r0},{c0} {rule}: {len(ids)} bldg {m2:.0f} m2, {nwin} windows ({t1:.1f}s); dJ lin {md['dJ_lin']:.6g} "
              f"delta {md['dJ']:.8g} ({md['iters']} it {t2:.1f}s) fresh {dJ_fresh:.8g} ({t3:.1f}s) rel {(md['dJ']-dJ_fresh)/dJ_fresh:.1e}", flush=True)
# window_sigma on the baseline fabric reproduces the stored field's tensors (incl. edge windows)
nbi, nbj = tf.have.shape
blocks = [tuple(b) for b in np.argwhere(tf.have)]
sel = [blocks[i] for i in rng.choice(len(blocks), 40, replace=False)]
edge = [b for b in blocks if tf.inside[b] < 0.9][:20]
worst = 0.0
for bi, bj in sel + edge:
    S, S_in = rt.window_sigma(fab.o, fab.inside, tf, bi, bj)
    worst = max(worst, np.abs(S - tf.sigma[bi, bj]).max(), np.abs(S_in - tf.sigma_in[bi, bj]).max())
print(f"window_sigma vs stored field on {len(sel)} + {len(edge)} edge windows: max abs diff {worst:.2e}", flush=True)
