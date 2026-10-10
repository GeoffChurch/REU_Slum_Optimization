"""Tensor fields of 5810@major: ts.tensor_field (periodic FFT-CG cell problems on moving
windows, each giving sigma, the inside-averaged sigma_in, the correctors chi and the source
corrector w on its central s x s block), unchanged from the 5810 study.

    ... uv run python research/roadless/fft_homog/region_fields.py <workers> <L:s> [<L:s> ...]

L and s in metres (L 100: s 25; L 50: s 10, the 5810 study's; L 200: s 50). Writes
OUT/fields_L<L>_s<s>.npz.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import numpy as np  # noqa: E402
import region_common as rc  # noqa: E402
import ts  # noqa: E402

if __name__ == "__main__":
    workers = int(sys.argv[1])
    if workers > 12:
        raise SystemExit("at most 12 workers")
    fab = rc.load_fabric()
    for spec in sys.argv[2:]:
        L, s = (int(x) for x in spec.split(":"))
        n, sp_ = int(round(L / rc.H)), int(round(s / rc.H))
        path = rc.OUT / f"fields_L{L}_s{s}.npz"
        if path.exists():
            print("exists:", path, flush=True)
            continue
        with rc.Monitor(f"fields: L {L} m stride {s} m, {workers} workers") as mon:
            tf = ts.tensor_field(fab.o, fab.inside, fab.f, n, sp_, workers)
        it = tf.iters[tf.have]
        print(f"L {L} m stride {s} m: {int(tf.have.sum())} windows, {tf.seconds:.0f} s on "
              f"{workers} workers ({tf.seconds * workers / tf.have.sum() * 1e3:.0f} ms per "
              f"window-core), CG iterations median {np.median(it):.0f} max {it.max()}",
              flush=True)
        np.savez(path, sigma=tf.sigma, sigma_in=tf.sigma_in, inside=tf.inside, have=tf.have,
                 chi=tf.chi, wf=tf.wf, s=tf.s, n=tf.n, iters=tf.iters, seconds=tf.seconds,
                 workers=workers, cpu_s=mon.cost.cpu_s)
        del tf
