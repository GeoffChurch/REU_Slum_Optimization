"""Score clearings of 5810@major with the real metric, as the stored region runs were scored:
lifted 'uni' (K 8, turning length 3 m) on the composite mesh h0.5a5x8, area population, J_2,
Lens A = 1 - (J / J0)^(1/2) (relax.Relaxation.exact at RTOL_SCORE, on the GPU).

    CUDA_PATH=/usr ... uv run python research/roadless/fft_homog/region_score_gpu.py \
        <in.json> [<in.json> ...] <out.csv>

each in.json: {name: [building indices]} (region_tiles.py restrict's truncations,
region_topup.py's clearings); the two stored clearings are always scored too (their stored
Lens A must reproduce). One GPU process; its device memory is sampled by nvidia-smi.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import common  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import region_common as rc  # noqa: E402
import relax  # noqa: E402


class GpuPeak:
    def __init__(self):
        self.peak, self._stop = 0, threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True).stdout.split()
            if out:
                self.peak = max(self.peak, int(out[0]))
            self._stop.wait(1.0)

    def stop(self) -> float:
        self._stop.set()
        self._th.join()
        return self.peak / 1024


if __name__ == "__main__":
    todo = {f"stored_{k}": v.tolist() for k, v in rc.stored_clearings().items()}
    for path in sys.argv[1:-1]:
        todo.update(json.loads(Path(path).read_text()))
    out = Path(sys.argv[-1])
    gpu = GpuPeak()
    with rc.Monitor("gpu scoring: Clearing on h0.5a5x8 (grid, baseline solve)"):
        t = time.time()
        c = relax._clearing(rc.BID, "gpu", "uni", common.mesh_of("0.5a5x8"))
        rel = relax.Relaxation(c, 2.0)
    print(f"{len(c.sc.grid.level)} cells, J0 {rel.J0:.8g}, {time.time() - t:.0f} s", flush=True)
    rows = []
    for name, ids in todo.items():
        r = np.zeros(c.n)
        r[np.asarray(ids, dtype=np.int64)] = 1.0
        t = time.time()
        J = rel.exact(r)
        row = dict(name=name, n_buildings=len(ids), D=float(c.cost @ r), J=J,
                   lensA=rel.perm(J), seconds=time.time() - t)
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    peak = gpu.stop()
    print(f"GPU peak {peak:.1f} GiB", flush=True)
    with open(rc.OUT / "costs.csv", "a") as f:
        f.write(f"gpu scoring: {len(rows)} exact scorings (device peak),"
                f"{sum(r['seconds'] for r in rows):.1f},,{peak:.2f},GiB on the device\n")
