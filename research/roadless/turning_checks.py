"""Checks for the turning sightline (SoftSightline with turn_m): (1) the vjp against central
differences, straight and turning; (2) CPU against GPU scans; (3) a T junction: does the stem
of the T gain from the bar it runs into, as the owner asked ("corridors that connect
reinforce")?

    CUDA_PATH=/usr PYTHONPATH=. pixi run python research/roadless/turning_checks.py [cpu,gpu]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lifted  # noqa: E402

H = 0.5
K = 8


def field(seed: int = 0, n: int = 48) -> np.ndarray:
    """An n x n open fraction field: open with scattered part-open buildings."""
    rng = np.random.default_rng(seed)
    o = np.ones((n, n))
    for _ in range(14):
        r, c = rng.integers(2, n - 8, size=2)
        hh, ww = rng.integers(3, 8, size=2)
        o[r:r + hh, c:c + ww] = rng.uniform(0.0, 0.3)
    o[0], o[-1], o[:, 0], o[:, -1] = 0.0, 0.0, 0.0, 0.0
    return o


def tee(n: int = 120, w: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A T: a bar along the top, a stem down the middle meeting it. (field, stem mask, the stem
    alone as a field)."""
    bar = np.zeros((n, n), dtype=bool)
    bar[n - 10 - w:n - 10, 5:n - 5] = True
    stem = np.zeros((n, n), dtype=bool)
    stem[10:n - 10 - w, n // 2 - w // 2:n // 2 + w // 2] = True
    return (bar | stem).astype(float), stem, stem.astype(float)


def fd_check(ss: lifted.SoftSightline, o: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    A = rng.standard_normal((K, *o.shape))
    xp = ss.scans.xp
    th = ss.scans.to_host

    def loss(x):
        return float((th(ss.layers(x, H, K)) * A).sum())
    grad = th(ss.vjp(o, H, K, A))
    for trial in range(3):
        d = rng.standard_normal(o.shape) * (o > 0) * (o < 1)    # interior of the [0, 1] box
        eps = 1e-5
        num = (loss(o + eps * d) - loss(o - eps * d)) / (2 * eps)
        ana = float((grad * d).sum())
        print(f"  {ss.name} {type(ss.scans).__name__}: directional derivative numeric {num:+.6e}"
              f" analytic {ana:+.6e} rel err {abs(num - ana) / max(abs(num), 1e-30):.1e}",
              flush=True)
    del xp


def main(devices: list[str]) -> None:
    o = field()
    print("(1) vjp vs central differences (random field, 48 x 48)")
    for dev in devices:
        for turn, q in ((np.inf, 1.0), (10.0, 1.0), (10.0, 8.0)):
            fd_check(lifted.SoftSightline(beta=3.0, kappa=2.0, r0_m=8.0, hill=2.0, turn_m=turn,
                                          sharp=q, scans=lifted.scans_of(dev)), o)
    if len(devices) == 2:
        print("(2) CPU vs GPU layers and vjp")
        A = np.random.default_rng(2).standard_normal((K, *o.shape))
        for turn, q in ((np.inf, 1.0), (10.0, 1.0), (10.0, 8.0)):
            out = []
            for dev in devices:
                ss = lifted.SoftSightline(beta=3.0, kappa=2.0, r0_m=8.0, hill=2.0, turn_m=turn,
                                          sharp=q, scans=lifted.scans_of(dev))
                out.append((ss.scans.to_host(ss.layers(o, H, K)),
                            ss.scans.to_host(ss.vjp(o, H, K, A))))
            print(f"  turn {turn:g} sharp {q:g}: layers max rel diff "
                  f"{np.abs(out[0][0] - out[1][0]).max() / np.abs(out[0][0]).max():.1e}, vjp "
                  f"{np.abs(out[0][1] - out[1][1]).max() / np.abs(out[0][1]).max():.1e}")
    print("(3) T junction: mean free path along the stem, alone vs meeting the bar (m)")
    t, stem, alone = tee()
    for turn, q in ((np.inf, 1.0), (30.0, 1.0), (10.0, 1.0), (30.0, 8.0), (10.0, 8.0),
                    (10.0, 32.0)):
        ss = lifted.SoftSightline(beta=1.0, kappa=2.0, turn_m=turn, sharp=q,
                                  scans=lifted.scans_of(devices[-1]))
        Rs = {}
        for name, f in (("alone", alone), ("T", t)):
            Rmax = np.zeros(f.shape)
            for _i, R in ss.runs(f, H):
                Rmax = np.maximum(Rmax, R)
            Rs[name] = Rmax[stem]
        print(f"  turn_m {turn:>4g} sharp {q:>2g}: best line through stem cells: alone "
              f"{Rs['alone'].mean():6.1f}, in the T {Rs['T'].mean():6.1f}  "
              f"(top quarter of the stem: {np.quantile(Rs['alone'], 0.75):6.1f} -> "
              f"{np.quantile(Rs['T'], 0.75):6.1f})")


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["cpu", "gpu"])
