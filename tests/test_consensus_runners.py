"""scripts/consensus_matrix.py's runners and checkpoint: where each recipient runs changes nothing
written, the output is in study order whatever order recipients finish in, a resumed run writes
what an uninterrupted one does, and no worker outlives the run that forked it."""
from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import scripts.consensus_matrix as study
from reblock.compare import load_permeability_config
from reblock.contracts import Block, Method
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from tests.test_consensus_matrix import _slab, _study
from tests.transplant.pool_fixtures import pool

ROOT = Path(__file__).resolve().parents[1]
PCFG = load_permeability_config()
RECIPIENTS = tuple(replace(_slab(w, h), block_id=f"r{i}")
                   for i, (w, h) in enumerate(((6, 5), (5, 4), (7, 3)), 1))
ORDER = tuple(b.block_id for b in RECIPIENTS)


def _study_of(runner: study.Runner, out: Path) -> study.Study:
    """Two real arms on synthetic recipients; the pool is never read."""
    arms: dict[str, Method] = {
        name: ClearanceReblocker(substrate=ChordSubstrate(), repulsion=repulsion, depth_target=1,
                                 max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M)
        for name, repulsion in (("own", 0.0), ("repelled", 3.0))}
    return study.Study(pool=pool(RECIPIENTS, with_paths=()), arms=arms, reference="own",
                       pcfg=PCFG, recipients=len(RECIPIENTS), runner=runner, out=out)


def _run(runner: study.Runner, out: Path) -> bytes:
    s = _study_of(runner, out)
    checkpoint = study.Checkpoint.resume(out, ORDER)
    runner.run(s, RECIPIENTS, checkpoint.record)
    return out.read_bytes() + checkpoint.skips_path.read_bytes()


def test_forked_writes_what_serial_does(tmp_path: Path) -> None:
    """The same bytes, row for row: forking changes where a recipient runs, and nothing else.

    FAULT INJECTION: submitting `range(len(blocks) - 1)` in `Forked.run` loses a recipient and
    fails this.
    """
    serial = _run(study.Serial(), tmp_path / "serial.parquet")
    assert serial == _run(study.Forked(workers=2), tmp_path / "forked.parquet")


def _results() -> dict[str, study.RecipientResult]:
    s = _study_of(study.Serial(), Path("unused.parquet"))
    return {b.block_id: study.run_recipient(s, b) for b in RECIPIENTS}


def test_the_checkpoint_is_in_study_order_and_resumes_to_the_same_bytes(tmp_path: Path) -> None:
    """Recipients finish in any order; the rows are written in study order after each one. A run
    stopped after two and resumed writes the bytes an uninterrupted run does.

    FAULT INJECTION: `all_rows` iterating `self.rows` (arrival order) instead of `self.order`
    fails the order assertion, and the resumed bytes.
    """
    results = _results()
    whole = study.Checkpoint.resume(tmp_path / "whole.parquet", ORDER)
    for r in ORDER:
        whole.record(results[r])

    stopped = study.Checkpoint.resume(tmp_path / "resumed.parquet", ORDER)
    for r in ("r3", "r1"):
        stopped.record(results[r])
    assert [row["recipient"] for row in stopped.all_rows()] == (
        [row["recipient"] for row in results["r1"].rows + results["r3"].rows])

    resumed = study.Checkpoint.resume(tmp_path / "resumed.parquet", ORDER)
    assert set(resumed.rows) == {"r1", "r3"}
    resumed.record(results["r2"])
    for suffix in (".parquet", ".skips.json"):
        assert ((tmp_path / "resumed").with_suffix(suffix).read_bytes()
                == (tmp_path / "whole").with_suffix(suffix).read_bytes())


def test_a_run_stopped_between_the_two_writes_reruns_the_recipient(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipient is done once its rows are written, so the skips go first: stopped between the
    two, the recipient is rerun rather than counted done without them.

    FAULT INJECTION: writing the rows before the skips in `Checkpoint.record` fails this.
    """
    skipped = replace(_results()["r1"], skips=[study.SkipRow(recipient="r1", arm="repelled",
                                                             reason="too few donors")])
    writes = 0
    real = study._replace

    def stopped_at_the_second(path: Path, write: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise KeyboardInterrupt
        real(path, write)  # type: ignore[arg-type]

    monkeypatch.setattr(study, "_replace", stopped_at_the_second)
    with pytest.raises(KeyboardInterrupt):
        study.Checkpoint.resume(tmp_path / "out.parquet", ORDER).record(skipped)
    monkeypatch.undo()
    assert "r1" not in study.Checkpoint.resume(tmp_path / "out.parquet", ORDER).rows


def test_a_write_stopped_partway_leaves_the_last_checkpoint_whole(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FAULT INJECTION: writing straight to `path` in `_replace` leaves a torn parquet."""
    results = _results()
    checkpoint = study.Checkpoint.resume(tmp_path / "out.parquet", ORDER)
    checkpoint.record(results["r1"])
    whole = (tmp_path / "out.parquet").read_bytes()

    def torn(frame: pd.DataFrame, path: Path) -> None:
        Path(path).write_bytes(whole[:100])
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", torn)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.record(results["r2"])
    assert (tmp_path / "out.parquet").read_bytes() == whole


def test_a_checkpoint_from_another_population_is_refused(tmp_path: Path) -> None:
    """Resuming under another `recipients` would mix two populations in one file."""
    checkpoint = study.Checkpoint.resume(tmp_path / "out.parquet", ORDER)
    checkpoint.record(_results()["r1"])
    with pytest.raises(ValueError, match="does not choose"):
        study.Checkpoint.resume(tmp_path / "out.parquet", ("r2", "r3"))


def _fails_fast(s: study.Study, block: Block) -> study.RecipientResult:
    if block.block_id == "r1":
        raise RuntimeError("r1 failed")
    time.sleep(30)
    raise AssertionError("a slow recipient was left running")


def _interrupted(result: study.RecipientResult) -> None:
    raise KeyboardInterrupt


def _returns_fast(s: study.Study, block: Block) -> study.RecipientResult:
    if block.block_id != "r1":
        time.sleep(30)
    return study.RecipientResult(recipient=block.block_id, rows=[], skips=[], seconds=0.0)


@pytest.mark.parametrize(("work", "record", "raised"), [
    (_fails_fast, lambda r: None, RuntimeError),        # a worker's error
    (_returns_fast, _interrupted, KeyboardInterrupt),   # an interrupt, in the parent
], ids=["worker-error", "interrupt"])
def test_an_error_terminates_every_worker_at_once(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, work: object, record: object,
        raised: type[BaseException]) -> None:
    """Not after the recipients still running have finished: those can take a minute each.

    FAULT INJECTION: dropping the `worker.terminate()` loop in `Forked.run` makes each case wait
    out the 30 s recipients, failing the time bound.
    """
    monkeypatch.setattr(study, "run_recipient", work)
    before = set(multiprocessing.active_children())
    t0 = time.time()
    with pytest.raises(raised):
        study.Forked(workers=3).run(_study_of(study.Serial(), tmp_path / "x.parquet"),
                                    RECIPIENTS, record)  # type: ignore[arg-type]
    assert time.time() - t0 < 15
    assert set(multiprocessing.active_children()) - before == set()


def _threaded() -> list[str]:
    return [f"{p['internal_api']}={p['num_threads']}" for p in threadpool_info()
            if p["num_threads"] != 1]


def _only_pinned(s: study.Study, block: Block) -> study.RecipientResult:
    threaded = _threaded()
    if threaded:
        raise AssertionError(f"{block.block_id} ran threaded: {threaded}")
    return study.RecipientResult(recipient=block.block_id, rows=[], skips=[], seconds=0.0)


@pytest.mark.parametrize("runner", [study.Serial(), study.Forked(workers=2)],
                         ids=["serial", "forked"])
def test_every_recipient_runs_blas_on_one_thread(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path, runner: study.Runner) -> None:
    """In this process for `Serial`, which restores the caller's threads after; in every worker for
    `Forked`.

    FAULT INJECTION: deleting `threadpool_limits(limits=1)` from `_worker_init`, or the `with
    threadpool_limits(limits=1)` from `Serial.run`, fails its case; pinning `Serial` for good
    (a bare call) fails the restore.
    """
    monkeypatch.setattr(study, "run_recipient", _only_pinned)
    with threadpool_limits(limits=2):
        if not _threaded():
            pytest.skip("no BLAS or OpenMP here runs more than one thread")
        caller = threadpool_info()
        runner.run(_study_of(runner, tmp_path / "x.parquet"), RECIPIENTS, lambda r: None)
        assert threadpool_info() == caller


# A study whose recipients never finish, run in a child the test can kill outright.
_KILLED_PARENT = """
import os, sys, time
from pathlib import Path

import scripts.consensus_matrix as study
from tests.test_consensus_runners import RECIPIENTS, _study_of

def announce_and_hang(s, block):
    (Path(sys.argv[1]) / f"worker-{os.getpid()}").touch()
    time.sleep(300)

study.run_recipient = announce_and_hang
s = _study_of(study.Serial(), Path(sys.argv[1]) / "x.parquet")
study.Forked(workers=2).run(s, RECIPIENTS[:2], print)
"""


def _alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc").is_dir():
        # A zombie is dead; `os.kill(pid, 0)` would call it alive until something reaps it.
        return stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_workers_die_with_a_killed_parent(tmp_path: Path) -> None:
    """A parent killed outright runs no cleanup at all; each worker notices within a second.

    FAULT INJECTION: not starting `_exit_with_parent` in `_worker_init` leaves both workers
    sleeping, failing the time bound.
    """
    parent = subprocess.Popen([sys.executable, "-c", _KILLED_PARENT, str(tmp_path)], cwd=ROOT)
    workers: list[int] = []
    try:
        deadline = time.time() + 120
        while len(workers) < 2:
            assert parent.poll() is None, "the parent died before its workers started"
            assert time.time() < deadline, "the workers never started"
            time.sleep(0.1)
            workers = [int(p.name.removeprefix("worker-")) for p in tmp_path.glob("worker-*")]
        parent.kill()
        parent.wait()
        deadline = time.time() + 10
        while any(_alive(w) for w in workers):
            assert time.time() < deadline, f"workers {workers} outlived their parent"
            time.sleep(0.1)
    finally:
        parent.kill()
        for w in workers:
            if _alive(w):
                os.kill(w, signal.SIGKILL)


def test_the_runner_is_configuration(offline_city_cache: Path) -> None:
    """`workers` defaults to this machine's cores, and 1 runs in this process."""
    assert _study([]).workers == os.cpu_count()
    assert isinstance(study.load_study(_study(["workers=1"])).runner, study.Serial)
    assert study.load_study(_study(["workers=3"])).runner == study.Forked(workers=3)
    with pytest.raises(ValueError, match="workers must be >= 1"):
        study.load_study(_study(["workers=0"]))
