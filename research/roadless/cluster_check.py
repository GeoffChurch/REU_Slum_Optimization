"""Guards on cluster.py's kinds, without a submit: each plans a run from a command line, routes and
scripts it as cluster_submit's submit does (the plan a dry submit prints and the sbatch it
stages), and checks what a task would get.

    PYTHONPATH=. uv run python research/roadless/cluster_check.py cpu <ids,> <items,>
    PYTHONPATH=. uv run python research/roadless/cluster_check.py gpu <recorded run>

`cpu`: a `cpu` run on each node of CPU_NODES at its own throttle is one CPU array pinned there
(`--nodelist`), with no GPU request, the given cores and memory, every (block, item) one task that
unpacks the block's tar through cluster_unpack.py and runs the command with both placeholders
filled; a run without `--node` is refused, and one past the node's throttle. The tars are empty
stand-ins (a plan checks only that they exist).

`gpu`: a `blocks` run's submit as its record keeps it (.cluster_submit/runs/<run>.json, argv),
planned again now, gives the task lines and card sizes the record holds, and the record's
plan was made by the launcher before the `cpu` kind: the GPU path unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cluster  # noqa: E402
from cluster_submit.arrays import route, script  # noqa: E402
from cluster_submit.cli import parser  # noqa: E402
from cluster_submit.submit import throttle_checked  # noqa: E402

PROJECT = cluster.PROJECT
COMMAND = ["--", "research/roadless/transparency_run.py", "tuning", "arm", "{id}", "{item}"]


def _parsed(argv: list[str]) -> argparse.Namespace:
    return parser(PROJECT).parse_args(argv)


def _plan(argv: list[str], workdir: Path) -> tuple[argparse.Namespace, object, tuple]:
    """The kind's plan for `argv`, routed and checked as submit does (cluster_submit's route
    and throttle_checked): (args, plan, arrays)."""
    args = _parsed(argv)
    kind = {k.name: k for k in PROJECT.kinds}[args.kind]
    workdir.mkdir(parents=True, exist_ok=True)
    plan = kind.plan(args, workdir)
    arrays, too_big = route(PROJECT.cluster, plan.tasks, node=plan.resources.node)
    if too_big:
        raise SystemExit(f"{len(too_big)} tasks too big for any card")
    throttle_checked(PROJECT, arrays, args.gpus)
    return args, plan, arrays


def _refused(argv: list[str], workdir: Path) -> str:
    """The refusal `argv` meets (argparse's or a plan's), or raise if it plans."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            _plan(argv, workdir)
    except SystemExit as e:
        return f"{e.code}: {err.getvalue().strip().splitlines()[-1] if err.getvalue() else ''}"
    raise AssertionError(f"planned, not refused: {argv}")


def check_cpu(ids: list[str], items: list[str]) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for b in ids:
            with tarfile.open(tmp / f"{b}.tar", "w"):
                pass
        common = ["--cpus", "1", "--mem", "4", "--time", "2:00:00", "--items", ",".join(items),
                  "--unpack", str(tmp / "{id}.tar"), ",".join(ids), *COMMAND]
        for node, at_once in cluster.CPU_NODES.items():
            argv = ["submit", "cpu", "guard", "--dry", "--node", node, "--gpus", str(at_once),
                    *common]
            args, plan, arrays = _plan(argv, tmp / "work")
            [a] = arrays
            text = script(PROJECT, "guard", "0" * 40, a, plan.resources, plan.prelude, args.gpus)
            head = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
            want = {f"#SBATCH --nodelist={node}", "#SBATCH --cpus-per-task=1",
                    "#SBATCH --mem=4G", f"#SBATCH --array=0-{len(ids) * len(items) - 1}%{at_once}"}
            gpu = [ln for ln in head if "gres" in ln or "--gpus" in ln]
            lines = [f'python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/'
                     f'{b}.tar" && python -u research/roadless/transparency_run.py tuning arm '
                     f'{b} {item}' for b in ids for item in items]
            assert a.card_gb == 0 and a.node == node, f"{node}: {a.label} array on {a.node}"
            assert want <= set(head), f"{node}: header lacks {want - set(head)}"
            assert not gpu, f"{node}: a GPU request in a CPU array: {gpu}"
            assert list(a.tasks) == lines, f"{node}: task lines {a.tasks[:2]}"
            assert plan.resources.node == node and all(t.cpu for t in plan.tasks)
            print(f"{node}: one CPU array of {len(a.tasks)} tasks pinned there, at most "
                  f"{at_once} at once, 1 cpu and 4 GB a task, no GPU request")
            over = ["submit", "cpu", "guard", "--dry", "--node", node, "--gpus", str(at_once + 1),
                    *common]
            print(f"  --gpus {at_once + 1} refused: {_refused(over, tmp / 'work')}")
        bare = ["submit", "cpu", "guard", "--dry", "--gpus", "4", *common]
        print(f"without --node refused: {_refused(bare, tmp / 'work')}")


def check_gpu(run: str) -> None:
    rec = json.loads((cluster.REPO / ".cluster_submit" / "runs" / f"{run}.json").read_text())
    argv = list(rec["argv"])
    if argv[:2] != ["submit", "blocks"]:
        raise SystemExit(f"{run} is not a blocks run: {argv[:2]}")
    argv[2] = f"{run}-check"
    with tempfile.TemporaryDirectory() as d:
        args, plan, arrays = _plan(argv[:3] + ["--dry"] + argv[3:], Path(d))
    got = [(a.card_gb, list(a.tasks), a.node, sorted(a.exclude)) for a in arrays]
    want = [(a["card_gb"], list(a["tasks"])) for a in rec["arrays"]]
    assert [(c, t) for c, t, *_ in got] == want, "the plan's arrays differ from the record's"
    assert plan.resources == cluster.cs.Resources(cpus=4, mem_gb=64, time=args.time, node=None)
    for c, t, node, exclude in got:
        print(f"{run}: gpu{c} array of {len(t)} tasks, on {node or 'any node'} but "
              f"{','.join(exclude)}: the record's tasks and card size")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cpu":
        check_cpu(a[1].split(","), a[2].split(","))
    elif a[0] == "gpu":
        check_gpu(a[1])
    else:
        raise SystemExit(f"unknown check {a[0]!r}")
