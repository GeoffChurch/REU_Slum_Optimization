"""Run the roadless research scripts on the Slurm cluster (`ssh ai`): one block per GPU task.

A run is a script command with `{id}` in it and a list of block ids. `submit` pushes the code
(tracked files at a clean HEAD), ships a bank of exactly those blocks (a node has no source data,
common.build_blocks reads the bank), sorts the blocks by the GPU memory they need (the eps world
holds every inside cell x 8 headings: ~1 GB per million unknowns), and submits one job array
per memory class to the nodes whose cards hold it, the classes chained so that at most MAX_GPUS
tasks run at once. The scripts write their rows where they would here (the cluster checkout's
research/roadless/); `sync` pulls new parquet files (never overwriting a local one) and the
task logs. The run's record (commit, command, job ids, blocks per class) is
cluster_runs/<name>.json.

Generic parts borrowed from GeoffChurch/bookgen's ltcluster (a registry per run, a clean tree
and its commit on every task line, an allow-listed pull) and GeoffChurch/mycooc's
mycoocluster (job arrays; always --mem; python -u; measured card sizes: h100 80 GB, quadro1-2
48 GB, tesla1-2 32 GB, orion broken). A shared launcher package is on BACKLOG.md.

    pixi run python research/roadless/cluster.py setup
    pixi run python research/roadless/cluster.py submit <name> <ids,|@file> <time> -- <script> <args with {id}>
    pixi run python research/roadless/cluster.py status|sync|cancel <name>
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

HOST = "ai"
REMOTE = "src/reblock"                  # the checkout, under the cluster home
RUNS = "reblock-runs"                   # the cluster's per-run tasks, bank and logs
LOCAL = HERE / "cluster_runs"
MAX_GPUS = 4                            # owner, 2026-10-03: the cluster is shared
CUPY = "cupy-cuda12x[ctk]==14.2.0"      # as here, with the CUDA 12 toolkit from pip: the cluster's
                                        # /usr/local/cuda is 11.8 (nodes' drivers are 12.8)
# (most million unknowns, class, where): a block goes to the first class that holds it
CLASSES = [(26.0, "32", ["--exclude=orion"]),
           (50.0, "48", ["--exclude=orion,tesla1,tesla2"]),
           (85.0, "80", ["--nodelist=h100"])]


def ssh(cmd: str) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], text=True, capture_output=True)
    if r.returncode:
        raise SystemExit(f"on {HOST}: {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def push() -> str:
    """The tracked files at HEAD to the cluster checkout; refuses uncommitted changes."""
    if subprocess.run(["git", "diff", "--quiet", "--ignore-submodules=none", "HEAD"],
                      cwd=REPO).returncode:
        raise SystemExit("commit first: the tasks name the commit they run")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True,
                          capture_output=True).stdout.strip()
    files = subprocess.run(["git", "ls-files", "--recurse-submodules", "-z"], cwd=REPO,
                           check=True, capture_output=True).stdout      # ext/ are submodules
    ssh(f"mkdir -p {REMOTE}")
    subprocess.run(["rsync", "-a", "--from0", "--files-from=-", ".", f"{HOST}:{REMOTE}/"],
                   cwd=REPO, input=files, check=True)
    ssh(f"echo {head} > {REMOTE}/.cluster_commit")
    return head


def setup() -> None:
    """Once: the code, pixi, the locked env, cupy, then a probe on a GPU node."""
    push()
    ssh("test -x ~/.pixi/bin/pixi || curl -fsSL https://pixi.sh/install.sh | bash")
    print(ssh(f"cd {REMOTE} && ~/.pixi/bin/pixi install --frozen 2>&1 | tail -3"))
    deps = "~/.cache/reblock-research/pydeps"
    print(ssh(f"cd {REMOTE} && ~/.pixi/bin/pixi run --frozen python -m pip install -q "
              f"--target {deps} '{CUPY}' 2>&1 | tail -3"))
    # the [ctk] extra drags numpy in, which would shadow the env's own on PYTHONPATH
    ssh(f"rm -rf {deps}/numpy {deps}/numpy-*.dist-info {deps}/numpy.libs")
    probe = ("import cupy as cp; d = cp.cuda.runtime.getDeviceProperties(0); "
             "print(d['name'].decode(), d['totalGlobalMem'] // 2**30, 'GB', "
             "float(cp.arange(10.0).sum()))")
    print(ssh(f"cd {REMOTE} && srun --gres=gpu:1 --mem=4G -t 5 --exclude=orion bash -c "
              f"'source research/roadless/cluster_env.sh && "
              f"~/.pixi/bin/pixi run --frozen python -c \"{probe}\"' 2>&1 | grep -v WARN"))


def _class(million_unknowns: float) -> tuple[str, list[str]] | None:
    for most, name, where in CLASSES:
        if million_unknowns <= most:
            return name, where
    return None


def submit(name: str, ids: list[str], limit: str, command: list[str]) -> None:
    import common
    if "{id}" not in " ".join(command):
        raise SystemExit("the command needs {id}")
    run = LOCAL / name
    if (LOCAL / f"{name}.json").exists():
        raise SystemExit(f"run {name} exists")
    run.mkdir(parents=True)
    head = push()
    bank = common.write_bank(ids, run / "bank.pkl")
    by_class: dict[str, list[str]] = {}
    where_of: dict[str, list[str]] = {}
    too_big = []
    for bid in ids:
        mu = bank[bid].boundary.area / 0.25 * 8 / 1e6
        c = _class(mu)
        if c is None:
            too_big.append(bid)
            continue
        by_class.setdefault(c[0], []).append(bid)
        where_of[c[0]] = c[1]
    remote = f"{RUNS}/{name}"
    for cls, bids in by_class.items():
        (run / f"{cls}.tasks").write_text(
            "".join(" ".join(command).replace("{id}", b) + "\n" for b in bids))
    ssh(f"mkdir -p {remote}")
    subprocess.run(["rsync", "-a", f"{run}/", f"{HOST}:{remote}/"], check=True)
    jobs, prev = {}, None
    for cls in sorted(by_class):
        args = ["sbatch", "--parsable", f"--job-name=rb-{name}-{cls}",
                f"--array=0-{len(by_class[cls]) - 1}%{MAX_GPUS}", f"--time={limit}",
                f"--output={remote}/%A_%a.out", *where_of[cls]]
        if prev is not None:
            args.append(f"--dependency=afterany:{prev}")
        args += [f"{REMOTE}/research/roadless/cluster_task.sbatch",
                 f"$HOME/{remote}/{cls}.tasks", f"$HOME/{remote}/bank.pkl"]
        prev = ssh(" ".join(args)).strip().split(";")[0]
        jobs[cls] = dict(job=prev, ids=by_class[cls])
        print(f"class {cls} GB: {len(by_class[cls])} blocks, job {prev}")
    (LOCAL / f"{name}.json").write_text(json.dumps(dict(
        name=name, commit=head, command=command, time=limit, submitted=time.strftime("%F %T"),
        jobs=jobs, too_big=too_big), indent=1) + "\n")
    if too_big:
        print(f"too big for any card ({len(too_big)}): {too_big}")


def _record(name: str) -> dict:
    return json.loads((LOCAL / f"{name}.json").read_text())


def status(name: str) -> None:
    rec = _record(name)
    ids = ",".join(j["job"] for j in rec["jobs"].values())
    print(ssh(f"sacct -X -j {ids} --format=JobID%18,State,Elapsed,NodeList -n | "
              f"awk '{{print $2}}' | sort | uniq -c"))
    print(ssh(f"squeue -j {ids} -o '%.18i %.8T %.10M %R' 2>/dev/null | head -12"))


def sync(name: str) -> None:
    """New parquet rows into the local research/roadless/ (never overwriting), and the logs."""
    subprocess.run(["rsync", "-am", "--ignore-existing", "--include=*/", "--include=*.parquet",
                    "--exclude=*", f"{HOST}:{REMOTE}/research/roadless/", f"{HERE}/"],
                   check=True)
    (LOCAL / name / "logs").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include=*.out", "--exclude=*",
                    f"{HOST}:{RUNS}/{name}/", f"{LOCAL / name / 'logs'}/"], check=True)
    logs = sorted((LOCAL / name / "logs").glob("*.out"))
    bad = [p.name for p in logs if "FAILED" in p.read_text() or "Traceback" in p.read_text()]
    print(f"{len(logs)} task logs; with a failure: {bad}")


def cancel(name: str) -> None:
    ids = " ".join(j["job"] for j in _record(name)["jobs"].values())
    print(ssh(f"scancel {ids}; echo cancelled {ids}"))


if __name__ == "__main__":
    verb = sys.argv[1]
    if verb == "setup":
        setup()
    elif verb == "submit":
        cut = sys.argv.index("--")
        name, spec, limit = sys.argv[2:5]
        ids = (Path(spec[1:]).read_text().split() if spec.startswith("@")
               else spec.split(","))
        submit(name, ids, limit, sys.argv[cut + 1:])
    elif verb == "status":
        status(sys.argv[2])
    elif verb == "sync":
        sync(sys.argv[2])
    elif verb == "cancel":
        cancel(sys.argv[2])
    else:
        raise SystemExit(f"unknown verb {verb!r}")
