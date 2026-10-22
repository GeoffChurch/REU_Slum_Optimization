"""cluster.py's `wave` verb (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.8 and 3.9,
with the plan's Rulings 4, 7 and 12): a wave submitted whole, then read and steered.

    C="uv run python research/roadless/cluster.py"
    $C wave submit <w> --preset <p> --family <f> --nodes <n,...> [--items <items,|@file>] \
        [--unpack TAR] [--with FILE ...] [--board PATH] [--dry] <ids,|@file> \
        -- <script> <args with {id} [and {item}]>
    $C wave <w> status                  # its summary, reaps, long jobs and classes unserved, runs
    $C wave <w> pools                   # shadow mode: a first lease per class with pending jobs
    $C wave <w> long                    # a long run for each LONG job without one
    $C wave <w> retry <id,...>|failed   # a request its steward applies
    $C wave <w> cancel                  # a request, and its pool and long runs cancelled

`submit`:
1. The wave's commit is HEAD (`git rev-parse HEAD`; the working tree need not be clean: every run
   of the wave is submitted `--at` it, Ruling 7). `<w>` is a run name's (cluster_submit's NAME),
   neither `reblock` (the campaign) nor `submit` (`wave submit ...` is always a submit), no record
   of the registry is a run of it, no record of it is here (`research/roadless/waves/<w>/`, whose
   final.db a reused name would overwrite), and (without --dry) the cluster has no
   `reblock-waves/<w>`. Nothing of the bank's code (BANK_CODE: common.py and all common.write_bank
   builds through, its imports from this repo and the Hydra config it composes) has an uncommitted
   change: the bank is built here by the working tree's code while every run is `--at` HEAD. The
   command's script is a file at the commit (cluster._script_at); it has `{id}`, and `{item}`
   exactly when --items is given; --unpack's tars end in `.tar` (the steward deletes the wave's
   inputs/*.tar at its end). Each of --nodes has its usable memory known (pool_plan.USABLE_GB) and
   serves the preset: the pool the market would start there (Waves.template's arguments) plans and
   routes, so a CPU ladder is refused off CPU_NODES and a GPU one on a node of unusable cards.
2. The bank (common.write_bank, every id checked: a misspelt one is refused before anything is
   planned), the areas (pool_plan.areas_of), the estimator (the family's rows, the earlier waves'
   records, FIT_MIN_KM2 and FIT_MIN_BLOCKS) and the seed (pool_plan.plan_wave: one job per (block,
   item), a block's jobs on one node), each job's line `python -u <command> --keep-standing` (a
   pool's job keeps the rows that stand), after `python -u research/roadless/cluster_unpack.py
   "$CLUSTER_SUBMIT_INPUTS/<tar>" && ` with --unpack. The seed and bank are written to
   `.cluster_submit/waves/<w>/` here (git-ignored; replaced by the next attempt).
3. Printed: the jobs by node and class, the long ones with their runs, the estimates' sources
   counted, the card- and CPU-hours estimated.
4. Without --dry: each run's submit planned with --dry first (a kind's refusal stops the wave
   before anything ships); `reblock-waves/<w>/inputs/` made on the cluster (mkdir: a name another
   machine took meanwhile is refused), the seed (`jobs.json`), the bank, the tars and the --with
   files rsynced there once (Ruling 7), then, each through cluster_submit `--at` the commit: the
   steward (`<w>-steward`, reading the allocation on --board: PROJECT.cluster.board() unless a
   smoke wave names a hand-made one) and a long run per LONG job (`<w>-long-<job>`: its need's
   memory, rounded up, and LONG_TIME times its estimate at most LONG_MAX_MINUTES, the partition's
   7 days). No pool: the steward's first summary puts each class in the next sync's demand, and
   the market starts its leases from Waves.template (Ruling 4). A failure once the directory is
   made (the rsync, a submit) says what was done and how to recover: with the steward live, `wave
   <w> long`; without, remove `reblock-waves/<w>` there to submit the wave again. With --dry, each
   run's submit is planned with --dry and nothing leaves this machine.

**Ruling (Task 7, as the controller corrected it): a fitted long job whose estimate passes the
partition is refused.** A LONG job whose estimate is the family's fit (no stored or measured run of
it) and whose estimate itself passes the partition's 7 days (4287's cycle_desire proposal, fitted
at about 584 h) refuses the submit, naming each such job, before anything is shipped or
submitted: the fit's worst rate says it may finish in no run, and capped at 7 days it would hold
its memory for a week to be killed. The person leaves its block or item out (another wave), or
checkpoints the method (the spec's Alternative C). Flagged and left out of the wave instead, it
would vanish from the wave's record with only a printed line to say so. Every other long job's run
is capped at 7 days: a fitted one whose estimate fits inside them (LONG_TIME times it need not),
and a stored or measured one, whose work has finished within the partition before.

`status`: the summary (one ssh call) by node and class, the workers reaped, the LONG jobs with no
unfinished long run (`wave <w> long` submits them), the classes with jobs pending and no pool lease
of the wave there (the market starts them as the allocation gives them units; in shadow mode,
`wave <w> pools`), and the wave's runs in the registry with their states.

`pools` (shadow mode only, where the market starts nothing; refused in active mode, where it
does): a first lease for each (node, class) with jobs pending and no pool lease of the wave there,
as Waves.template names and argues it (`<w>-<node>-<class>-s<N>`, `_pool_args`), at the wave's
commit. A lease "there" is a pool run of the wave that runs, or whose continuation is undecided
(`_undone`).

`long`: a long run for each LONG job of the summary (pool_summary.LongJob, sized from it) with no
unfinished long run, `<w>-long-<job>`, or `<w>-long-<job>-r<N>` when the registry has that.

`retry <id,...>|failed` and `cancel`: a request file in `reblock-waves/<w>/requests/`
(`<stamp>-<verb>.json`, read exactly by pool_steward.read_request), written whole: to a
temporary name the steward's glob (`*.json`) never matches, then renamed (cluster_demand's
board_script's way). The steward applies it on its next round, so a request needs a live wave
(its steward live in this registry); once only failed jobs remain, the steward has ended the
wave, and a failed job runs again only in a new one. `retry` names jobs of the wave's seed (its
`inputs/jobs.json`, read over ssh): `failed` names every one, and the steward's retry takes back
only those FAILED. `cancel` asks the steward to cancel every job not ended (it then writes the
final snapshot, deletes the bank and tars, and exits 0: Ruling 12), and cancels every pool run
and every long run of the wave that runs or whose continuation is undecided (CANCELLED_KINDS): a
long run on a cancelled job would otherwise run on, its bank and tars deleted under it. The
steward is not cancelled: it ends the wave. Each cancel is tried (they are independent: one that
fails leaves the others' recorded), and the failures are reported together. Without a live
steward, no request is written, and the runs are cancelled all the same.

Every verb but `status` (and a dry submit) is followed by reblock's demand, as cluster_submit's
verbs of STATES_DEMAND are (cluster.main, from `Ran.states_demand`).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cluster
import cluster_submit as cs
import pool_plan as pp
import pool_queue as pq
import pool_summary as psum
from cluster_submit.arrays import route
from cluster_submit.cli import UNITS_DIR, parser
from cluster_submit.context import Context, Shell, utcnow
from cluster_submit.gpu import SHADOW
from cluster_submit.project import NAME
from cluster_submit.registry import Registry, RunRecord
from cluster_submit.remote import SubprocessRunner
from cluster_submit.submit import throttle_checked
from pool_ladder import LADDERS, Ladder, class_name
from pool_steward import BANK, INPUTS, REQUESTS, SEED, read_request

PROJECT = cluster.PROJECT
WAVES = cluster.WAVES
#: --unpack's tars' suffix: the steward deletes the wave's inputs/*.tar at its end
#: (pool_steward.TARS).
TAR = ".tar"
#: What every pooled job's line passes its script (Ruling 8's `Keep`): a row that stands is its
#: identity done, so a double claim or a job run again after a partial write wastes a run only.
KEEP = "--keep-standing"
#: A long run's walltime is this many times its job's estimate (the spec's 3.8), at most
#: LONG_MAX_MINUTES. Cost only, never a row.
LONG_TIME = 1.5
LONG_MAX_MINUTES = cluster.LONG_MAX_MINUTES
#: The kinds of a wave's runs, and those `cancel` cancels (the steward ends the wave itself).
WAVE_KINDS = (cluster.Pool.name, cluster.Steward.name, cluster.Long.name)
CANCELLED_KINDS = (cluster.Pool.name, cluster.Long.name)
#: Where a submit writes a wave's seed and bank here, below the registry's directory.
WORK = "waves"
#: The code that builds a wave's bank here, the working tree's, which must be as the wave's commit
#: has it (its runs are `--at` that commit): common.write_bank and every module of this repo it
#: imports (regions.py's extract and OSM; gap_exits.py, pickled by reference in the bank, and the
#: digests.py it imports; scripts' fetch_kblock_fixtures.py, which src/reblock's footprints read;
#: src/reblock's Block producers) and the Hydra config its source composes (conf/). A directory is
#: read whole by `git status`. Guard `wave` checks the imports a bank's write makes lie below these.
BANK_CODE = ("research/roadless/common.py", "research/roadless/regions.py",
             "research/roadless/gap_exits.py", "research/roadless/digests.py",
             "scripts/__init__.py", "scripts/fetch_kblock_fixtures.py", "src/reblock", "conf")


# ---- the edge: what the verbs act on, built once where the command line is read ---------------

#: One of cluster_submit's verbs on reblock's project (`cs.main`): its arguments, its exit; a
#: refusal raises SystemExit, as cs.main does.
Verb = Callable[[Sequence[str]], int]


@dataclass(frozen=True, kw_only=True)
class Edge:
    """The world the verbs act on: `ctx` (the registry, and the login node through its runner),
    `verb` (each submit and cancel), the rows root the estimators read (`rows`), the features'
    areas (`features`) and the earlier waves' records (`records`). Built once by `here` for
    cluster.main; a guard builds its own."""

    ctx: Context
    verb: Verb
    rows: Path
    features: Path
    records: Path


def here(argv: Sequence[str]) -> Edge:
    """This checkout's edge: a cluster_submit Context as cs.main makes one (a person's shell, the
    project's registry), cs.main for every verb, and this checkout's rows and records."""
    ctx = Context(project=PROJECT, run=SubprocessRunner(), registry=Registry(cluster.REPO),
                  units_dir=UNITS_DIR, now=utcnow, entry=cluster.HERE / "cluster.py",
                  argv=("wave", *argv), machine=socket.gethostname(), caller=Shell(),
                  sleep=time.sleep, parse=parser(PROJECT).parse_args)
    return Edge(ctx=ctx, verb=lambda a: cs.main(PROJECT, list(a)), rows=cluster.HERE,
                features=cluster.HERE / "consolidated_rows" / "summary_features.csv",
                records=cluster.REPO / cluster.WAVE_RECORDS)


class Ran(NamedTuple):
    """A verb's exit, and whether reblock's demand is stated after it (it submitted, cancelled
    or requested something, or tried to)."""

    code: int
    states_demand: bool


class Call(NamedTuple):
    """One of cluster_submit's verbs on one run of the wave: the run, and the verb's arguments."""

    run: str
    argv: tuple[str, ...]


def _ran(edge: Edge, argv: Sequence[str]) -> int:
    """`argv` through `edge.verb`: its exit, or a refusal's (said), 1 for one with a message."""
    try:
        return edge.verb(argv)
    except SystemExit as e:
        if e.code is None or e.code == 0:
            return 0
        print(f"[wave] {shlex.join(argv)}: {e.code}", file=sys.stderr)
        return e.code if isinstance(e.code, int) else 1


def _each(edge: Edge, calls: Sequence[Call], *, what: str) -> int:
    """Each of `calls` in order; the first that fails stops the rest, said with the runs done
    before it and those not tried: 0 when every one ran, else 1."""
    for k, c in enumerate(calls):
        code = _ran(edge, c.argv)
        if code:
            done = ", ".join(x.run for x in calls[:k]) or "none"
            left = ", ".join(x.run for x in calls[k + 1:]) or "none"
            print(f"[wave] {what}: {c.run} failed (exit {code}); done before it: {done}; not "
                  f"tried: {left}", file=sys.stderr)
            return 1
    return 0


def _every(edge: Edge, calls: Sequence[Call], *, what: str) -> int:
    """Each of `calls`, independent of the others: every one tried, the failures said together:
    0 when every one ran, else 1."""
    failed = [c.run for c in calls if _ran(edge, c.argv)]
    if failed:
        print(f"[wave] {what}: {', '.join(failed)} failed (each said above); the rest ran",
              file=sys.stderr)
        return 1
    return 0


# ---- a wave's runs in the registry ------------------------------------------------------------

def _wave_runs(ctx: Context, wave: str) -> list[RunRecord]:
    """The registry's runs of `wave` (its steward, pools and long runs), oldest first. An
    unreadable record is refused (SystemExit): a run of the wave among them would go unseen."""
    recs, unreadable = ctx.registry.all()
    if unreadable:
        raise SystemExit(f"unreadable records: {', '.join(unreadable)}: a run of wave {wave} "
                         "among them would go unseen")
    return [r for r in recs if r.kind in WAVE_KINDS and ctx.parse(r.argv).wave == wave]


def _undone(rec: RunRecord) -> bool:
    """Whether run `rec` runs, or may yet be continued: unfinished, or finished with its
    continuation undecided (not settled)."""
    return rec.finished_at is None or not rec.settled


def _pools_held(ctx: Context, runs: Iterable[RunRecord]) -> set[tuple[str, str]]:
    """The (node, class) of each pool run of `runs` that is `_undone`: a lease there."""
    out: set[tuple[str, str]] = set()
    for r in runs:
        if r.kind == cluster.Pool.name and _undone(r):
            a = ctx.parse(r.argv)
            out.add((a.node, class_name(LADDERS[a.preset].cards, a.mem)))
    return out


def _long_held(ctx: Context, runs: Iterable[RunRecord]) -> set[int]:
    """The jobs of the unfinished long runs of `runs` (a long run is never continued)."""
    return {ctx.parse(r.argv).job for r in runs
            if r.kind == cluster.Long.name and r.finished_at is None}


def _live_commit(ctx: Context, wave: str) -> str:
    """`wave`'s commit, its live steward's (cluster.live_stewards), or a SystemExit: a wave whose
    steward is not live here has ended here, and nothing of it is started or requested."""
    live = cluster.live_stewards(ctx)
    if wave not in live:
        raise SystemExit(f"wave {wave} is not live in this registry (no live steward of it): its "
                         "steward starts, applies and ends nothing more")
    return live[wave].commit


def _ladder_of(s: psum.Summary) -> Ladder:
    """The ladder `s` names, one of LADDERS, or a SystemExit."""
    if s.ladder not in LADDERS:
        raise SystemExit(f"wave {s.wave}'s summary names the ladder {s.ladder!r}, none of this "
                         f"machine's ({', '.join(sorted(LADDERS))})")
    return LADDERS[s.ladder]


# ---- long runs --------------------------------------------------------------------------------

def _long_minutes(minutes: float) -> int:
    """A long run's walltime for a job estimated at `minutes`: LONG_TIME times them, at most
    LONG_MAX_MINUTES, rounded up to the minute."""
    return math.ceil(min(LONG_TIME * minutes, LONG_MAX_MINUTES))


def _long_name(ctx: Context, wave: str, job: psum.LongJob) -> str:
    """A long run's name: `<w>-long-<job>`, or `<w>-long-<job>-r<N>` (N the first the registry
    lacks) when the registry has it (an earlier run of the job)."""
    base = f"{wave}-long-{job.job}"
    if not ctx.registry.exists(base):
        return base
    n = 1
    while ctx.registry.exists(f"{base}-r{n}"):
        n += 1
    return f"{base}-r{n}"


def _long_call(ctx: Context, *, wave: str, preset: str, commit: str, job: psum.LongJob) -> Call:
    """The submit of a long run of `job` at `commit` (the spec's 3.8): on its node, its need's
    memory, its walltime `_long_minutes`."""
    run = _long_name(ctx, wave, job)
    return Call(run=run, argv=(
        "submit", cluster.Long.name, run, "--at", commit, "--wave", wave, "--preset", preset,
        "--job", str(job.job), "--node", job.node, "--mem", str(job.mem_gb),
        "--minutes", str(_long_minutes(job.minutes))))


def _long_of(j: pq.JobSeed) -> psum.LongJob:
    """A LONG job of a seed as a summary lists it (pool_summary.long_jobs: its need rounded up)."""
    return psum.LongJob(job=j.id, block=j.block, item=j.item, node=j.node,
                        mem_gb=math.ceil(j.gb), minutes=j.minutes)


def _unbounded(jobs: Sequence[pq.JobSeed]) -> list[pq.JobSeed]:
    """The LONG jobs whose estimate is a fit and itself passes LONG_MAX_MINUTES: the Ruling's,
    refused."""
    return [j for j in jobs if j.lane == pq.Lane.LONG and j.source.startswith(pp.FIT_SOURCE)
            and j.minutes > LONG_MAX_MINUTES]


# ---- submit -----------------------------------------------------------------------------------

def _head() -> str:
    """The wave's commit: HEAD of this checkout."""
    return subprocess.run(["git", "-C", str(cluster.REPO), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _wave_name(wave: str) -> str:
    """`wave`, a run name's (NAME) and neither CAMPAIGN nor `submit`; else a SystemExit."""
    if not NAME.fullmatch(wave):
        raise SystemExit(f"wave {wave!r}: a wave's name is a run name's ({NAME.pattern})")
    if wave in (cluster.CAMPAIGN, "submit"):
        raise SystemExit(f"wave {wave!r}: not the campaign's name, nor `submit` (`wave submit "
                         "...` is always a submit)")
    return wave


def _unused(ctx: Context, wave: str) -> None:
    """No run of the registry is one of `wave`'s, else a SystemExit naming them."""
    runs = _wave_runs(ctx, wave)
    if runs:
        names = ", ".join(r.name for r in runs)
        raise SystemExit(f"wave {wave} has runs in this registry ({names}): a wave's name is new")


def _recorded(records: Path, wave: str) -> None:
    """No record of `wave` is here (`records`/<wave>: an earlier wave's summary, ledger and
    final.db, which a wave of its name would overwrite), else a SystemExit."""
    if (records / wave).exists():
        raise SystemExit(f"{records / wave} exists: an earlier wave {wave}'s record, which a wave "
                         "of its name would overwrite; a wave's name is new")


def _uncommitted(repo: Path, paths: Sequence[str]) -> list[str]:
    """Those of `paths` (below `repo`) that `git status` lists: changed, staged or untracked."""
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *paths],
                         check=True, capture_output=True, text=True).stdout
    return [line[3:] for line in out.splitlines()]


def _no_directory(ctx: Context, wave: str) -> None:
    """The cluster has no `<WAVES>/<wave>`, else a SystemExit (asked before the bank is built)."""
    if ctx.remote().test(f"test -e {shlex.quote(f'{WAVES}/{wave}')}"):
        raise SystemExit(f"{WAVES}/{wave} exists on {PROJECT.cluster.host}: a wave's name is new")


#: _claim's exit when the wave's directory exists.
TAKEN = 3


def _claim(ctx: Context, wave: str) -> None:
    """`<WAVES>/<wave>/inputs/` made on the cluster, its wave's directory by one mkdir (atomic:
    another machine's wave of the name, made since `_no_directory`, is refused)."""
    d = shlex.quote(f"{WAVES}/{wave}")
    r = ctx.remote().probe(
        f"mkdir -p {shlex.quote(WAVES)} || exit 2; mkdir {d} 2>/dev/null || {{ test -e {d} && "
        f"exit {TAKEN}; exit 2; }}; mkdir {shlex.quote(f'{WAVES}/{wave}/{INPUTS}')} || exit 2")
    if r.returncode == TAKEN:
        raise SystemExit(f"{WAVES}/{wave} exists on {PROJECT.cluster.host}: another wave of the "
                         "name was made meanwhile; nothing shipped or submitted")
    if r.returncode:
        raise SystemExit(f"{WAVES}/{wave}/{INPUTS} could not be made on "
                         f"{PROJECT.cluster.host} (exit {r.returncode}): {r.stderr.strip()[-300:]}")


def _servable(*, wave: str, ladder: Ladder, nodes: Sequence[str], commit: str,
              work: Path) -> None:
    """Each of `nodes` (unique) has its usable memory known (pool_plan.USABLE_GB) and serves
    `ladder`: the pool the market would start there (Waves.template's arguments, at the smallest
    rung) plans, routes and passes the throttle check as a submit does; else a SystemExit."""
    if not nodes or not all(nodes) or len(set(nodes)) != len(nodes):
        raise SystemExit(f"--nodes: empty or repeated entries in {list(nodes)}")
    unknown = [n for n in nodes if n not in pp.USABLE_GB]
    if unknown:
        raise SystemExit(f"--nodes: no usable memory known for {', '.join(unknown)} "
                         f"(pool_plan.USABLE_GB: {', '.join(pp.USABLE_GB)})")
    for n in nodes:
        args = parser(PROJECT).parse_args([
            "submit", cluster.Pool.name, f"{wave}-{n}-check", "--dry", "--at", commit,
            *cluster._pool_args(wave=wave, ladder=ladder, node=n, rung=ladder.rungs[0])])
        plan = cluster.Pool().plan(args, work, sha=commit)
        arrays, too_big = route(PROJECT.cluster, plan.tasks, node=plan.resources.node)
        if too_big:
            raise SystemExit(f"{n}'s card holds no {ladder.name} worker")
        throttle_checked(PROJECT, arrays, args.gpus)


def _job_line(line: str, *, block: str, item: str, tar: Path | None) -> str:
    """A job's line: the command for (`block`, `item`), keeping what stands (KEEP), after
    unpacking `tar` through cluster_unpack.py (a block's jobs may unpack it at once)."""
    run = f"python -u {line.replace('{id}', block).replace('{item}', item)} {KEEP}"
    if tar is None:
        return run
    return (f'python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/{tar.name}" '
            f"&& {run}")


def _inputs(files: Sequence[Path]) -> tuple[Path, ...]:
    """The wave's inputs, each a file, their names unique (one directory holds them)."""
    names = Counter(f.name for f in files)
    twice = sorted(n for n, k in names.items() if k > 1)
    if twice:
        raise SystemExit(f"two inputs named {', '.join(twice)} (the seed is {SEED}, the bank "
                         f"{BANK}): one directory holds a wave's inputs")
    absent = [str(f) for f in files if not f.is_file()]
    if absent:
        raise SystemExit(f"inputs absent: {', '.join(absent)}")
    return tuple(files)


def _source_kind(source: str) -> str:
    """An estimate's source as the plan counts it: `rows`, `wave`, `smoke`, or `fit:<scope>`."""
    head, _, rest = source.partition(":")
    return f"{head}:{rest.partition(':')[0]}" if source.startswith(pp.FIT_SOURCE) else head


def _print_plan(seed: pq.Seed, *, nodes: Sequence[str], longs: Sequence[tuple[Call, pq.JobSeed]],
                work: Path) -> None:
    """The plan of `seed` (the module's doc, submit 3)."""
    h, jobs = seed.header, seed.jobs
    lad = h.ladder
    rungs = {class_name(lad.cards, r): r for r in lad.rungs}
    print(f"wave {h.wave} at {h.commit[:12]}: {len(jobs)} jobs of {h.family} ({lad.name}) on "
          f"{', '.join(nodes)}; its seed and bank in {work}")
    for node in nodes:
        here_ = [j for j in jobs if j.node == node]
        pooled = [j for j in here_ if j.lane == pq.Lane.POOL]
        classes = Counter(j.cls for j in pooled)
        shown = ", ".join(f"{c} {classes[c]}" for c in sorted(classes, key=lambda c: rungs[c]))
        n_long = len(here_) - len(pooled)
        print(f"  {node}: {shown or 'no pool job'}"
              + (f" ({sum(j.minutes for j in pooled) / 60:.1f} h)" if pooled else "")
              + (f"; {n_long} long" if n_long else ""))
    for c, j in longs:
        print(f"  long: job {j.id} ({j.block}{' ' + j.item if j.item else ''}) on {j.node}, "
              f"{j.gb:.1f} GB and {j.minutes:.0f} min ({j.source}): run {c.run}, "
              f"{math.ceil(j.gb)} GB for {cluster.slurm_time(_long_minutes(j.minutes))}")
    sources = Counter(_source_kind(j.source) for j in jobs)
    print("  estimates: " + ", ".join(f"{k} {n}" for k, n in sorted(sources.items())))
    hours = sum(j.minutes for j in jobs) / 60
    print(f"  estimated: {lad.cards * hours:.1f} card-hours, {lad.cpus * hours:.1f} CPU-hours "
          f"({hours:.1f} job-hours, {lad.cards} cards and {lad.cpus} CPUs a job)")
    print("  no pool is submitted: the steward's first summary puts each class in the next "
          "sync's demand, and the market starts its leases (Waves.template; in shadow mode, "
          f"`wave {h.wave} pools`)")


def _recovery(wave: str, *, done: str) -> str:
    """What a person does after a submit failed once `<WAVES>/<wave>` was made."""
    return (f"[wave] {wave}: {WAVES}/{wave} is made on {PROJECT.cluster.host}, {done}: with its "
            f"steward live, `wave {wave} long` submits its long runs; without, remove "
            f"{WAVES}/{wave} there to submit the wave again")


def _submit(edge: Edge, a: argparse.Namespace) -> Ran:
    import common  # geopandas and the block sources: only a submit needs them

    ctx = edge.ctx
    wave = _wave_name(a.wave)
    _unused(ctx, wave)
    _recorded(edge.records, wave)
    dirty = _uncommitted(cluster.REPO, BANK_CODE)
    if dirty:
        raise SystemExit(f"{', '.join(dirty)}: uncommitted changes, and the bank is built here by "
                         "the working tree's code (BANK_CODE: common.write_bank, what it imports "
                         "and the config it composes) while every run of the wave is --at HEAD: "
                         "commit or set them aside first")
    if not a.dry:
        _no_directory(ctx, wave)
    items = [] if a.items is None else cluster._listed(a.items)
    line = cluster._command(a, "{id}", *(["{item}"] if items else []))
    if not items and "{item}" in line:
        raise SystemExit("the command has {item}, and no --items to fill it")
    commit = _head()
    cluster._script_at(a, commit)
    ids = cluster._listed(a.ids)
    for what, names in (("ids", ids), ("items", items)):
        if len(set(names)) != len(names) or not all(names):
            raise SystemExit(f"--{what}: empty or repeated entries in {names}")
    if a.unpack is not None and Path(a.unpack).suffix != TAR:
        raise SystemExit(f"--unpack {a.unpack}: a tar's name ends in {TAR} (the steward deletes "
                         f"the wave's inputs/*{TAR} at its end)")
    tars = cluster._unpacks(a.unpack, ids)
    ladder = LADDERS[a.preset]
    nodes = a.nodes.split(",")
    work = ctx.registry.root.parent / WORK / wave
    if work.exists():
        shutil.rmtree(work)          # a dry or refused attempt's
    work.mkdir(parents=True)
    _servable(wave=wave, ladder=ladder, nodes=nodes, commit=commit, work=work)
    try:
        bank = common.write_bank(ids, work / BANK)
    except ValueError as e:          # an id the source does not yield (common._known)
        raise SystemExit(f"wave {wave}: {e}") from None
    estimator = pp.Estimator(
        family=pp.FAMILIES[a.family](edge.rows), ladder=ladder,
        areas=pp.areas_of(edge.features, bank), history=pp.history_of(edge.records, a.family),
        fit_min_km2=pp.FIT_MIN_KM2, fit_min_blocks=pp.FIT_MIN_BLOCKS)
    seed = pp.plan_wave(
        wave=wave, commit=commit, family=a.family, ladder=ladder, estimator=estimator,
        nodes=nodes, usable_gb=pp.USABLE_GB, ids=ids, items=items or [""],
        line=lambda b, i: _job_line(line, block=b, item=i,
                                    tar=None if a.unpack is None else tars[b]),
        created=ctx.stamp())
    unbounded = _unbounded(seed.jobs)
    if unbounded:
        raise SystemExit(
            f"wave {wave}: fitted long jobs whose estimate passes the partition's "
            f"{LONG_MAX_MINUTES / 1440:g} days, refused (the Ruling in cluster_wave.py): "
            + "; ".join(f"{j.block} {j.item}, {j.minutes / 60:.0f} h ({j.source})"
                        for j in unbounded)
            + ": leave their blocks or items out of this wave")
    steward = f"{wave}-steward"
    calls = [Call(run=steward, argv=("submit", cluster.Steward.name, steward, "--at", commit,
                                     "--wave", wave, "--board", a.board))]
    longs = [(_long_call(ctx, wave=wave, preset=ladder.name, commit=commit, job=_long_of(j)), j)
             for j in seed.jobs if j.lane == pq.Lane.LONG]
    calls += [c for c, _ in longs]
    bad = [c.run for c in calls if not NAME.fullmatch(c.run) or ctx.registry.exists(c.run)]
    if bad:
        raise SystemExit(f"wave {wave}: runs named {', '.join(bad)}, not run names "
                         f"({NAME.pattern}) or in the registry already")
    (work / SEED).write_text(pq.seed_json(seed))
    inputs = _inputs([work / SEED, work / BANK, *tars.values(), *a.extra])
    _print_plan(seed, nodes=nodes, longs=longs, work=work)
    print("  each run planned with --dry:" if a.dry else
          "  each run planned with --dry first (nothing ships if one is refused):")
    code = _each(edge, [c._replace(argv=(*c.argv, "--dry")) for c in calls],
                 what=f"wave {wave}'s dry submits")
    if a.dry or code:
        print(f"  {WAVES}/{wave} on {PROJECT.cluster.host} not checked, nothing shipped"
              if a.dry else f"  nothing shipped or submitted: {WAVES}/{wave} is not made")
        return Ran(code=code, states_demand=False)
    _claim(ctx, wave)
    remote = ctx.remote()
    try:
        remote.rsync("-a", *(str(f) for f in inputs), remote.at(f"{WAVES}/{wave}/{INPUTS}/"))
    except SystemExit:
        print(_recovery(wave, done="nothing submitted"), file=sys.stderr)
        raise
    print(f"  shipped to {WAVES}/{wave}/{INPUTS}: {', '.join(f.name for f in inputs)}")
    code = _each(edge, calls, what=f"wave {wave}'s submits")
    if code:
        print(_recovery(wave, done="its submits stopped at the one that failed"),
              file=sys.stderr)
    else:
        print(f"wave {wave}: submitted {', '.join(c.run for c in calls)}")
    return Ran(code=code, states_demand=True)


# ---- status, pools, long ----------------------------------------------------------------------

def _wanting_pools(s: psum.Summary, held: set[tuple[str, str]]) -> list[psum.ClassLine]:
    """The lines of `s` with jobs pending and no pool lease of the wave there."""
    return [x for x in s.lines if x.pending and (x.node, x.cls) not in held]


def _run_state(r: RunRecord) -> str:
    """A run's state as its record has it."""
    if r.finished_at is None:
        state = "unfinished"
    elif r.continued_by is not None:
        state = f"finished {r.finished_at}, continued by {r.continued_by}"
    elif r.settled:
        state = f"finished {r.finished_at}"
    else:
        state = f"finished {r.finished_at}, its continuation undecided"
    return ("cancelled, " if r.cancelled_at is not None else "") + state


def _status(edge: Edge, wave: str) -> Ran:
    ctx = edge.ctx
    runs = _wave_runs(ctx, wave)
    got = cluster.wave_summaries(ctx.remote(), [wave])
    code = 0
    if wave in got.read:
        s = got.read[wave]
        print(f"wave {wave}: {s.state.value}, at {s.commit[:12]}, ladder {s.ladder}; its summary "
              f"of {s.at} (round {s.round})")
        for x in s.lines:
            print(f"  {x.node} {x.cls}: {x.pending} pending, {x.claimed} claimed, {x.long} long, "
                  f"{x.done} done, {x.failed} failed; {x.workers} workers"
                  + (f"; {x.minutes:.0f} min a job" if x.pending or x.claimed else ""))
        for w in s.reaped:
            print(f"  reaped: {w.task} of {w.run}, {'holding a job' if w.held_job else 'idle'}"
                  + (", at its walltime" if w.walltime else ""))
        held = _long_held(ctx, runs)
        for j in s.long:
            if j.job not in held:
                print(f"  long, no run: job {j.job} ({j.block}{' ' + j.item if j.item else ''}) "
                      f"on {j.node}, {j.mem_gb} GB, {j.minutes:.0f} min: `cluster.py wave "
                      f"{wave} long`")
        for x in _wanting_pools(s, _pools_held(ctx, runs)):
            print(f"  no pool: {x.node} {x.cls}, {x.pending} pending: the market starts one as "
                  f"the allocation gives it units (in shadow mode: `cluster.py wave {wave} "
                  "pools`)")
    else:
        print(f"wave {wave}: its summary does not read ({got.unread[wave]})")
        code = 1
    for r in runs:
        print(f"  run {r.name} ({r.kind}, at {r.commit[:12]}): {_run_state(r)}")
    if not runs:
        print("  no run of it in this registry")
    return Ran(code=code, states_demand=False)


def _pools(edge: Edge, wave: str) -> Ran:
    ctx = edge.ctx
    mode = cs.launches.mode_of(PROJECT)
    if mode.name != SHADOW.name:
        raise SystemExit(f"the GPU market is in {mode.name} mode: it starts wave {wave}'s pools "
                         f"from Waves.template (Ruling 4); `wave {wave} pools` is shadow mode's")
    commit = _live_commit(ctx, wave)
    s = cluster.wave_summary(ctx, wave)
    ladder = _ladder_of(s)
    calls = []
    for x in _wanting_pools(s, _pools_held(ctx, _wave_runs(ctx, wave))):
        run = cluster._start_name(ctx, wave, x.node, x.cls)
        calls.append(Call(run=run, argv=(
            "submit", cluster.Pool.name, run, "--at", commit,
            *cluster._pool_args(wave=wave, ladder=ladder, node=x.node, rung=x.mem_gb))))
    if not calls:
        print(f"wave {wave}: every class with jobs pending has a pool lease (its summary of "
              f"{s.at})")
        return Ran(code=0, states_demand=False)
    return Ran(code=_each(edge, calls, what=f"wave {wave}'s pools"), states_demand=True)


def _long(edge: Edge, wave: str) -> Ran:
    ctx = edge.ctx
    commit = _live_commit(ctx, wave)
    s = cluster.wave_summary(ctx, wave)
    ladder = _ladder_of(s)
    held = _long_held(ctx, _wave_runs(ctx, wave))
    calls = []
    for j in s.long:
        if j.job not in held:
            calls.append(_long_call(ctx, wave=wave, preset=ladder.name, commit=commit, job=j))
    if not calls:
        print(f"wave {wave}: every LONG job has a long run (its summary of {s.at})")
        return Ran(code=0, states_demand=False)
    return Ran(code=_each(edge, calls, what=f"wave {wave}'s long runs"), states_demand=True)


# ---- requests: retry, cancel ------------------------------------------------------------------

#: request_script's exit when the wave has no directory on the cluster.
NO_WAVE = 3


def request_script(wave: str, name: str, text: str) -> str:
    """The remote shell, run in the login directory, that writes request `text` whole as `name`
    in `<WAVES>/<wave>/<REQUESTS>/`: to a temporary file of this shell's own beside it, ending in
    `.tmp` so the steward's glob (`*.json`) never matches it, then a rename (cluster_demand's
    board_script's way). Exit NO_WAVE when the wave has no directory there, 2 when it could not
    write."""
    wave_dir = f"{WAVES}/{wave}"
    d = f"{wave_dir}/{REQUESTS}"
    tmp = shlex.quote(f"{d}/.{name}.") + "$$.tmp"
    payload = base64.b64encode(text.encode()).decode()
    return (f"test -d {shlex.quote(wave_dir)} || exit {NO_WAVE}\n"
            f"mkdir -p {shlex.quote(d)} || exit 2\n"
            f"echo {payload} | base64 -d > {tmp} && mv -f {tmp} {shlex.quote(f'{d}/{name}')} "
            f"|| {{ rm -f {tmp}; exit 2; }}\n")


def _request(ctx: Context, wave: str, verb: str, value: dict[str, object]) -> str:
    """Request `value` written in `wave`'s requests as `<stamp>-<verb>.json` (request_script),
    read back first as its steward reads it (pool_steward.read_request); its name."""
    name = f"{ctx.now().strftime('%Y%m%dT%H%M%S%fZ')}-{verb}.json"
    text = json.dumps(value) + "\n"
    read_request(text, name)
    r = ctx.remote().probe(request_script(wave, name, text))
    if r.returncode == NO_WAVE:
        raise SystemExit(f"{WAVES}/{wave} is not on {PROJECT.cluster.host}: no request written")
    if r.returncode:
        raise SystemExit(f"the request {name} could not be written in {WAVES}/{wave}/{REQUESTS} "
                         f"(exit {r.returncode}): {r.stderr.strip()[-300:]}")
    return name


def _seed_ids(ctx: Context, wave: str) -> list[int]:
    """The job ids of `wave`'s seed on the cluster (`<WAVES>/<w>/inputs/jobs.json`, read exactly),
    or a SystemExit."""
    path = f"{WAVES}/{wave}/{INPUTS}/{SEED}"
    r = ctx.remote().probe(f"cat {shlex.quote(path)}")
    if r.returncode:
        raise SystemExit(f"{path} on {PROJECT.cluster.host} does not read (exit {r.returncode}): "
                         f"{r.stderr.strip()[-300:]}")
    try:
        seed = pq.read_seed(r.stdout, f"{PROJECT.cluster.host}:{path}")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return [j.id for j in seed.jobs]


def _retry(edge: Edge, wave: str, jobs: str) -> Ran:
    ctx = edge.ctx
    _live_commit(ctx, wave)
    seed = _seed_ids(ctx, wave)
    if jobs == "failed":
        ids = seed
    else:
        try:
            ids = [int(x) for x in jobs.split(",")]
        except ValueError:
            raise SystemExit(f"retry {jobs}: job ids, comma-separated, or `failed`") from None
        unknown = sorted(set(ids) - set(seed))
        if unknown or len(set(ids)) != len(ids):
            raise SystemExit(f"retry {jobs}: ids repeated, or no job of wave {wave}'s: {unknown}")
    name = _request(ctx, wave, "retry", {"retry": ids})
    print(f"wave {wave}: requested {name}, a retry of "
          + ("every failed job" if jobs == "failed" else f"jobs {jobs}")
          + ": its steward takes back those FAILED on its next round")
    return Ran(code=0, states_demand=True)


def _cancel(edge: Edge, wave: str) -> Ran:
    ctx = edge.ctx
    if wave in cluster.live_stewards(ctx):
        name = _request(ctx, wave, "cancel", {"cancel": True})
        print(f"wave {wave}: requested {name}: its steward cancels every job not ended and ends "
              "the wave on its next round")
    else:
        print(f"wave {wave}: no live steward of it in this registry, so no request written")
    calls = [Call(run=r.name, argv=("cancel", r.name)) for r in _wave_runs(ctx, wave)
             if r.kind in CANCELLED_KINDS and _undone(r)]
    code = _every(edge, calls, what=f"wave {wave}'s cancels")
    if not code:
        print(f"wave {wave}: cancelled " + (", ".join(c.run for c in calls) or "no run")
              + " (pools and long runs; its steward ends the wave)")
    return Ran(code=code, states_demand=True)


# ---- the command line -------------------------------------------------------------------------

def _submit_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cluster.py wave submit",
                                 description="submit a wave whole (cluster_wave.py's doc)")
    ap.add_argument("wave", help="the wave's name: new, a run name's")
    ap.add_argument("--preset", required=True, choices=sorted(LADDERS),
                    help="the wave's ladder (pool_ladder.LADDERS)")
    ap.add_argument("--family", required=True, choices=sorted(pp.FAMILIES),
                    help="the family of work whose rows the estimates read (pool_plan.FAMILIES)")
    ap.add_argument("--nodes", required=True,
                    help="the nodes its blocks go to, comma-separated (a block's jobs on one)")
    ap.add_argument("--items", help="items, comma-separated, or @file: one job per (block, "
                                    "item), the item in the command's {item}")
    ap.add_argument("--unpack", metavar="TAR",
                    help=f"a tar per block, its path with {{id}} and ending in {TAR}: shipped "
                         "with the wave's inputs and unpacked in the job's checkout before its "
                         "command (cluster_unpack.py)")
    ap.add_argument("--with", dest="extra", type=Path, action="append", default=[],
                    help="a file shipped with the wave's inputs (repeatable); a job reads it as "
                         "$CLUSTER_SUBMIT_INPUTS/<its name>")
    ap.add_argument("--board", default=PROJECT.cluster.board(),
                    help="where the steward reads the market's allocation (relative: below the "
                         "cluster home); it changes no row (default: PROJECT.cluster.board())")
    ap.add_argument("--dry", action="store_true",
                    help="plan the wave and each run's submit; nothing leaves this machine")
    ap.add_argument("ids", help="block ids, comma-separated, or @file of ids")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="-- <script> <args with {id} [and {item}]>")
    return ap


def _steer_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cluster.py wave",
                                 description="read and steer a wave (cluster_wave.py's doc)")
    ap.add_argument("wave")
    verbs = ap.add_subparsers(dest="verb", required=True)
    verbs.add_parser("status", help="its summary, reaps, long jobs and classes unserved, runs"
                     ).set_defaults(run=lambda edge, a: _status(edge, a.wave))
    verbs.add_parser("pools", help="shadow mode: a first lease per class with pending jobs"
                     ).set_defaults(run=lambda edge, a: _pools(edge, a.wave))
    verbs.add_parser("long", help="a long run for each LONG job without one"
                     ).set_defaults(run=lambda edge, a: _long(edge, a.wave))
    r = verbs.add_parser("retry", help="a request: these FAILED jobs back to wait")
    r.add_argument("jobs", help="job ids, comma-separated, or `failed` (every failed job)")
    r.set_defaults(run=lambda edge, a: _retry(edge, a.wave, a.jobs))
    verbs.add_parser("cancel", help="a request: every job not ended cancelled; its pool and "
                                    "long runs cancelled"
                     ).set_defaults(run=lambda edge, a: _cancel(edge, a.wave))
    return ap


def main(argv: Sequence[str], *, edge: Edge) -> Ran:
    """The `wave` verb's `argv` (after `wave`) on `edge` (the module's doc)."""
    if argv[:1] == ["submit"]:
        return _submit(edge, _submit_parser().parse_args(argv[1:]))
    a = _steer_parser().parse_args(argv)
    _wave_name(a.wave)
    run: Callable[[Edge, argparse.Namespace], Ran] = a.run
    return run(edge, a)
