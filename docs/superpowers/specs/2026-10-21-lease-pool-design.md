# The lease pool: reblock's cluster waves as short, shareable leases

Owner, 2026-10-21: "the better long-term shape"; "I'd rather do it right the first time".
A design for the owner's review; nothing of it is built.
Brief: .superpowers/sdd/2026-10-21-lease-pool/brief.md; GeoffChurch/mycooc's lessons beside it (mycooc-lessons.md).
Designed on GeoffChurch/cluster_submit 89b75c4 (README "Leases", "GPU market", "Rules a task must follow", "Placement", "Settling a run"; launches.py, gpu.py, project.py) and on GeoffChurch/mycooc's scheduler at 727e75de (jobs.py, worker.py, liveness.py, gpu_demand.py, scripts/cluster.py).
The market-demand branch (research/market-demand) had no commit when this was written: the spec assumes it lands a writer of today's kinds' demand on the 89b75c4 pin, and says where the pool replaces it (section 5).
Every number is from rows and run records already in this repo (onecard, baselines, lineup lens and proposal rows, `.cluster_submit/runs/*/sacct.txt`, desire_fit.out); none was re-measured.

## 1. The problem

### Today

- research/roadless/cluster.py has two kinds: `blocks`, one GPU task per (block, item), and `cpu`, one CPU task per (block, item) on orion or quadro2.
- Every task of a run gets the run's one `Resources`, so a run is sized for its largest block.
  - cp-roads4-desire-g3b asked 540 GB a task for blocks whose fitted need is 214 to 289 GB (three were resized by hand on 2026-10-21).
  - `blocks` asks 64 GB of host memory for every GPU task; the presets' peaks are a median 7.4 GB on the large blocks and 18.7 GB on the oversized.
- Tasks run to their end, minutes to days: t7-cut-s a median 4 minutes, cycle_desire 48 to 55 h on 63612, 62441 and 62803, 4287's g3a predicted at up to about 178 h against a 7-day wall.
- `more` answers Done for both: neither kind has a routine exit, so nothing is ever a lease.
- On 2026-10-21 at 18:34Z, g3a (850 GB) and g3b (540 GB) held 1.39 TB of orion's 1.5 TB as "the account's jobs no campaign claims" (`cluster.py gpu`).
  By the market's accounts the account was 40% of all use, mostly reblock's memory (GeoffChurch/cluster_submit's GPU market spec, section 13.3).

### What "shared" means under the market's rules

The market never preempts a job.
A campaign shares a node only through three things, and a wave must do all three:

1. **It states demand** every round: units of a bundle (cards, CPUs, memory) per node, with each unit's expected minutes.
   Without it, its jobs are held outside the market, charged to nobody, and subtracted from everyone's capacity.
2. **Its work runs as leases.**
   A lease's workers run at most the node's tier's lease length (2 h on active nodes, 3 h on quiet ones; every node was active on 2026-10-21) and retire with a routine exit.
   The sync continues a lease at `cs.launches.width`: in active mode, the units the allocation gives the campaign on that node and bundle, less what its other leases there hold.
   Freed capacity starts further leases (`<campaign>-<node>-s<N>`).
3. **Its workers step aside** at a job boundary when the allocation's `yields` ask, so a lighter user waits at most for one job.

Today's waves do none of the three; the market-demand branch adds the first.
The other two need jobs that end within a lease, and workers that choose their next job.

## 2. The workloads, measured

Per job (median / p90 / max); "fits" means within a 2-h lease with room for its margin.

| work | jobs | time per job | peak RSS per job | fits a 2-h lease |
|---|---|---|---|---|
| presets on one card (large / oversized) | 60 / 22 | 8 / 16 / 53 min; 22 / 38 / 97 min | 7.4 / 12 / 21 GB; 19 / 41 / 65 GB (card at most 7.6 GB) | 81 of 82 (63612: 1.61 h) |
| presets on a core (small) | 220 | 2.4 / 6.2 / 19 min | 0.6 / 0.9 / 1.6 GB | all |
| baselines leak, bar, runtime (a job per block and part) | 906 | small under 2.5 min; large 4 / 7-13 / 34 min; oversized 12-23 / 26-50 / 81 min | small 1.4 / 1.4 / 1.9 GB; large 3 / 5 / 10; oversized 9.5 / 41 / 64 | all |
| lenses | 303 | at most 4 min (cp-lenses-big) | not stored (asked 48 GB) | all |
| lineup-score, a network (Task 7's classes s / m / l / x) | 565 | 4 / 7 / 23; 14 / 20 / 23; 30 / 37 / 44; 49 / 50 / 50 min; 4287's arterial network 3.0 h, 2.6 of them building its carved mesh | large 3.6 / 5.9 / 12 GB; oversized 9.8 / 41 / 65 | 564 of 565 |
| uniform scoring on a card (ul-large, ul-oversized) | 46 | large 5 to 40 min; oversized 26 to 89 min | 63612 55.8 GB | all |
| screens on a card (cp-screen-w1) | 15 a wave | 13 / 23 / 77 min | as the presets | all |
| proposals: clearance, looped, cycle_native, resistance_lp, grid | 82 each | 1 / 6 min / 4.2 h (one block, 30848) | at most 3.6 GB | all but 30848's |
| proposals: the arterial | 79 | large 19 min / 2.1 h / 5.6 h; oversized 2.2 / 4.8 / 5.1 h, and three stopped unfinished by the owner's ruling | 0.5 / 0.5 / 1.0 GB | large 51 of 60; oversized 8 of 19 |
| proposals: cycle_desire | 78 | large 48 of 61 under 1.5 h; oversized 4.4 to 55 h, g3 predicted 43 to 178 h | 8.8 GB/km^2 (4.5 to 10.9 a task); up to 165 GB, g3 214 to 453 GB | large 48 of 61; oversized none |

- Features run locally (they read the OSM extract): not a cluster workload.
- What a re-run mostly does (presets, baselines, lenses, scoring) fits a lease: all but two of 2,076 jobs (63612's presets, the lineup-score of 4287's arterial network).
- What does not is road proposal: cycle_desire on every oversized block and a fifth of the large, the arterial on over half of the oversized and a sixth of the large, and 30848's other methods.
- Memory spreads from under 3.4 GB (174 networks) to 65 GB within one kind of work (lineup-score), so one size per run wastes most of what it asks; Task 7's runs by class (12 / 24 / 56 / 88 GB) were the first, hand-made fix.

## 3. The design

### 3.1 The parts

A wave is one campaign of the market, under the wave's name, made of four parts:

- **a queue** of its jobs, on the cluster's shared home;
- **pools** of identical short workers, one pool per (node, memory class), each a chain of leases;
- **a steward**, one small CPU lease per wave, which reaps dead claims, asks workers to step aside, feeds the queue and states the wave's demand every minute;
- **a long lane** for the few jobs no lease can hold: plain runs sized per job, stating demand but not leased.

Three kinds replace `blocks` and `cpu`: `pool` (a lease of workers), `steward` and `long`; a `wave` verb submits them together.
The shape is GeoffChurch/mycooc's: its executor is the steward (without the adaptive decisions), its scheduler-pool the pool.

### 3.2 The queue

- **Where:** `$HOME/reblock-waves/<wave>/queue.db` on ai's NFS home, beside the wave's inputs (the bank, `--unpack`'s tars) and the steward's ledger.
  - Not in a run directory: a wave outlives each of its runs, and a settled run's directory is released by the keeper or `gc`.
  - Not in the checkout: the checkout is shared by every run at its commit and holds results, not state.
  - cluster_submit neither lists nor releases it, and the disk market counts it in the home's unowned part; it is small (SQLite of KBs to MBs), and the steward deletes the bank and tars when the wave ends (open question 3).
- **What:** SQLite in journal_mode DELETE, each operation its own connection and one BEGIN IMMEDIATE transaction, as mycooc's job table (lesson 1).
  Tables: `header` (the wave, its commit, family, preset, metric name, ladder and margins: written once), `jobs`, `attempts` (one per claim: worker, node, class, start, end, exit, peak, OOM evidence, outcome), `workers` (Slurm task, run, node, class, end time, registered, retired) and `aside`.
- **A job** is a row: block, item, group, line, node, class, estimate (minutes and peak GB, with their source), state (pending, claimed, done, failed, long, cancelled) and strikes, unique on (block, item).
  Its group is its block: a group's jobs run on one node ("keep every arm of a block on one node": results can differ by machine).
- **A claim** is one transaction: the pending job of the worker's node and class whose estimate fits the walltime it has left, longest estimate first, becomes the worker's, with a new attempt.
  Longest first puts a lease's big jobs at its start and lets small ones fill its tail.
- **The lock is not what keeps results right.**
  GeoffChurch/runstate's SQLite channel documents that POSIX locks on NFS can admit two winners (wiki, "Postgres coordination: what it actually retires"); mycooc's 4,747 claims on ai's home saw none.
  Here a double claim wastes one run and nothing else: every output is a write-once row (`clearings.write_once`, an exclusive hard link), so a second writer of an identity cannot replace the first (3.3).
- **Done** is recorded by the worker that holds the claim, in one transaction, only while the claim is still its own: a reaped worker's late finish stays an attempt, and the job stays as the reaper left it.
  Done means the command exited 0.
- **A dead worker's claim goes back to pending**, as if never claimed (lesson 6), on evidence only, never on elapsed time (lesson 2):
  1. the worker's own exit (`finally: retire`), for every exit it lives through;
  2. the steward's round, when squeue (`-h -r -u $USER -o %i`) no longer lists a registered worker's task and the answer lists the steward's own (else it abstains: lesson 15, mycooc's liveness.py).
  - The reaper records whether the worker held a job, which `more` reads (3.7).
  - When the worker's recorded Slurm end time has passed, the reaper records a walltime kill and raises the job's time estimate to at least what it ran.
  - No self-retirement on SIGTERM: it needs all three legs of the wiki's "Self-retiring claim on walltime" (the shell's delivery, forwarding, the child's retirement) or it is worse than nothing; the walltime fit and the reaper cover it.
- **Behind one Protocol** (`Queue`), so it can move to Postgres with mycooc's (lesson 1) without its callers changing.

### 3.3 Jobs and their outputs

- A job is what a task line is today: one command for one (block, item), such as `consolidated.py presets {id} gpu default,hard,cheap,cheap005` or `road_rescore.py lineup-score <scratch> {id} {item}`.
  Not smaller: a presets job's one-card timing, the whole lineup with its setup, is a measurement the pass keeps.
- reblock's jobs are not resumable segments, so mycooc's lesson 3 does not carry: a job killed midway starts again.
  What survives a kill is every row it finished, each written once at its identity's path.
- **So a job run again after a partial write keeps what stands and writes the rest.**
  Today a second run refuses (`clearings.check_new` raises Exists).
  A `Standing` Strategy, resolved on each script's command line: `Refuse` (a person's run: today's behaviour) or `Keep` (a pool's job: a row that stands is that identity done, at the check and at the write, where an Exists from a double claim means the same).
  Only the pool's job lines pass `--keep-standing`.
  Every script a wave runs gains it, with a guard: consolidated's baselines, presets, lenses and screen, road_rescore's lineup-score and propose, arterial_study, uniform.py.
  A script that cannot keep what stands is not poolable until it can (open question 4).
- Outputs come home as today: rows under research/roadless in the checkout, which every run at the commit shares, pulled by `IntoTree("research/roadless", ("*.parquet",))`.

### 3.4 Pools and memory classes

- **A class is a bundle** (cards, CPUs, memory GB), what one worker holds, so one class on one node is one demand line and one pool (lesson 10).
  GPU classes vary host memory only: the largest GPU peak stored is 7.6 GB, and every usable card holds 32 GB.
- **The ladder is a wave preset's**: a few rungs, each preset documented with what it optimizes and gives up.
  - `cpu-score` (1 CPU): 4, 12, 24, 56, 88 GB, Task 7's classes plus a 4-GB rung, which 174 of the 565 lineup networks fit.
  - `gpu-presets` (1 card, 4 CPUs): 16, 32, 64, 96 GB.
  - A ladder's top rung fits its node less the tier's reserve: 248 GB on the GPU nodes and quadro2 (376 less 128), 1,381 on orion.
- **A job's class is the smallest rung holding its estimated peak**, from an estimator Strategy per family of work, built once where the wave is configured:
  - its own stored peak, from the last finished run of the same work on the same block (`peak_rss_gb` in the onecard, baselines, lineup lens and proposal rows, or an earlier wave's final queue), times 1.2;
  - with none, an area fit over the family's finished jobs at its worst per-km^2 rate, times 1.5 (desire_fit.out: the worst block 24% above the fit; the memory note "Cluster memory from measured peak");
  - each estimate records its source, and the submit prints the counts by source;
  - a peak is a finished job's maximum RSS, never a reading while it runs (lesson 9).
  - Time estimates come the same way: a stored wall times 1.25, else the family's fit at its worst rate.
- **OOM promotes; it never drops.**
  - The worker runs each job as a child and reads its own Slurm job's cgroup OOM-kill count before and after (`memory.events` on cgroup v2, `memory.oom_control` on v1): an increase means the job was killed for memory.
  - The job then moves up a rung, pending, its attempt kept with its peak; past the top rung it goes to the long lane (3.8) at 1.5 times its last peak.
  - A child killed by SIGKILL with no OOM count is a strike (3.5), not a promotion: no guess (open question 9: ai's cgroup version).
- **A promoted job's class may have no pool on its node.**
  The steward's demand states it, but the market starts a lease only from a lease of that bundle on that node in the registry (`launches.newest`).
  Until cluster_submit can start a campaign's lease from a template the campaign names (open question 2), `wave <w> pools` submits the missing pool, and `status` names it.

### 3.5 The worker

`python -u research/roadless/pool.py work --wave <w> --node <n> --class <c>`: every task of a lease alike (`launches.widened`).
Its loop, after mycooc's `worker.work`:

1. register its Slurm task, run, node, class and end time (`SLURM_JOB_END_TIME`); a queue not made yet (the steward's first round makes it) is an open, empty one, waited for as in 5;
2. if asked to step aside (3.6): retire, exit 75;
3. claim a job of its node and class whose estimate fits the walltime left less the margin (10 minutes in both presets); if none fits while some are pending: retire, exit 75 (lesson 4);
4. run it as `bash -c <line>` in the checkout, its output in the worker's log between job markers (cluster_submit pulls the logs), its peak and wall from `os.wait4`; record it done, failed (a strike) or killed for memory (3.4); go to 2;
5. with nothing pending in its class: if the queue is closed (a fixed wave, or every screen stopped), exit 0, drained; if open, wait a poll, and retire with 75 after the preset's idle limit (lesson 4: a paused campaign's workers held 6 GPUs idle for 5 hours).

- **Strikes.**
  A strike is an attempt that ends failed with no more of the job's rows standing than before it (lesson 7's progress, counted in rows).
  After three the job is failed, for a person (`wave <w> retry`).
  Neither a kept Exists nor a promotion is a strike.
- **A failed job never fails its worker.**
  cluster_submit continues a lease only when none of its tasks failed (`its_kinds_call`), so a worker that exited for one bad job would end its chain; mycooc's worker exits non-routine after abandoning a job, here the job's state carries the failure.
  The worker's only non-routine exits are its own faults: a queue of another commit (3.11), an unreadable queue, or three different jobs failed in a row (a bad node, as quadro2's CUDA was): the lease then ends and a person looks.
- The worker inherits cluster_env.sh: one thread per BLAS pool and unbuffered output (lessons 13, 14); a GPU worker is the one process on its card.

### 3.6 The steward

One CPU lease per wave (1 CPU and 4 GB, to be measured; open question 1), its walltime 2 h, retiring with 75 before it, and continued by its own rule while the wave has a job neither done, failed nor cancelled (a long-lane job included).
It is no campaign lease (`Campaigns.lease` is None for it), so the market neither widens nor starts it; its run is named in the demand, so it is charged.
Every 60 s it runs a round:

1. **reap** (3.2);
2. **ask workers to step aside**: it reads the allocation from the board (the home is shared) through its mode (`cs.gpu.mode_named(a.mode).aside`), and for each node and bundle the wave owes, asks that many live workers there, counting those asked since that allocation, idle ones first, then those whose job ends soonest: mycooc's `Queue.ask_aside`, adopted whole; an allocation older than the market's stale_after asks nobody;
3. **feed** the queue through a Strategy: `Fixed` (the jobs granted at submit, the queue closed from the start) or `Screened` (3.12);
4. **state the wave's demand** with `cs.gpu.write_demand` (3.7);
5. **write summary.json** atomically: per node and class the jobs pending, claimed, done and failed; the workers reaped, each with whether it held a job and was past its walltime; the jobs waiting for the long lane; the queue's state; `more`, `status` and the `wave` verb read it;
6. append the round to its ledger; once every job is done, failed or cancelled, write the queue's final snapshot (SQLite's backup), delete the bank and tars, and exit 0.

Why a steward and not this machine's sync: the last bullet of 3.7, and Alternative E.

### 3.7 Demand, `more` and starts

- **Demand**: one file per wave; `runs` are the chain bases of its steward, pools, starts and long runs, read from the workers' registrations rather than squeue.
  - One line per (node, class) with jobs bound there: `units` the jobs there that are pending or claimed and fit a lease, `minutes` their estimates' mean, `urgency` None (a fixed wave values its jobs alike).
  - The long lane's jobs are not units: their runs are named, so what they hold is attributed and charged.
  - Every line's units fall to 0 as its jobs finish, so a finished wave never claims units.
- **`more` of a pool lease** (`Campaigns.lease` says it is its wave's lease on its node, of its class's bundle):
  - no steward of the wave unfinished in this registry: Done, as mycooc's pool answers when no executor runs;
  - summary.json older than the lease's end (`gone_at`), or a task killed at its walltime and not yet reaped: raise, and the next sync asks again;
  - a task killed at its walltime that held no job: Done (lesson 5);
  - nothing pending in its class on its node: Done;
  - else `cs.launches.width(ctx, rec, today=lambda: cs.Continue(workers=min(width_of(rec), pending)))`.
  - It reads summary.json with one `ctx.remote().probe` (`cat`), as `launches.current` reads the allocation.
- **Starts are cluster_submit's**: where the allocation gives the wave more units of a class on a node than its leases there hold, the sync starts a lease from the newest of that bundle there.
  `Campaigns.commit(ctx, wave)` answers the wave's commit while its steward runs, else Done.
- **Why on the cluster, not in this machine's sync.**
  cluster_submit 89b75c4 calls no project code on each sync, and its watch timer stops once nothing is unfinished.
  A wave whose pool chains had all ended Done (the market gave its classes nothing for a while) would then never start again: no sync would run, and no fresh demand would be there to start from.
  The steward is an unfinished run for as long as the wave has work: it keeps the timer syncing and the demand fresh with no change to cluster_submit, and it sits where squeue answers for reaping and the board is written directly.
  mycooc's executor plays the same part.

### 3.8 The long lane

- A job goes long when its estimate does not fit a fresh lease on an active node (2 h less the margin): at submit, when a walltime kill raises its estimate past that, or when it is killed for memory at the top rung.
- A `long` run has one task per job, its own `--mem` (1.5 times the peak) and `--time` (1.5 times the estimate, at most the partition's 7 days), at the wave's commit, on the job's node; its line is the worker's `--job <id>` mode (claim that job, run it, record it, exit).
- The `wave` verb submits them at the wave's start, and `wave <w> long` those that became long later (summary.json lists them, and `status` shows them): a person or an agent, since nothing in a sync submits a run of another kind.
- Long runs are not leased: not continued, widened or asked to step aside; the demand names them with no units, so the market charges and subtracts what they hold.
- **Checkpointing** is the alternative for a method whose long runs recur (cycle_desire and the arterial on oversized blocks).
  Neither recurs in the planned work (the re-run keeps the networks as proposed), so none is built (Alternative C; open question 8).

### 3.9 Submitting and naming a wave

`cluster.py wave submit <w> --preset <p> --family <f> --nodes <n,...> [--items ...] [--unpack TAR] [--with FILE] <ids> -- <command>`:

1. at the clean HEAD (cluster_submit refuses a dirty tree), plan every job (its line, group, estimate and class), assign each group to one of `--nodes` (balanced by estimated minutes), and route the long ones;
2. write the queue's seed (jobs.json), the bank (`common.write_bank`, at this commit) and the tars to a local work directory;
3. `submit steward <w>-steward` with them as inputs: its first round makes the queue from the seed (idempotent: unique on block and item) and moves the inputs into the wave's directory;
4. `submit pool <w>-<node>-<class> --at <commit>`, one worker wide, for each (node, class) with jobs: the market widens each through starts;
5. `submit long <w>-long-<n>` for each long job.

- Names: `<w>` is the campaign (`[A-Za-z0-9][A-Za-z0-9_-]*`); the chains are `<w>-steward`, `<w>-<node>-m<GB>` (`g<GB>` for a GPU class) with their `-l<N>`, the market's `<w>-<node>-s<N>`, and `<w>-long-<n>`.
- Also `wave <w> status | pools | long | retry | cancel`; cancel closes the queue and cancels the steward and every pool run, and long runs only by name.

### 3.10 Results and records

- Rows: in the checkout, pulled by every lease's sync, as today (3.3).
- The wave's record: the steward's pull mirrors its ledger and summary.json, and at the end the final queue, into research/roadless/waves/<w>/ here.
  The final queue holds each job's attempts with their class, peak, wall and node: the stored data the next wave's estimators read (3.4).
- Logs: each worker's task log, job by job, as cluster_submit pulls them.

### 3.11 A wave's identity across leases

- Its commit: every run of the wave is at the steward's commit; continuations by cluster_submit (`checkout.Pinned`, never HEAD), starts by `Campaigns.commit`, the `wave` verb's later submits by `--at`.
- Its plan: the job lines are frozen in the queue at submit, from the plan made at that commit; a lease's re-plan reads nothing but the worker script at the commit (`tree.blob_at`), so a later edit here changes no job.
- Its metric: in the code at the commit, and named in the queue's header.
- A worker refuses a queue whose header names another commit than its checkout's manifest (`.cluster_submit_manifest.json`), and exits non-routine.
- The header's preset, ladder and margins are recorded once; they change cost, never results.

### 3.12 Screening waves

- `Screened` feeds the queue from GeoffChurch/epistemetry's `advance` (screening.py's presets, consolidated's `screen`).
  Each round it advances each screen from the rows standing in the checkout, grants the blocks `Running.want` names (at most the preset's wave ahead), cancels the pending jobs of a screen that has `Stopped`, and closes the queue once every screen has stopped.
- So a screening wave stops when its screen stops, within a round and at most one wave of jobs over.
- `urgency` could carry the screen's value per cost (the GPU market spec, section 5); it stays None until the market reads urgency.
- Built after `Fixed`, since the re-run needs none (open question 7).

### 3.13 mycooc's lessons

- Adopted: 1 (the SQLite queue, behind a Protocol), 2 (death from the scheduler, never from elapsed time), 4 (walltime fit and idle retirement, exit 75), 5 (continue only a kill mid-job), 6 (a dead claim back to pending), 8 (leases at the wave's commit), 9 (peaks from finished RSS), 10 (a class is a demand line), 11 and 12 (step aside at job boundaries; leases of 2 to 3 h), 13 and 14 (cluster_env.sh does both), 15 (the steward abstains, `more` raises).
- Different for reblock:
  - 3: reblock's jobs are whole (block, item) commands, not segments that resume from a checkpoint; what survives a kill is its finished rows (`Keep`, 3.3).
  - 7: progress is counted in rows standing, and a failed job never ends its worker, since a non-routine exit ends the lease's chain.
  - mycooc's executor decides what runs; a fixed wave decides nothing, and a screened one decides through epistemetry.

## 4. Alternatives considered

- **A. Today's arrays, with demand stated** (the market-demand branch).
  Charged honestly, and nearly built; but nothing is shared: tasks hold what they asked to the end (days), sized for the largest, and the market can neither narrow them nor ask them to step aside.
  It stays only until the migration.
- **B. Arrays per memory class, without leases.**
  Fixes the sizing at little code (Task 7 did it by hand: runs of 12, 24, 56 and 88 GB); still run to the end, so no sharing and no stepping aside, and the long tasks still hold for days.
  The pool's classes keep its gain; on everything else the pool is ahead.
- **C. Checkpointing every method**, so every job resumes and no long lane is needed.
  Every optimizer's state (SIMP, the greedies, the arterial, cycle_desire) would need serializing and a resume path, with a risk to identity at each resume, and for the jobs that already fit a lease (all but a handful) it buys nothing.
  Revisit per method, if its long runs recur.
- **D. A queue of files** (claims by `link`, checked with `stat`, as NFS needs).
  No SQLite on NFS; but every query by node, class and estimate scans a directory, and mycooc's SQLite has the measured record on ai.
- **E. No steward: this machine's sync** reaps (from sacct, exactly), feeds and states demand, and workers step aside on their own reading of the allocation.
  One process fewer; but it needs cluster_submit to call the project on each sync and to keep its timer while a campaign waits, an ssh round per sync, and this machine awake for the demand to stay fresh (open question 1).
- **F. The pool in GeoffChurch/cluster_submit**, shared with mycooc.
  The queue, worker, `more` and demand are the same for both; mycooc's job table is shaped by its adaptive executor (budgets, priorities), reblock's by fixed waves.
  Built here behind Protocols, and moved when both run on it (open question 2).
- **G. A job per row** (a preset, a D level): finer jobs fit leases more easily, but each pays the block's model setup again, and the one-card timing is lost.
- **H. Workers that run smaller classes when theirs is empty**: fewer idle workers, but memory the market allocated to the large class then holds small work where the market cannot see it.

## 5. Migration

- **When:** after cp-roads4-desire-g3a and g3b have finished and synced (their records are of kind `cpu`, and followup.md's steps run on today's kinds), and after the market-demand branch has bumped cluster_submit to 89b75c4.
- **One commit** replaces `blocks` and `cpu` with `pool`, `steward` and `long`, adds the `wave` verb and a `Campaigns` (`Waves`: a pool run is its wave's lease; a wave's commit is its steward's while it runs), and deletes the old kinds and their guard (`cluster_check cpu`), rewritten for the pool.
  No old kind stays as an option: a wave reproduces any array, and nothing would choose one.
- **The market-demand writer** of today's kinds goes with them; the steward states every wave's demand, reusing what of it is general (the board's path, a Demand's construction).
- **Existing records** (about 330 settled runs of `blocks` and `cpu`) move out of the registry to `.cluster_submit/archive/`, each once it is settled, unpinned and without unpulled results.
  A record whose kind the project lacks is refused by `kind_of` wherever a verb needs its kind, and `launches.leases` reads every record on each sync; the registry should hold only runs its kinds can read.
  Archived, they stay JSON for the scripts that read them (desire_fit.py, compute.py); rewriting their kind would misstate what ran (open question 6).
- **Nothing to migrate on the cluster**: no queue exists yet, and the old runs' checkouts and directories are released by the keeper as now.

## 6. Guards

research/roadless/pool_check.py, cluster_check's convention: each passes, and fails under its fault.

1. **The claim.** N processes claim from one 200-job queue at once (on local disk, and once on ai's home in the smoke wave): every job claimed once and done once.
   Fault `split-tx` (the select and the update in two transactions, a sleep between): a job claimed twice.
2. **Write-once under a double claim.** Two workers run one job: one set of rows, and the second worker's Exists kept as done.
   Fault `refuse` (`Standing` = `Refuse` in the pool): a strike.
3. **A dead claim.** A worker SIGKILLed holding a job, with squeue stubbed to omit its task: the job back to pending, reaped as mid-job; a squeue that fails, or answers without the steward's own task: nothing reaped.
   Faults `by-age` (claims older than an hour reaped): a live long job reaped; `blind` (a failed squeue read as empty): every worker reaped.
4. **The walltime fit.** With X s left a worker never claims a job estimated over X less the margin, and retires with 75 while such jobs wait.
   Fault `no-fit`: a job claimed that the walltime cannot hold.
5. **The class choice.** On the stored rows each job's class is the smallest rung holding 1.2 times its own peak, and a block with none stored gets its family's worst-rate fit at 1.5 times.
   Fault `median-rate`: 53556's cycle_desire (24% above the fit) classed below its measured peak.
6. **OOM promotion.** A child that allocates past its class in a Slurm cgroup (live, on ai) is promoted, never dropped, and past the top rung it waits for the long lane.
   Fault `fail-on-oom`: the job ends failed.
7. **Demand from the queue.** Units per (node, class) are the jobs pending or claimed there; a finished wave states 0 everywhere; long jobs are runs, not units.
   Faults `count-done` (a finished wave's units not 0) and `every-node` (a group's jobs counted on each of the wave's nodes).
8. **`more`.** Continue only while its class has jobs on its node and every walltime kill held a job; Done for an idle kill, no steward or an empty class; raise while a kill is unreaped.
   Fault `continue-idle`: a lease continued whose killed worker held nothing.
9. **Identity.** A worker given a queue of another commit exits non-routine.
   Fault: the check removed, and it runs the jobs.

Before the first real wave, a smoke wave on ai: two classes, an OOM forced, a walltime kill forced, and a step-aside asked through a hand-made allocation.

## 7. The re-run as the first user

The re-run under the paved-only boost (.superpowers/sdd/2026-10-20-paved-boost/brief.md), as waves; its numbers are the old metric's stored rows, and the boost moves meshes and times little.

- `features`: local, as before (the OSM extract).
- `pb-base` (`cpu-score`, orion and quadro2): baselines leak, bar and runtime on the store's 302 blocks: 906 jobs, 39 core-h; by class 4 GB 764, 12 GB 107, 24 GB 20, 56 GB 12, 88 GB 3; the longest 81 minutes, so every job leased.
- `pb-presets-gpu` (`gpu-presets` on h100, quadro1, tesla1 and tesla2): the presets on the 82 large and oversized blocks and 5810@major, one card a block: 20.4 GPU-h; by class 16 GB 58, 32 GB 18, 64 GB 5, 96 GB 1; 63612 (1.61 h) to the long lane.
  At most 4 cards while `max_gpus` is 4 (open question 5).
- `pb-presets-cpu` (`cpu-score`): the 220 small blocks on cores: 12 core-h, all at 4 GB.
- `pb-lenses`, after the presets: every preset clearing's lenses on cores, classed from each block's baseline peak (the same mesh and model), with promotion to correct it.
- `pb-lineup` (`cpu-score` on quadro2, as the brief asks): lineup-score on the lineup's networks (565 scored under the old metric) and t7-fast's 3: 77.5 h; by class 4 GB 174, 12 GB 301, 24 GB 61, 56 GB 23, 88 GB 6; 4287's arterial network (3.0 h) to the long lane, every other job under 50 minutes.
- After g3: its five networks through followup.md's migration, then `pb-lineup-g3` (4287 at 88 GB, the other four at 56).

So the re-run's long lane holds two jobs, and everything else shares the cluster at the market's widths.

## 8. Open questions for the owner

1. **A steward, or a hook in cluster_submit?**
   The steward needs no change to cluster_submit but is one more CPU job per wave; the hook (Alternative E) needs cluster_submit to call projects each sync and to keep its timer while a campaign waits.
   Recommended: the steward.
2. **Where the pool lives, and starts from a template.**
   Build it here behind Protocols and move it to GeoffChurch/cluster_submit once mycooc and reblock both use it, or ask the cluster_submit session for it now?
   Either way, cluster_submit could start a campaign's lease from a template the campaign names (`Campaigns`), so a class reached by promotion needs no hand-submitted first lease.
   Recommended: here first; ask for the template now.
3. **The queue's home**, `$HOME/reblock-waves/<w>/`, outside cluster_submit's tree: acceptable, or should the wave's state be a path the steward's run retains (`Pull.retained()`) so the disk market accounts for it?
4. **`Keep` for partial rows** in every pooled script (a small change in each, with a guard), or a job with partial rows failed for a person?
   Recommended: `Keep`; promotions and walltime kills leave partial rows routinely.
5. **The 4-card cap** (`max_gpus = 4`, 2026-10-03): keep it now that the market allocates cards, or let the market decide reblock's GPU widths?
6. **Old records**: archive them out of the registry (recommended), or keep them where verbs that need their kind refuse them?
7. **Screening in the first build**, or `Fixed` only until the re-run is done (recommended)?
8. **Checkpointing cycle_desire or the arterial**: only if their oversized proposals will be re-run; is any planned?
9. **OOM evidence**: the worker reads its cgroup's OOM count; ai's cgroup version is to be probed at build, and if it cannot be read, an OOM cannot be told from another SIGKILL (the job is then a strike, not a promotion).
10. **The re-run's timing**: wait for the pool (its build, guards and a smoke wave), or run it on today's kinds with demand stated (the market-demand branch) and move to the pool after?
11. **Groups fixed to a node at submit**: the first build does not move a block's untouched jobs to a node where the market allocates more (mycooc's `reroute` does, a job at a time); add it if a wave's nodes are allocated unevenly.
12. **The ladders and margins** (rungs, 1.2 and 1.5 on memory, 1.25 on time, a 10-minute walltime margin, 3 strikes): set from the measurements above; any you would set otherwise?
