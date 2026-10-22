"""The line a row's writer prints once the row stands, which a pool's worker counts (the lease
pool's plan, docs/superpowers/plans/2026-10-21-lease-pool.md, Ruling 6), and the one spelling of
the flags that say what a run does with a row that stands (clearings.standing_of, Ruling 8). A
module of its own, importing nothing, so a worker reads the marker without clearings' imports
(numpy, pandas, shapely) sitting in its cgroup beside its job, and a wave writes a job's flag
(cluster_wave) without them.
"""

#: What clearings prints, the row's path after it, once a row it writes stands (write_once, under
#: Refuse and Keep, and Replace's write after its os.replace): a pool's worker counts these lines
#: of its job's output into the job's attempt (pool_worker.run_job), so a job that failed having
#: written a row ran further and is no strike. A row kept or refused is not written, so not
#: printed.
ROW_WRITTEN = "[row] wrote "

#: What clearings prints, the row's path after it, for a row a run under Keep found standing and
#: so did not make (at its check, a reuse, or its write: clearings.Keep): its job ran less than
#: its whole work. A pool's worker counts these lines too (pool_worker.run_job), and an earlier
#: wave's measure of a job is only an attempt that kept none (pool_plan.history_of).
KEPT = "[row] kept "

#: A person naming a deliberate replacement of the rows a run would write (clearings.Replace).
REPLACE = "--replace"
#: A pool's job: a row that stands is that identity done (clearings.Keep); every pooled job's line
#: ends with it (cluster_wave.job_line).
KEEP_STANDING = "--keep-standing"
