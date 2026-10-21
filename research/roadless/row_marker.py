"""The line a row's writer prints once the row stands, which a pool's worker counts (the lease
pool's plan, docs/superpowers/plans/2026-10-21-lease-pool.md, Ruling 6). A module of its own,
importing nothing, so a worker reads the marker without clearings' imports (numpy, pandas,
shapely) sitting in its cgroup beside its job.
"""

#: What clearings.write_once prints, the row's path after it, once a row it writes stands: a
#: pool's worker counts these lines of its job's output into the job's attempt
#: (pool_worker.run_job), so a job that failed having written a row ran further and is no strike.
#: A row kept or refused is not written, so not printed.
ROW_WRITTEN = "[row] wrote "
