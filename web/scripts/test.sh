#!/usr/bin/env bash
# Compile web/test/*.test.ts with tsc, then run the emitted JS under Node's built-in test
# runner. Invoked as `npm test` (package.json's "test" script), which runs with cwd=web/ --
# both the tsconfig path and the "test" dir below are relative to that.
#
# Why not the simpler `node --test --experimental-strip-types test/` (Node 22 can run
# TypeScript directly via type-stripping, no compile step)? Two behaviors of this Node line
# rule it out:
#   1. A bare directory positional arg to `node --test` throws MODULE_NOT_FOUND trying to
#      require() the directory itself, instead of recursively scanning it for test files.
#      Reproduces even with the flag entirely absent (plain .js-only project) -- it is a
#      defect in this Node line's directory-argument handling, not something the flag causes.
#   2. --experimental-strip-types does not rewrite a `.js` import specifier to a sibling `.ts`
#      file. transform.test.ts imports "../src/view/transform.js" -- the standard TS
#      convention, naming the future compiled output rather than the source file -- and
#      Node's loader requires the literal on-disk name to match. With only transform.ts on
#      disk this fails ERR_MODULE_NOT_FOUND; it is how type-stripping works, not a bug.
# Compiling first with tsc sidesteps both: real .js files land on disk under matching names,
# and below we hand `node --test` an explicit file list rather than a directory.
#
# Why `tsc -p tsconfig.test.json` and not the base tsconfig.json? tsconfig.test.json is the
# config with "types": ["node"] (see its own comment) so that node:assert/node:test resolve --
# the base config deliberately omits Node's ambient globals from browser widget code. It
# extends the base config, so tsc still type-checks and emits src/view/transform.ts too, via
# the same import graph the test file walks.
#
# Why run tsc and node --test as separate statements (not `tsc ... && node --test ...`)? tsc
# can exit non-zero purely from a type-checking concern while still emitting valid JS --
# noEmitOnError isn't set, deliberately -- so `&&` would let a tsc-only failure silently skip
# running the tests altogether. Running both unconditionally, then capturing node --test's own
# exit code as STATUS right after it runs, means this script's exit code is exactly the test
# outcome, never tsc's. (This is also why the script has no blanket `set -e`: that would abort
# on tsc's own non-zero exit, exactly the behavior this paragraph rules out. The two checks
# below are therefore explicit, not a global flag.)
#
# Why `find "$OUTDIR/test" -name '*.test.js'` rather than a flat glob or a bare directory arg?
# A flat glob ("$OUTDIR"/test/*.test.js) does not match a nested test/widgets/foo.test.js, so
# a future subdirectory under web/test/ would silently never run. `find` walks recursively and
# hands `node --test` an explicit file list, sidestepping the bare-directory bug above too.
#
# Why `mapfile -d '' -t files < <(find ... -print0)` instead of `node --test $(find ...)`? Two
# defects in the old unquoted-substitution form, found in the fix-wave review (I5): `node --test`
# with ZERO file arguments exits 0 -- so a broken tsconfig.test.json "include", a renamed
# web/test/, or a `.spec.ts` (instead of `.test.ts`) naming drift would all silently run and
# "pass" zero tests. And an unquoted `$(find ...)` word-splits on whitespace in a filename. NUL-
# delimited `find -print0` into a real bash array survives both: `${#files[@]}` can distinguish
# "zero results" from "one file", which is asserted explicitly below, and no word-splitting
# happens on the array elements when they are expanded quoted (`"${files[@]}"`).
#
# Why build the esbuild bundle here, and check its exit code? test/widgets-bundle.test.ts
# evaluates ../docs/js/widgets.js directly (the artifact that ships, not just the src/ modules)
# -- but `pixi run test`'s web-test task and the `web` task (which is what actually runs esbuild)
# are independent leaves of pixi.toml's dependency graph, neither depending on the other. Without
# building here, `pixi run test` alone would either read a stale bundle from a previous `pixi run
# web` or fail outright on a machine that never ran it. Building it as this script's first step
# makes the test suite self-sufficient: no ordering requirement on `web` having run first, here or
# in CI. `|| exit 1` matters because esbuild does NOT overwrite ../docs/js/widgets.js on a failed
# build -- without checking the exit code, a build failure would leave a STALE bundle in place and
# widgets-bundle.test.ts would happily evaluate yesterday's successful build instead.
npm run build || exit 1

# Why build the reblock wheel here as well? test/pyodide-parity.test.ts boots the pinned Pyodide
# for real and has micropip install ../dist/reblock-0.1.0-py3-none-any.whl into it -- the same
# self-sufficiency argument as the esbuild bundle above, and the same staleness hazard, sharpened:
# dist/ is gitignored, so a fresh checkout has NO wheel (the test says so by name and prints this
# command), and a checkout that already has one would otherwise have the parity test measure
# whatever src/reblock looked like when that wheel was last built rather than what it looks like
# now -- which is the one thing a parity guard must not do. `pixi run` rather than a bare `pip` so
# this works from a plain shell as well as from `pixi run test`; nesting it inside an outer `pixi
# run` was measured to work and to keep cwd. `..` is the project directory (pyproject.toml lives
# one level up from web/). Micropip never tries to resolve the scientific stack from PyPI because
# [project] dependencies is empty: the built wheel's METADATA carries no Requires-Dist at all
# (unzipped and checked), so there is nothing to resolve, and --no-deps cannot change that -- a
# flag on `pip wheel` has no way to alter metadata that pip itself already wrote from
# `pyproject.toml`. What --no-deps actually does IS an optimisation: it stops pip from also
# resolving and writing any dependency wheels into ../dist alongside reblock's own. Harmless
# today (there are none to resolve), but the reason to keep the flag if dependencies ever stops
# being empty.
pixi run pip wheel --no-deps --wheel-dir ../dist .. || exit 1

# Why is the Pyodide parity test on this gate rather than in a job of its own? MEASURED (Node
# v24.12.0, this machine, five fresh warm runs): one full boot -- loadPyodide, the seven
# loadPackage packages, micropip, the wheel install and solve.py's module body -- costs 8.9-9.3 s
# with the distribution's wheels already cached in node_modules/pyodide/; a single cold run, when
# loadPackage fetches ~48 MB from jsDelivr and caches them there, cost 11.2 s. The whole file,
# which boots a second minimal interpreter for its pandas-version assertion, runs in 12.2-12.5 s
# warm (six runs) and cost 14.6 s cold (single run). These are single-machine figures, not a
# calibrated benchmark -- read them as an order of magnitude, not to the tenth of a second. This
# piece's plan set 90 s as the cost at which the test earns its own npm script and its own CI job;
# 15 s is not close to it even at the high end of the spread, so it runs here with everything else
# and is deselected nowhere. Two consequences worth knowing rather than rediscovering:
#
#  1. `pixi run test` reaches this through `npm ci`, which DELETES node_modules -- so CI pays the
#     cold number every run, and needs a network to do it. That network dependency is on
#     jsDelivr's AVAILABILITY, not its integrity: every package entry in the version-locked
#     `pyodide-lock.json` carries its own sha256 (checked: it matches the cached wheel's bytes on
#     disk). Pyodide's BROWSER loader enforces that hash through `fetch`'s native `integrity`
#     option (read directly out of `node_modules/pyodide/pyodide.mjs`) -- so for what actually
#     ships to a reader, a compromised or drifted CDN cannot silently substitute bytes. That
#     protection does NOT reach this test, though: Node's own binary-fetch path in the same loader
#     takes the hash as an argument and never reads it. CONFIRMED by corrupting a cached wheel on
#     disk and loading it anyway -- `loadPackage` accepted it, and the only failure was Python's
#     zipfile hitting the corrupted bytes later, not a hash mismatch. So a compromised CDN could
#     substitute bytes into THIS suite's run without either Pyodide or this test noticing; it is
#     only the shipped browser build the sha256 protects.
#  2. The wheels `loadPackage` fetches come from exactly `https://cdn.jsdelivr.net/pyodide/
#     v0.29.2/full/` -- the same path `PYODIDE_INDEX_URL` names, hardcoded in the loader off its
#     own version constant rather than read from the lockfile (also read directly out of
#     `pyodide.mjs`). So this test and a real browser install byte-identical package wheels; only
#     the interpreter core comes from the npm devDependency rather than the CDN.

OUTDIR=$(mktemp -d)
tsc -p tsconfig.test.json --outDir "$OUTDIR" --noEmit false
mapfile -d '' -t files < <(find "$OUTDIR/test" -name '*.test.js' -print0)
if (( ${#files[@]} == 0 )); then
  echo "no compiled test files found under $OUTDIR/test -- a broken tsconfig.test.json," \
       "a renamed web/test/, or a *.test.ts naming drift would all produce this silently" \
       "otherwise (see this script's own comment)" >&2
  rm -rf "$OUTDIR"
  exit 1
fi
node --test "${files[@]}"
STATUS=$?
rm -rf "$OUTDIR"
exit $STATUS
