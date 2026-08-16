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
