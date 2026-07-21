# Example-regeneration harness — design

**Status: approved (2026-07-21).** Make example generation reproducible and one-command: every
example self-logs, every auto-README cites the exact command that produced it, the maps link carries
a QR code, and a single entry point regenerates them all.

## Motivation

Today there is **no** single command to regenerate the examples. Each of seven example dirs is a
separate manual invocation; `method-comparison`'s `run.log` is a hand-copied Hydra log; and no
generated README says how it was produced. We want: one-command regenerate-all, a `run.log` in every
example dir, a "How this was generated" section citing the command, and a QR code beside each Google
Maps link.

## Current landscape (verified)

- **Six `multiblock_*` dirs** — each one self-emitting command
  `pixi run python -m scripts.gen_multiblock_example <metric> [city]` (metrics `depth`,
  `depth_density`, `density_compactness`; cities `capetown` default + `nairobi`). It writes
  `meta.json` then a machine README via `scripts/gen_example_readme.py` (a pure dir-reader off
  `meta.json`). Out-dir logic: `examples/multiblock_<metric>` (capetown) else
  `examples/<city>/multiblock_<metric>` (`gen_multiblock_example.py:57-58`). No `run.log` today.
- **`method-comparison`** — a *different* generator: `python -m reblock.compare data=capetown_full
  "block_ids=[[ZAF.9.3.1_1_40972]]" methods=[topology,clearance,greedy_arterial_buildable,osm_footpaths]
  max_blocks=1 all_methods.greedy_arterial_buildable.max_roads=8 desire_source.snapshot=…`. It writes
  to the Hydra runtime dir (`outputs/…`, `compare.py:141`), NOT the example dir; its `run.log` is
  Hydra's auto `compare.log`, manually copied in. Its README is **hand-authored** — a two-axis
  metric-basis explainer (external/internal connectivity + displacement) with per-method
  interpretation and design-doc links a pure dir-reader can't produce. **This prose is worth keeping.**
- **No orchestration** anywhere (no Makefile/just/script; pixi tasks are test/lint/run/compare only).

## Parts

1. **Self-logging generator.** In `gen_multiblock_example.main()`, right after `out.mkdir(...)`
   (`:59`), install a stdout+stderr tee to `<out>/run.log` (a small `Tee` wrapping the real streams;
   also route `logging` to that stderr so any `log.info` lands in the file). Restore streams in a
   `finally`. The single command now produces data + README + `run.log`. Captures the main-process
   narrative (region build, `dense_compact` warnings, `run_two_lens` per-method output, the summary).
   Fork-worker GIF output may not be captured — acceptable (the narrative is main-process).

2. **Command provenance.** `gen_multiblock_example` computes the exact reproduce command string
   (`pixi run python -m scripts.gen_multiblock_example <metric>[ <city>]`) and adds `command` (and the
   qr filename) to `meta.json` (`:121-129`). `gen_example_readme` gains a new **gated** section
   `## How this was generated` (rendered only when `meta.get("command")` exists) showing the command in
   a fenced block and linking `run.log` when `(run_dir/"run.log").exists()`. No signature change — it
   stays a pure meta/dir reader, preserving the "numbers can't drift" property.

3. **QR codes.** Add `segno` (pure-Python, no runtime deps) to `[tool.pixi.dependencies]`. In
   `gen_multiblock_example`, after computing `maps_url`, write `<out>/maps_qr.png`
   (`segno.make(maps_url).save(path, scale=…, border=…)`) and record it in `meta.json`.
   `gen_example_readme` embeds `maps_qr.png` beside the existing Google Maps link in the screen
   section (`gen_example_readme.py:40-52`). PNG (not SVG) for GitHub-render parity with the other
   raster artifacts.

4. **Regenerate-all entry point.** New `scripts/regenerate_examples.sh`: with no args, regenerates all
   six `(metric, city)` combos (each self-logs) then `method-comparison` (part 5); with args, a subset.
   Wire a pixi task `regen-examples = "bash scripts/regenerate_examples.sh"` in `pyproject.toml`
   (`:63-70`) → `pixi run regen-examples`.

5. **method-comparison.** Keep the hand-authored README (prose is special). Automate only its
   **data + log**: the regenerate script runs `reblock.compare … hydra.run.dir=examples/method-comparison`
   (self-targeting the dir) and renames the resulting Hydra `compare.log` → `run.log`. It joins
   `regen-examples`; no more manual copy. Add a one-line "How this was generated" note + `run.log` link
   + a drift caveat to its README (its hand-transcribed numbers may lag a fresh run — auto-sync is out
   of scope). Note: Hydra will drop a `.hydra/` dir there; gitignore it (or the script cleans it).

## Testing / verification

- Regenerate one `multiblock_*` example → confirm `run.log`, `maps_qr.png`, and the
  "How this was generated" section appear, and the README still renders.
- `pixi run regen-examples` (or a single-arg subset) → logs land in the right dirs; `git status`
  shows only expected artifacts (new `run.log`/`maps_qr.png` + refreshed data), no spurious churn.
- `pixi run test` stays green (no library code changed except `gen_*` scripts).

## Out of scope

- Auto-syncing `method-comparison`'s hand-transcribed numbers to a fresh run.
- The Tier-B / secondary identity-tuple refactor (separate thread).
