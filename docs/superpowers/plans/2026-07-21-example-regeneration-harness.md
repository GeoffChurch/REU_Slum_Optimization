# Example-Regeneration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every reblock example reproducible by one command — self-logging to `run.log`, an auto-README that cites its generating command, a QR code for the maps link, and a single `regen-examples` entry point.

**Architecture:** Add two small helpers (a stdout/stderr→file tee, a QR writer) to `scripts/gen_multiblock_example.py`; carry the generating command + QR filename through `meta.json` so the pure dir-reader `gen_example_readme.py` renders a "How this was generated" section and the QR image without new required args; wrap it all in `scripts/regenerate_examples.sh` (+ a pixi task) that also regenerates `method-comparison` in place via a Hydra run-dir override.

**Tech Stack:** Python 3, pixi, segno (QR, conda-forge), Hydra, pytest.

## Global Constraints

- **Commit per task on the current branch `repulsion-and-example-harness`** (do NOT create new branches). Each task: implement, tests green, then `git commit` that task's files. End every commit message with the two trailers `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01MLHAJnMJzWeR7xN725dFkg`.
- `segno` is added to `[tool.pixi.dependencies]` (conda-forge), then `pixi install`.
- `gen_example_readme.py` MUST stay a **pure `meta.json`/dir reader** — no new *required* kwargs; provenance (command, QR path) flows through `meta.json`, so the "numbers can't drift" property holds.
- README-embedded images are **PNG** (GitHub-render parity with the existing jpg/png/gif artifacts).
- Every task ends with `PYTHONPATH=$(pwd) pixi run pytest tests/ -q` green.
- New tests live in `tests/test_gen_examples.py`.

---

### Task 1: QR writer + segno dependency

**Files:**
- Modify: `pyproject.toml` (add `segno` under `[tool.pixi.dependencies]`, ~L27-42)
- Modify: `scripts/gen_multiblock_example.py` (add `write_maps_qr`)
- Test: `tests/test_gen_examples.py` (new)

**Interfaces:**
- Produces: `write_maps_qr(url: str, path: Path, *, scale: int = 4, border: int = 2) -> None` — writes a PNG QR of `url` to `path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gen_examples.py
from pathlib import Path
from scripts.gen_multiblock_example import write_maps_qr

def test_write_maps_qr_makes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "maps_qr.png"
    write_maps_qr("https://www.google.com/maps/@-33.9,18.5,18z", out)
    data = out.read_bytes()
    assert out.exists() and len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py::test_write_maps_qr_makes_a_png -v`
Expected: FAIL — `ModuleNotFoundError: segno` or `ImportError: cannot import name 'write_maps_qr'`.

- [ ] **Step 3: Add the dependency**

In `pyproject.toml`, under `[tool.pixi.dependencies]`, add:
```toml
segno = "*"
```
Run: `pixi install`

- [ ] **Step 4: Write minimal implementation**

Near the top of `scripts/gen_multiblock_example.py` add `import segno` and:
```python
def write_maps_qr(url: str, path: Path, *, scale: int = 4, border: int = 2) -> None:
    """Write a PNG QR code of `url` (e.g. the Google Maps locator) to `path`."""
    segno.make(url, error="m").save(str(path), scale=scale, border=border)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -q`
Expected: PASS. Then `PYTHONPATH=$(pwd) pixi run pytest tests/ -q` stays green.

---

### Task 2: stdout/stderr → file tee (the self-logging primitive)

**Files:**
- Modify: `scripts/gen_multiblock_example.py` (add `_tee_to_file`)
- Test: `tests/test_gen_examples.py`

**Interfaces:**
- Produces: `_tee_to_file(path: Path)` — a context manager that mirrors everything written to `sys.stdout`/`sys.stderr` (and root `logging`) into `path`, while still writing to the originals; restores on exit.

- [ ] **Step 1: Write the failing test**

```python
import logging, sys
from scripts.gen_multiblock_example import _tee_to_file

def test_tee_to_file_captures_print_and_logging(tmp_path):
    log = tmp_path / "run.log"
    with _tee_to_file(log):
        print("hello-stdout")
        sys.stderr.write("hello-stderr\n")
        logging.getLogger("x").info("hello-logging")
    text = log.read_text()
    assert "hello-stdout" in text and "hello-stderr" in text and "hello-logging" in text
    # streams restored:
    assert sys.stdout is sys.__stdout__ or hasattr(sys.stdout, "write")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py::test_tee_to_file_captures_print_and_logging -v`
Expected: FAIL — `ImportError: cannot import name '_tee_to_file'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/gen_multiblock_example.py` (imports `contextlib`, `logging`, `sys`):
```python
@contextlib.contextmanager
def _tee_to_file(path: Path):
    """Mirror stdout+stderr (and root logging at INFO) into `path` for the duration; restore after."""
    f = open(path, "w", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, *streams): self._streams = streams
        def write(self, s):
            for st in self._streams: st.write(s)
        def flush(self):
            for st in self._streams: st.flush()

    orig_out, orig_err = sys.stdout, sys.stderr
    root = logging.getLogger()
    handler = logging.StreamHandler(f)
    handler.setFormatter(logging.Formatter("[%(name)s][%(levelname)s] %(message)s"))
    prev_level = root.level
    root.addHandler(handler)
    if prev_level == logging.NOTSET or prev_level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        sys.stdout = _Tee(orig_out, f)
        sys.stderr = _Tee(orig_err, f)
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        root.removeHandler(handler)
        root.setLevel(prev_level)
        f.flush()
        f.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -q`
Expected: PASS. Then full suite green.

---

### Task 3: README provenance + QR section (pure dir-reader)

**Files:**
- Modify: `scripts/gen_example_readme.py` (`gen_example_readme`, ~L32-114)
- Test: `tests/test_gen_examples.py`

**Interfaces:**
- Consumes: `meta.json` keys `command` (str) and `maps_qr` (filename str), plus a `run.log` file in `run_dir`. All OPTIONAL — sections are gated on presence.
- Produces: rendered markdown containing a `## How this was generated` section and a QR `![...](maps_qr.png)` embed when those inputs exist.

- [ ] **Step 1: Write the failing test**

```python
import json
from scripts.gen_example_readme import gen_example_readme

def _seed_run_dir(d, **meta):
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "run.log").write_text("some log\n")
    (d / "maps_qr.png").write_bytes(b"\x89PNG\r\n\x1a\n")

def test_readme_includes_command_and_qr(tmp_path):
    _seed_run_dir(tmp_path,
        command="pixi run python -m scripts.gen_multiblock_example depth",
        maps_qr="maps_qr.png", maps_url="https://maps.example/x",
        flagged=3, total_blocks=100, deepest_block="B1", deepest_depth=7.0)
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "## How this was generated" in md
    assert "pixi run python -m scripts.gen_multiblock_example depth" in md
    assert "run.log" in md
    assert "maps_qr.png" in md

def test_readme_omits_provenance_when_absent(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "## How this was generated" not in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -k readme -v`
Expected: FAIL — the section/QR strings are absent.

- [ ] **Step 3: Implement the gated blocks**

In `gen_example_readme` (`meta` already read at ~L34-35). In the screen section, right after the `maps_url` link line (~L52), add:
```python
    if meta.get("maps_qr") and (run_dir / meta["maps_qr"]).exists():
        parts.append(f'\n<a href="{meta.get("maps_url","")}">'
                     f'<img src="{meta["maps_qr"]}" alt="Google Maps QR" width="120"></a>\n')
```
Just before the final `return "\n".join(parts)` (~L114), add:
```python
    cmd = meta.get("command")
    if cmd:
        log_link = "\nThe full run log is in [`run.log`](run.log)." if (run_dir / "run.log").exists() else ""
        parts.append("\n## How this was generated\n\n"
                     "This example is machine-generated — one self-logging command emits the data, "
                     "maps, curves, and this README:\n\n"
                     f"```bash\n{cmd}\n```{log_link}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -q`
Expected: PASS. Full suite green.

---

### Task 4: Wire self-logging + provenance + QR into the generator

**Files:**
- Modify: `scripts/gen_multiblock_example.py` (`main`, ~L52-133; add `example_command`)
- Test: `tests/test_gen_examples.py` (for `example_command`)

**Interfaces:**
- Consumes: `write_maps_qr` (Task 1), `_tee_to_file` (Task 2).
- Produces: `example_command(metric: str, city: str) -> str`; `main()` now writes `run.log`, `maps_qr.png`, and `meta.json` keys `command` + `maps_qr`.

- [ ] **Step 1: Write the failing test for the command helper**

```python
from scripts.gen_multiblock_example import example_command

def test_example_command_capetown_omits_city():
    assert example_command("depth", "capetown") == \
        "pixi run python -m scripts.gen_multiblock_example depth"

def test_example_command_other_city_appends_it():
    assert example_command("depth_density", "nairobi") == \
        "pixi run python -m scripts.gen_multiblock_example depth_density nairobi"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -k example_command -v`
Expected: FAIL — `cannot import name 'example_command'`.

- [ ] **Step 3: Implement `example_command` + wire `main()`**

Add the helper:
```python
def example_command(metric: str, city: str) -> str:
    base = f"pixi run python -m scripts.gen_multiblock_example {metric}"
    return base if city == "capetown" else f"{base} {city}"
```
In `main()`, wrap the body from just after `out.mkdir(parents=True, exist_ok=True)` (`:59`) through the end of the work in `with _tee_to_file(out / "run.log"):`. After `maps_url` is computed (~`:88`), add:
```python
        write_maps_qr(maps_url, out / "maps_qr.png")
```
In the `meta` dict (`:121-128`) add:
```python
        "command": example_command(metric_name, city),
        "maps_qr": "maps_qr.png",
```

- [ ] **Step 4: Run the helper tests + full suite**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Integration smoke (orchestrator-run, ~2 min)**

Run: `PYTHONPATH=$(pwd) pixi run python -m scripts.gen_multiblock_example depth`
Expected: `examples/multiblock_depth/run.log` and `examples/multiblock_depth/maps_qr.png` exist, and `examples/multiblock_depth/README.md` contains a `## How this was generated` section + the QR embed. (Verified during review, not a pytest.)

---

### Task 5: `regenerate_examples.sh` + pixi task + method-comparison

**Files:**
- Create: `scripts/regenerate_examples.sh`
- Modify: `pyproject.toml` (`[tool.pixi.tasks]`, ~L63-70)
- Modify: `examples/method-comparison/README.md` (provenance note)
- Modify: `.gitignore` (ignore `examples/method-comparison/.hydra/`)
- Test: `tests/test_gen_examples.py` (dry-run listing)

**Interfaces:**
- Produces: `scripts/regenerate_examples.sh [--dry-run] [metric city]...` — no target args ⇒ all six `(metric, city)` combos + method-comparison; `--dry-run` prints commands without running.

- [ ] **Step 1: Write the failing test (dry-run enumerates the work)**

```python
import subprocess, os
def test_regenerate_dry_run_lists_all(tmp_path):
    env = {**os.environ}
    r = subprocess.run(["bash", "scripts/regenerate_examples.sh", "--dry-run"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for m in ("depth", "depth_density", "density_compactness"):
        assert f"gen_multiblock_example {m}" in out            # capetown
        assert f"gen_multiblock_example {m} nairobi" in out    # nairobi
    assert "reblock.compare" in out and "method-comparison" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -k regenerate -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Write the script**

`scripts/regenerate_examples.sh`:
```bash
#!/usr/bin/env bash
# Regenerate reblock examples. No args => all; else pass "<metric> <city>" pairs.
# --dry-run prints the commands without running them.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

METRICS=(depth depth_density density_compactness)
CITIES=(capetown nairobi)

gen_multiblock() {  # <metric> <city>
  local metric="$1" city="$2"
  run pixi run python -m scripts.gen_multiblock_example "$metric" $([[ "$city" == capetown ]] || echo "$city")
}

gen_method_comparison() {
  local dir="examples/method-comparison"
  run pixi run python -m reblock.compare data=capetown_full \
    "block_ids=[[ZAF.9.3.1_1_40972]]" \
    "methods=[topology,clearance,greedy_arterial_buildable,osm_footpaths]" max_blocks=1 \
    all_methods.greedy_arterial_buildable.max_roads=8 \
    "desire_source.snapshot=$dir/desire_lines_40972.geojson" \
    "hydra.run.dir=$dir"
  # Hydra writes <job>.log (job name 'compare') into the run dir; rename to run.log.
  run bash -c "[[ -f '$dir/compare.log' ]] && mv -f '$dir/compare.log' '$dir/run.log' || true"
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do gen_multiblock "$1" "$2"; shift 2; done
else
  for m in "${METRICS[@]}"; do for c in "${CITIES[@]}"; do gen_multiblock "$m" "$c"; done; done
  gen_method_comparison
fi
```
Make executable: `chmod +x scripts/regenerate_examples.sh`.

- [ ] **Step 4: Run the dry-run test**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/test_gen_examples.py -k regenerate -v`
Expected: PASS. Also `bash -n scripts/regenerate_examples.sh` (syntax) exits 0.

- [ ] **Step 5: Wire the pixi task + gitignore + method-comparison note**

In `pyproject.toml` `[tool.pixi.tasks]` add:
```toml
regen-examples = "bash scripts/regenerate_examples.sh"
```
In `.gitignore` add:
```
examples/method-comparison/.hydra/
```
In `examples/method-comparison/README.md`, under the `## Reproduce` heading, append a line:
```markdown
> Regenerated in-place by `pixi run regen-examples` (or the command above with `hydra.run.dir=examples/method-comparison`); its Hydra log becomes `run.log`. The tables below are hand-written and can lag a fresh run — see `run.log` for the current figures.
```

- [ ] **Step 6: Full suite green**

Run: `PYTHONPATH=$(pwd) pixi run pytest tests/ -q`
Expected: PASS.

---

## Self-Review

- **Spec coverage:** (1) self-logging → Task 2 + Task 4; (2) command provenance → Task 3 + Task 4; (3) QR → Task 1 + Task 3 + Task 4; (4) regenerate-all + pixi task → Task 5; (5) method-comparison data+log+note → Task 5. All five parts covered.
- **Placeholders:** none — real code in every code step.
- **Type consistency:** `write_maps_qr`, `_tee_to_file`, `example_command`, and the `meta.json` keys `command`/`maps_qr` are named identically across Tasks 1-5.
