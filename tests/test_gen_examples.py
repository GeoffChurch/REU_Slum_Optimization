import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from scripts.gen_example_readme import gen_example_readme
from scripts.gen_multiblock_example import _tee_to_file, example_command, write_maps_qr


def test_write_maps_qr_makes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "maps_qr.png"
    write_maps_qr("https://www.google.com/maps/@-33.9,18.5,18z", out)
    data = out.read_bytes()
    assert out.exists() and len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic


def test_tee_to_file_captures_print_and_logging(tmp_path):
    log = tmp_path / "run.log"
    before_out, before_err = sys.stdout, sys.stderr
    with _tee_to_file(log):
        print("hello-stdout")
        sys.stderr.write("hello-stderr\n")
        logging.getLogger("x").info("hello-logging")
    text = log.read_text()
    assert "hello-stdout" in text and "hello-stderr" in text and "hello-logging" in text
    assert sys.stdout is before_out and sys.stderr is before_err   # streams restored, not a _Tee


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


def test_example_command_capetown_omits_city():
    assert example_command("depth", "capetown") == \
        "pixi run python -m scripts.gen_multiblock_example depth"


def test_example_command_other_city_appends_it():
    assert example_command("depth_density", "nairobi") == \
        "pixi run python -m scripts.gen_multiblock_example depth_density nairobi"


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
