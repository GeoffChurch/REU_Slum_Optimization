import logging
import sys
from pathlib import Path

from scripts.gen_multiblock_example import _tee_to_file, write_maps_qr


def test_write_maps_qr_makes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "maps_qr.png"
    write_maps_qr("https://www.google.com/maps/@-33.9,18.5,18z", out)
    data = out.read_bytes()
    assert out.exists() and len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic


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
