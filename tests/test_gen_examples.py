from pathlib import Path

from scripts.gen_multiblock_example import write_maps_qr


def test_write_maps_qr_makes_a_png(tmp_path: Path) -> None:
    out = tmp_path / "maps_qr.png"
    write_maps_qr("https://www.google.com/maps/@-33.9,18.5,18z", out)
    data = out.read_bytes()
    assert out.exists() and len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic
