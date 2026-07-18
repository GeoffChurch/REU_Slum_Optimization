import json
from pathlib import Path

from scripts.gen_example_readme import gen_example_readme

_FIX = Path(__file__).resolve().parent / "data/example_fixture"


def test_generated_readme_reflects_meta_and_lens_csvs() -> None:
    md = gen_example_readme(_FIX, metric_name="depth", formula="depth = √(nA)/P",
                            blurb="Deepest street-access fabric.")
    assert "depth = √(nA)/P" in md                     # formula line
    assert "13,800" in md and "83,192" in md           # screen stat from meta.json (thousands-sep)
    assert "12" in md and "11,006" in md               # region stats
    assert "clearance" in md                            # a lens-CSV row rendered
    assert "![screen](screen.jpg)" in md               # figure embed (present file)


def test_sections_are_data_gated(tmp_path: Path) -> None:
    # a dir with meta.json but NO lens CSVs omits the two-lens section, without erroring.
    (tmp_path / "meta.json").write_text(json.dumps(
        {"metric": "depth", "flagged": 5, "total_blocks": 10, "deepest_block": "b",
         "deepest_depth": 3, "region_members": 1, "region_parcels": 2,
         "region_mean_depth": 2.0, "region_mean_density_per_ha": 9.0}))
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "two-lens" not in md.lower() and "Lens A" not in md   # no lens CSVs -> no section
    assert "flagged" in md.lower()                                # screen section still present
