import json
from pathlib import Path

from scripts.gen_example_readme import gen_example_readme

_FIX = Path(__file__).resolve().parent / "data/example_fixture"

_FULL_META = {"metric": "depth", "flagged": 5, "total_blocks": 10, "deepest_block": "b",
              "deepest_depth": 3, "region_members": 1, "region_parcels": 2,
              "region_mean_depth": 2.0, "region_mean_density_per_ha": 9.0}


def test_generated_readme_reflects_meta_and_lens_csvs() -> None:
    md = gen_example_readme(_FIX, metric_name="depth", formula="depth = √(nA)/P",
                            blurb="Deepest street-access fabric.")
    assert "depth = √(nA)/P" in md                     # formula line
    assert "13,800" in md and "83,192" in md           # screen stat from meta.json (thousands-sep)
    assert "12" in md and "11,006" in md               # region stats
    assert "clearance" in md                            # a lens-CSV row rendered
    assert "![screen](screen.jpg)" in md               # figure embed (present file)


def test_top_scoring_wording_is_metric_neutral() -> None:
    # §1's label must not claim "Deepest ... rings" for a compound metric whose top-scoring block
    # isn't the metro's deepest (e.g. depth_density) -- wording is "Top-scoring ... (peel depth N)".
    md = gen_example_readme(_FIX, metric_name="depth_density", formula="f", blurb="b")
    assert "Top-scoring: `ZAF.9.3.1_1_5810` (peel depth 24)" in md
    assert "Deepest" not in md and "rings." not in md


def test_sections_are_data_gated(tmp_path: Path) -> None:
    # a dir with meta.json but NO lens CSVs omits the two-lens section, without erroring.
    (tmp_path / "meta.json").write_text(json.dumps(_FULL_META))
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "two-lens" not in md.lower() and "Lens A" not in md   # no lens CSVs -> no section
    assert "flagged" in md.lower()                                # screen section still present


def test_partial_meta_omits_affected_section_without_raising(tmp_path: Path) -> None:
    # a present-but-partial meta.json (missing region_members) must not KeyError -- the §2 section
    # is simply omitted while §1 (fully present) still renders.
    partial = {k: v for k, v in _FULL_META.items() if k != "region_members"}
    (tmp_path / "meta.json").write_text(json.dumps(partial))
    md = gen_example_readme(tmp_path, metric_name="depth", formula="f", blurb="b")
    assert "flagged" in md.lower()                     # §1 (complete) still renders
    assert "Grow the region" not in md                 # §2 (incomplete) is omitted
    assert "region_members" not in md
