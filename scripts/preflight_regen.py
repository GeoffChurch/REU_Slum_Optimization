"""What would `regen-examples` change? Answer in minutes, before paying for the run.

A full regeneration is 8-10 hours, almost all of it in `cycle_native` and
`greedy_arterial_access_displacement`. Those are keyed on REGION CONTENT, so they are cache hits
when the region is unchanged and a multi-hour recompute when it is not -- and a six-parcel
difference is enough to trigger the full recompute. Twice now a run has been started on the
assumption that regions would hold, and twice that assumption was wrong, discovered only at the
end.

This recomputes just the cheap half -- screen selection and region growth, no methods, no renders
-- and diffs the resulting membership against each committed `meta.json`'s `region_members`. Every
per-block peel it needs is already in the L2 cache from the last run, so it costs minutes.

Read it as a cost estimate, not a verdict:

    all UNCHANGED  -> the methods will cache-hit; a regeneration is renders and bookkeeping
    any CHANGED    -> that variant re-runs every method from scratch, hours each

    pixi run python -m scripts.preflight_regen
    pixi run python -m scripts.preflight_regen --variants depth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from reblock.pipeline import build_regions
from reblock.presets import load_stages

VARIANTS = ("depth", "depth_density", "density_compactness")
CITIES = ("capetown", "nairobi")


def committed(variant: str, city: str) -> tuple[Path, dict[str, object]] | None:
    """The committed region fingerprint, or None if that variant is not baked.

    Exact membership when `region_member_ids` is present (written from 2026-09-19); otherwise the
    three fields every older meta.json already carries -- seed block, block count, parcel count.
    That triple is not a proof of identity, but it caught both changes seen so far: a 15 -> 13
    block region, and a 37-block region whose parcel count moved by six.
    """
    cfg = OmegaConf.load(f"conf/example/{variant}.yaml")
    slug = str(cfg.example.slug)                                    # type: ignore[union-attr]
    d = Path(f"examples/{slug}") if city == "capetown" else Path(f"examples/{city}/{slug}")
    meta = d / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text())
    return d, {"seed": str(m["deepest_block"]), "blocks": int(m["region_members"]),
               "parcels": int(m["region_parcels"]),
               "ids": [str(b) for b in m["region_member_ids"]] if "region_member_ids" in m
               else None}


def rebuilt(variant: str, city: str) -> dict[str, object]:
    """The same fingerprint under the CURRENT code and config -- screen + growth, no methods."""
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config",
                      overrides=[f"+example={variant}", f"data={city}_full"])
    stages = load_stages(cfg)
    source, screen, region_builder = stages.source, stages.screen, stages.region_builder
    pinned = cfg.block_ids
    groups = None if pinned is None else [list(g) for g in pinned]
    region = build_regions(source, screen, region_builder, groups, int(cfg.max_blocks))[0]
    ids = [str(b.block_id) for b in region]
    ranked = screen.select(source) or []
    return {"seed": str(ranked[0]) if ranked else ids[0], "blocks": len(ids),
            "parcels": sum(len(b.parcels) for b in region), "ids": ids}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--cities", default=",".join(CITIES))
    args = ap.parse_args()

    changed = 0
    for city in args.cities.split(","):
        for variant in args.variants.split(","):
            found = committed(variant, city)
            if found is None:
                print(f"{city:<10}{variant:<22} (not baked -- skipping)")
                continue
            d, before = found
            after = rebuilt(variant, city)
            keys = ("seed", "blocks", "parcels")
            diffs = [k for k in keys if before[k] != after[k]]
            old_ids, new_ids = before["ids"], after["ids"]
            if isinstance(old_ids, list) and isinstance(new_ids, list) and old_ids != new_ids:
                diffs.append("ids")
            if not diffs:
                exact = " (exact membership)" if isinstance(old_ids, list) else " (fingerprint)"
                print(f"{city:<10}{variant:<22} UNCHANGED  {after['blocks']} blocks{exact}")
                continue
            changed += 1
            print(f"{city:<10}{variant:<22} CHANGED    " + ", ".join(
                f"{k}: {before[k]} -> {after[k]}" for k in keys if k in diffs))
            if isinstance(old_ids, list) and isinstance(new_ids, list):
                for b in sorted(set(new_ids) - set(old_ids)):
                    print(f"    + {b}")
                for b in sorted(set(old_ids) - set(new_ids)):
                    print(f"    - {b}")
            print(f"    -> {d} re-runs every method from scratch")

    print(f"\n{changed} variant(s) would change." if changed else "\nNothing would change.")
    print("Cost: an unchanged variant is renders only; a changed one is hours of methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
