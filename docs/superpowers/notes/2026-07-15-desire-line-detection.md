# Deriving informal desire-lines from cheap signals — a negative result

**Date:** 2026-07-15
**Outcome:** ship `osm_footpaths` (OSM-only). Detecting the informal footpath network from a cheaper
signal — satellite imagery or the building-point geometry — was explored thoroughly and dropped;
neither matches OSM's human-mapped network. This note records what was tried and why, so the space
doesn't get re-explored from scratch.

## The goal

`osm_footpaths` (formerly `dream_come_true`) reblocks by laying the settlement's REAL existing
footpaths as roads. Its one dependency is OpenStreetMap coverage. The question: can we recover an
equivalent network *without* OSM — from imagery (works anywhere with a satellite pass) or from the
building points the dataset already has (works everywhere, offline, byte-stable)?

## What was tried

**Imagery (Esri World Imagery, 0.25 m/px @ z19) — 6 classical-CV approaches, all fail in dense fabric:**
colour threshold, footprint-gap, Frangi/Sato vesselness, wide-disk morphological opening,
texture-gap-skeleton, and watershed per-shack instance segmentation. Watershed *does* segment
individual shacks into clean rectangles (the "shacks are rectangles" intuition is correct), but that
doesn't yield a path network. Two hard limits, both visible in the spike imagery:
- **Resolution floor:** a real 2 m path is ~8 px and a roof-to-roof seam is ~3–4 px; with ±2 px
  segmentation error you cannot width-threshold "path" from "seam".
- **No tone separation:** narrow interior alleys are shadowed and tonally identical to packed grey
  roofs. Only the *wide* corridors (the settlement's main thoroughfares, and the bare-earth streets
  of the surrounding formal grid) expose enough contiguous warm bare earth to detect — and clipped to
  the informal blocks those are sparse (e.g. block 40972: 142 m in-block vs OSM's 644 m).

An honest "warm-earth wide-corridor" detector was built and validated (it recovers the main corridors)
but it under-delivers *inside* the dense core, which is where reblocking value is. The full imagery
implementation (`ImageryDesireLines`, fetch + detector + tests) is preserved on branch
`dream-come-true-cv`; the spikes are in `scratchpad/spike_*.py` (mosaic, detect, rect, watershed,
widecorridor).

**Building-point geometry — 4 approaches, all hit the same ceiling.** Scored against OSM on block
40972 (recall = fraction of OSM covered within 4 m; precision = fraction of detected on a real path):

| method | recall | precision |
|---|---|---|
| distance-to-point threshold (D ≥ 4.5 m) | 0.50 | 0.34 |
| Meijering ridge + Otsu (width-free) | 0.55 | 0.30 |
| flood, Gaussian-KDE terrain | 0.27 | 0.28 |
| flood, distance terrain (+ thin-barrier breach) | 0.48 | 0.35 |

**Ceiling ≈ 0.5 recall / 0.35 precision** regardless of parameterization, because all four read the
same underlying signal (distance/density to building points) and it has two structural limits:
building *points* lack footprint extent (fuzzy gaps), and **people don't walk every geometric gap**
(so precision caps near ⅓ — most wide gaps aren't paths). The elegant width-free ridge and the
flood/breach ideas improve connectivity but do not raise recall. Spikes:
`scratchpad/spike_parcelpoints.py`, `spike_ridge.py`, `spike_flood.py`.

## The insight that decided it

Synthetic point-geometry desire-lines **fall between two stools**: they're *not real* (that's OSM —
clean, 100 % real, the reason the method exists) and *not optimized* for reblocking (that's
`clearance`/`arterial`, which maximize access directly and would beat a gap-tracing network at the
actual objective). There's no niche where they win. So there's no reason to ship one.

## Decision

OSM is the clean source. `osm_footpaths` uses it, full stop. If a trained building-footprint /
segmentation model or an external AI-footprint dataset (e.g. Google Open Buildings, which covers most
of Africa incl. Cape Town) becomes available later, the resolution-floor gap→path problem still
persists — but precise footprints would at least raise the geometry ceiling above ~0.5, so that's the
one route worth revisiting.
