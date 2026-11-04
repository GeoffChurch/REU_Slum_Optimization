/** The four cheap screening metrics (design §3.1), each block's rank by score, and the prefix
 * precision/recall a floor slider reads off that ranking.
 *
 * Mirrors `reblock.metric`'s `Density`, `DepthProxy`, `Product([DepthProxy(), Density()])` and
 * `Product([Density(), Compactness()])`, and `scripts/gen_screen_map.py`'s own `_score` -- all
 * arithmetic on a block's `n`/`area_m2`/`perimeter_m`, cheap enough to run client-side on every
 * block whenever the reader switches metric or drags the floor slider, so switching metrics costs
 * one client-side sort, not a re-fetch. `web/test/screen-model.test.ts` pins the four formulas and
 * checks the shipped floors' pool size and precision/recall against
 * `examples/screen-bakeoff/screen_comparison.csv`, which computes them by an entirely different
 * route (`tests/test_screen_map_bundle.py`'s own, independent check) -- two paths agreeing is the
 * strongest guard available on this widget.
 */
import type { CityBundle } from "../screen_map.js";

export type MetricName =
  | "density"
  | "depth_density_proxy"
  | "density_compactness"
  | "depth_proxy";

/** Each formula is independent and self-contained -- `depth_density_proxy` is NOT
 * `METRICS.depth_proxy(...) * METRICS.density(...)`, even though that IS its mathematical
 * definition (`reblock.metric`'s `Product([DepthProxy(), Density()])`). A product is invariant
 * under swapping which name is bound to which factor, so composing it from the other two entries
 * would leave `depth_density_proxy` blind to exactly the bug this Record's closed-set typing exists
 * to catch elsewhere: a `density`/`depth_proxy` body swap. Four separate literal expressions keep
 * the four entries independently wrong-able, which is the point of keying by the `MetricName`
 * union in the first place -- a fifth name is a compile error, and each of the four is checked on
 * its own in `web/test/screen-model.test.ts`'s "each metric is its published formula". */
export const METRICS: Record<MetricName, (n: number, areaM2: number, perimeterM: number) => number> = {
  density: (n, areaM2) => n / areaM2,
  depth_proxy: (n, areaM2, perimeterM) => Math.sqrt(n * areaM2) / perimeterM,
  density_compactness: (n, _areaM2, perimeterM) => n / (perimeterM * perimeterM),
  depth_density_proxy: (n, areaM2, perimeterM) =>
    (Math.sqrt(n * areaM2) / perimeterM) * (n / areaM2),
};

/** One score per block, in bundle order. `b.n`/`area_m2`/`perimeter_m` all carry exactly
 * `b.n_blocks` entries (`tests/test_screen_map_bundle.py::test_every_column_has_n_blocks_entries`
 * pins it at the bake), so the `!`s below read a bundle invariant, not an unchecked guess. */
export function scores(b: CityBundle, metric: MetricName): Float64Array {
  const f = METRICS[metric];
  const out = new Float64Array(b.n_blocks);
  for (let i = 0; i < b.n_blocks; i++) out[i] = f(b.n[i]!, b.area_m2[i]!, b.perimeter_m[i]!);
  return out;
}

/** Block indices, best-scoring first. */
export function ranking(b: CityBundle, metric: MetricName): Int32Array {
  const s = scores(b, metric);
  const order = new Int32Array(b.n_blocks);
  for (let i = 0; i < b.n_blocks; i++) order[i] = i;
  return order.sort((i, j) => s[j]! - s[i]!);
}

export interface Selection { count: number; precision: number | null; recall: number | null }

/** The floor slider's readout: how many blocks score at or above `floor` -- matching
 * `reblock.metric.Gate`'s `absolute` kind (`s >= value`) and `scripts/gen_screen_map.py`'s own
 * `_score(...) >= spec.value` -- and, where the bundle carries ground truth, how that pool scores
 * against it.
 *
 * Takes `order`/`s` from the caller rather than recomputing them, so a widget re-rendering on every
 * floor-slider drag re-sorts once per metric switch, not once per pixel of drag. `order` must be
 * `ranking`'s descending permutation for the SAME metric `s` was scored with; binary search over it
 * finds the prefix length in O(log n): because `s[order[i]]` is non-increasing in `i`, the indices
 * scoring `>= floor` are exactly a prefix, and this finds that prefix's length -- the first index
 * whose score drops below `floor`, or `order.length` if none does. */
export function selectAt(b: CityBundle, order: Int32Array, s: Float64Array,
                         floor: number): Selection {
  let lo = 0, hi = order.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (s[order[mid]!]! >= floor) lo = mid + 1;
    else hi = mid;
  }
  const count = lo;

  const informal = b.informal;
  if (informal === undefined) return { count, precision: null, recall: null };

  let hits = 0;
  for (let i = 0; i < count; i++) hits += informal[order[i]!]!;
  let total = 0;
  for (const v of informal) total += v;

  return {
    count,
    precision: count > 0 ? hits / count : 0,
    recall: total > 0 ? hits / total : 0,
  };
}

/** Each block's place on the score view's diverging ramp (render/city.ts `SCORE_RAMP`), 0 to 1,
 * the floor at 1/2: its area-weighted mid-rank by score among the blocks on its own side of the
 * floor, so each side spans half the ramp however few blocks it holds -- a block below the floor
 * at the share of the below side's area scoring lower than it (counting half its own), mapped
 * into [0, 1/2); a selected block the same over the selected side, into [1/2, 1]. Recomputed
 * whenever the floor or the metric moves, since both move every rank.
 *
 * `order` and `s` as `selectAt` takes them: `ranking`'s descending permutation for the metric `s`
 * was scored with. The selected side is `selectAt`'s, `s >= floor`. */
export function rankShades(b: CityBundle, order: Int32Array, s: Float64Array,
                           floor: number): Float64Array {
  let below = 0, above = 0;
  for (let i = 0; i < order.length; i++) {
    const k = order[i]!;
    if (s[k]! >= floor) above += b.area_m2[k]!;
    else below += b.area_m2[k]!;
  }
  const out = new Float64Array(b.n_blocks);
  // The area scoring lower than the block, lowest first. Every block below the floor comes before
  // every selected one in this order, so on the selected side the area below is `run - below`.
  let run = 0;
  for (let i = order.length - 1; i >= 0; i--) {
    const k = order[i]!;
    const a = b.area_m2[k]!;
    const up = s[k]! >= floor;
    out[k] = up ? 0.5 + (0.5 * (run - below + a / 2)) / above : (0.5 * (run + a / 2)) / below;
    run += a;
  }
  return out;
}

/** The floor slider's track length: positions 0 to `FLOOR_STEPS`, in whole steps. */
export const FLOOR_STEPS = 1000;

/** The floor slider's log track for one (city, metric): position 0 is the lowest block's score,
 * `FLOOR_STEPS` the highest, and log10(floor) runs linearly between them, so every decade of
 * score gets the same share of the track. The scores span up to five and a half decades (Cape
 * Town's depth_density_proxy, 4.44e-7 to 0.138): a linear track put that metric's shipped floor
 * 9% along, its whole calibrated neighbourhood in a few pixels of drag; the log track puts it 81%
 * along. The full [min, max], not a percentile window: every block's score stays reachable,
 * and a `?floor=` clamped into [min, max] (screen-map.ts's `syncFloor`) stays on the track.
 *
 * The position is only the control's; state, the URL and every readout carry the floor itself. */
export interface FloorScale {
  readonly min: number;
  readonly max: number;
  /** The track position nearest `floor`, a whole number of steps, for a floor in [min, max]. */
  toSlider(floor: number): number;
  /** The floor at track position `u`, a whole number of steps in [0, FLOOR_STEPS]. */
  fromSlider(u: number): number;
}

/** `s`'s own log track. Throws unless every score is positive and they are not all equal: a log
 * is undefined at or below zero, and a zero-width range has no track to lay out. Neither occurs in
 * a committed bundle (the smallest minimum of the eight city/metric pairs is Cape Town's
 * density_compactness, 7.43e-9), so this is a raise, not a fallback. */
export function floorScale(s: Float64Array): FloorScale {
  let min = Infinity, max = -Infinity;
  for (const v of s) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!(min > 0 && max > min)) {
    throw new Error(`a log floor track needs positive scores that are not all equal; `
                  + `these run from ${min} to ${max}`);
  }
  const lo = Math.log10(min), hi = Math.log10(max);
  return {
    min,
    max,
    toSlider: (floor) => Math.round((FLOOR_STEPS * (Math.log10(floor) - lo)) / (hi - lo)),
    // The ends exactly, not through the log: 10 ** log10(min) can land an ulp above `min`, and the
    // track's left end would then leave the lowest block unselected (`selectAt` keeps s >= floor);
    // likewise an ulp above `max` at the right end would select nothing.
    fromSlider: (u) => u === 0 ? min
      : u === FLOOR_STEPS ? max
      : 10 ** (lo + (u / FLOOR_STEPS) * (hi - lo)),
  };
}
