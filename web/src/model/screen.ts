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
