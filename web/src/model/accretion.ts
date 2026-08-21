/** The region-growth greedy, mirroring `DenseClusterRegionBuilder` in src/reblock/region.py.
 *
 * Run in the browser rather than baked, so the budget slider is live and any block in the shipped
 * neighbourhood can be a seed. That is affordable because the rule needs no geometry: the depth
 * proxy is three multiplications on numbers the bundle carries, and adjacency is precomputed.
 *
 * `web/test/accretion.test.ts` pins every step of this against `hood.json`'s `reference` cases,
 * which are `DenseClusterRegionBuilder`'s OWN output -- not a re-derivation of its rule. If this
 * file and region.py ever disagree, that test is what says so.
 */
import type { HoodBlock } from "../hood.js";

/** `sqrt(n*A)/P` -- region.py's `_depth_proxy`, including its zero-safety.
 *
 * Zero-safe matters more here than it looks: a NaN would win every argmax silently, because every
 * comparison against NaN is false and the reduction would simply keep whatever it started with. */
export function depthProxy(n: number, areaM2: number, perimeterM: number): number {
  if (perimeterM <= 0) return 0;
  return Math.sqrt(Math.max(0, n) * Math.max(0, areaM2)) / perimeterM;
}

export interface Growth {
  /** Indices into `blocks`, seed first, then each block in the order it was added. */
  order: number[];
  buildings: number;
  /** True when growth ran out of frontier below budget -- region.py's `if not frontier: break`.
   * In the widget this is the edge of the LOADED neighbourhood, and it is labelled, not hidden. */
  stoppedAtEdge: boolean;
}

export function growth(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): Growth {
  const cluster = new Set<number>([seedIndex]);
  const order = [seedIndex];
  let buildings = blocks[seedIndex]!.n;

  while (buildings < maxBuildings) {
    const frontier = new Set<number>();
    for (const i of cluster) for (const j of blocks[i]!.adj) if (!cluster.has(j)) frontier.add(j);
    if (frontier.size === 0) return { order, buildings, stoppedAtEdge: true };

    let best = -1;
    for (const j of frontier) if (best < 0 || beats(blocks, j, best)) best = j;
    cluster.add(best);
    order.push(best);
    buildings += blocks[best]!.n;
  }
  return { order, buildings, stoppedAtEdge: false };
}

/** region.py's `min(frontier, key=lambda j: (-score(j), -counts[j], ids[j]))`, as a comparison:
 * higher depth proxy wins, then higher building_count, then LOWER block_id. */
function beats(blocks: HoodBlock[], a: number, b: number): boolean {
  const x = blocks[a]!;
  const y = blocks[b]!;
  const sx = depthProxy(x.n, x.area_m2, x.perimeter_m);
  const sy = depthProxy(y.n, y.area_m2, y.perimeter_m);
  if (sx !== sy) return sx > sy;
  if (x.n !== y.n) return x.n > y.n;
  return x.block_id < y.block_id;
}

export function grow(blocks: HoodBlock[], seedIndex: number, maxBuildings: number): number[] {
  return growth(blocks, seedIndex, maxBuildings).order;
}
