// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type { DemoBundle, DemoScreenBlock } from "../demo.js";
import type { Widget } from "../mount.js";
import type { StateFactory, StateSource } from "../state.js";
import { enumParam, nullableNumberParam, type UrlCodec } from "../url/param.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
import { sizeCanvas } from "../render/canvas.js";
import { fitBbox, toScreen, type Bbox, type View } from "../view/transform.js";

/** The geometry behind the shipped screen, depth_density_proxy = √(nA)/P · n/A, on the block's
 * real outline. No shape is presumed.
 *
 * Let s = √(A/n), the side of one building's share of the ground (a parcel). P/s is how many
 * parcels fit along the street -- the block's frontage slots -- and √(nA)/P = n ÷ (P/s) exactly:
 * the homes behind each frontage slot, for a block of any shape. The score is that divided by s²,
 * which is also n / (P·s): homes per square metre of the one-parcel-deep band along the street.
 *
 * "Rows deep" is only the strip's reading of homes-per-slot: in a strip each slot's column runs
 * straight in to the middle, so homes per slot is rows to the middle; in a square or a disk the
 * columns narrow into wedges, and homes per slot is about half the deepest home's rows.
 *
 * The drawing: the outline, ticked every s along its street, so the slots can be counted; a row
 * of parcels of side s along its longest straight edge, to scale, and the same stretch magnified
 * in an inset. The picker swaps the outline for a disk, a square or a strip of the same area and
 * buildings -- shape enters the score only through P -- or the block for the city's median one;
 * the slider changes the buildings, and with them s, the slot count and the crowding.
 */
const LABEL = "ScreenGeometry";

export const GEO_BLOCKS = ["followed", "typical"] as const;
export type GeoBlock = (typeof GEO_BLOCKS)[number];
/** The block's own outline, or a reference shape of its area. */
export const OUTLINES = ["own", "disk", "square", "strip"] as const;
export type Outline = (typeof OUTLINES)[number];
/** The strip's length over its width. */
export const STRIP_ELONGATION = 16;

/** `homes` is `null` for "the block's own", screen-map.ts's `floor: null` reasoning: a switch of
 * block lands on that block's own count rather than carrying the last block's. */
export interface GeometryState {
  block: GeoBlock;
  outline: Outline;
  homes: number | null;
}

export const SCREEN_GEOMETRY_URL: UrlCodec<GeometryState> = {
  block: enumParam("geo", GEO_BLOCKS),
  outline: enumParam("outline", OUTLINES),
  homes: nullableNumberParam("homes", 6),
};

type Ring = [number, number][];

/** The score's factors for `n` buildings on `area` m² behind `perimeter` m of street. */
export interface Factors {
  side: number;      // s = √(A/n), m
  slots: number;     // P/s
  perSlot: number;   // n ÷ (P/s) = √(nA)/P
  crowding: number;  // n/A, per m²
  score: number;     // perSlot × crowding = perSlot / s²
}

export function factors(area: number, perimeter: number, n: number): Factors {
  const side = Math.sqrt(area / n);
  const slots = perimeter / side;
  const perSlot = n / slots;
  const crowding = n / area;
  return { side, slots, perSlot, crowding, score: perSlot * crowding };
}

/** The street length of `outline` at the block's area: the block's own (as measured,
 * unsimplified) for "own", the reference shapes' exactly. */
export function perimeterOf(outline: Outline, blk: DemoScreenBlock): number {
  const a = blk.area_m2;
  switch (outline) {
    case "own": return blk.perimeter_m;
    case "disk": return 2 * Math.sqrt(Math.PI * a);
    case "square": return 4 * Math.sqrt(a);
    case "strip": {
      const w = Math.sqrt(a / STRIP_ELONGATION);
      return 2 * (w + STRIP_ELONGATION * w);
    }
  }
}

function signedArea(r: Ring): number {
  let s = 0;
  for (let i = 0; i < r.length; i++) {
    const [x0, y0] = r[i]!;
    const [x1, y1] = r[(i + 1) % r.length]!;
    s += x0 * y1 - x1 * y0;
  }
  return s / 2;
}

function bbox(rings: Ring[]): Bbox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const r of rings) {
    for (const [x, y] of r) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return { minX, minY, maxX, maxY };
}

/** The outline drawn: the block's own rings, or a reference shape of its area centred on its
 * bbox's centre (the disk as a 96-gon, which only the drawing uses: its perimeter is
 * `perimeterOf`'s exact one). */
export function ringsOf(outline: Outline, blk: DemoScreenBlock): Ring[] {
  const own = blk.rings.map((r) => r.map(([x, y]) => [x, y] as [number, number]));
  if (outline === "own") return own;
  const b = bbox(own);
  const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2, a = blk.area_m2;
  const rect = (w: number, h: number): Ring => [
    [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
    [cx - w / 2, cy + h / 2]];
  switch (outline) {
    case "disk": {
      const r = Math.sqrt(a / Math.PI);
      return [Array.from({ length: 96 }, (_, i) => {
        const t = (2 * Math.PI * i) / 96;
        return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as [number, number];
      })];
    }
    case "square": return [rect(Math.sqrt(a), Math.sqrt(a))];
    case "strip": {
      const w = Math.sqrt(a / STRIP_ELONGATION);
      return [rect(STRIP_ELONGATION * w, w)];
    }
  }
}

/** A point every `step` metres along every ring, with the unit normal pointing into the block:
 * left of travel on a counter-clockwise exterior, right on a clockwise one, and the other way
 * round on a hole (the block lies outside it). */
export function slotTicks(rings: Ring[], step: number): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  rings.forEach((r, k) => {
    const sign = (k === 0 ? 1 : -1) * Math.sign(signedArea(r));
    let offset = 0;      // how far into the next edge the next tick falls
    for (let i = 0; i < r.length; i++) {
      const [x0, y0] = r[i]!;
      const [x1, y1] = r[(i + 1) % r.length]!;
      const len = Math.hypot(x1 - x0, y1 - y0);
      if (len === 0) continue;
      const ux = (x1 - x0) / len, uy = (y1 - y0) / len;
      let d = offset;
      for (; d < len; d += step) out.push([x0 + ux * d, y0 + uy * d, -uy * sign, ux * sign]);
      offset = d - len;
    }
  });
  return out;
}

/** The exterior ring's longest edge -- where the row of parcels and the inset go, since a
 * straight stretch shows a row of slots most plainly: its start, unit direction, inward normal,
 * length, and how far along it the first slot tick falls (`slotTicks`'s carry from the edges
 * before it), so the parcels drawn on it stand exactly between ticks. */
export function longestEdge(rings: Ring[], step: number): {
  x: number; y: number; ux: number; uy: number; nx: number; ny: number; len: number;
  firstTick: number;
} {
  const r = rings[0]!;
  const sign = Math.sign(signedArea(r));
  let best = { x: 0, y: 0, ux: 1, uy: 0, nx: 0, ny: 1, len: -1, firstTick: 0 };
  let offset = 0;
  for (let i = 0; i < r.length; i++) {
    const [x0, y0] = r[i]!;
    const [x1, y1] = r[(i + 1) % r.length]!;
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len === 0) continue;
    if (len > best.len) {
      const ux = (x1 - x0) / len, uy = (y1 - y0) / len;
      best = { x: x0, y: y0, ux, uy, nx: -uy * sign, ny: ux * sign, len, firstTick: offset };
    }
    offset += Math.ceil((len - offset) / step) * step - len;
  }
  return best;
}

/** The inset's window, in parcel sides across; the parcels drawn along the edge; the buildings
 * slider's span (the block's own count over this, to this times it, log scale); the colours the
 * plate draws on its white. */
const INSET_SLOTS = 9;
const ROW_PARCELS = 7;
const BUILDINGS_SPAN = 8;
const STEPS = 1000;
const BLOCK_FILL = "#ebe6de";
const PARCEL = "#990000";            // --sbu-red-fill: the parcels are what the eye should find
const PARCEL_FILL = "rgba(153, 0, 0, 0.18)";

const toStep = (v: number, lo: number, hi: number): number =>
  Math.round((STEPS * Math.log(v / lo)) / Math.log(hi / lo));
const fromStep = (k: number, lo: number, hi: number): number =>
  lo * Math.exp((k / STEPS) * Math.log(hi / lo));
const fmt = (v: number, sig: number): string =>
  Number(v.toPrecision(sig)).toLocaleString("en-US");

function outlineLabel(o: Outline): string {
  switch (o) {
    case "own": return "Its outline";
    case "disk": return "Disk";
    case "square": return "Square";
    case "strip": return "Strip";
  }
}

export const screenGeometry: Widget<GeometryState> = (host, makeState) => {
  const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
  void fetch(src)
    .then((r) => {
      if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
      return r.json() as Promise<DemoBundle>;
    })
    .then((b) => boot(host, makeState, b))
    .catch((err: unknown) => showWidgetError(host, LABEL, err));
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K,
                                                   cls: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  return e;
}

function group(label: string): HTMLDivElement {
  const g = el("div", "sbu-geometry__group");
  g.setAttribute("role", "group");
  g.setAttribute("aria-label", label);
  return g;
}

function boot(host: HTMLElement, makeState: StateFactory<GeometryState>, b: DemoBundle): void {
  const blocks: Record<GeoBlock, DemoScreenBlock> = {
    followed: b.screen.followed, typical: b.screen.typical,
  };
  for (const g of GEO_BLOCKS) {
    const ring = blocks[g].rings[0];
    if (ring === undefined || ring.length < 3) {
      throw new Error(`${LABEL}: block ${blocks[g].block_id} has no outline`);
    }
  }
  const floor = b.screen.floor;
  const street = b.encoding.street;
  const state: StateSource<GeometryState> = makeState({
    block: "followed", outline: "own", homes: null,
  });

  const caption = host.querySelector("figcaption");
  const wrap = el("div", "sbu-geometry");

  const blockPick = group("Block");
  const blockButtons = GEO_BLOCKS.map((g) => {
    const btn = el("button", "");
    btn.type = "button";
    btn.textContent = g === "followed" ? `Block ${blocks.followed.block_id}` : "A typical block";
    btn.addEventListener("click", () => state.set({ block: g, homes: null }));
    blockPick.append(btn);
    return [g, btn] as const;
  });
  const outlinePick = group("Outline");
  const outlineButtons = OUTLINES.map((o) => {
    const btn = el("button", "");
    btn.type = "button";
    btn.textContent = outlineLabel(o);
    btn.addEventListener("click", () => state.set({ outline: o }));
    outlinePick.append(btn);
    return [o, btn] as const;
  });
  const homesLabel = el("label", "sbu-geometry__slider");
  const homesIn = el("input", "");
  homesIn.type = "range";
  homesIn.min = "0";
  homesIn.max = String(STEPS);
  homesIn.step = "1";
  const homesOut = el("output", "");
  homesLabel.append("Buildings ", homesIn, homesOut);
  homesIn.addEventListener("input", () => {
    const own = blocks[state.get().block].n;
    state.set({ homes: Math.round(fromStep(Number(homesIn.value), own / BUILDINGS_SPAN,
                                              own * BUILDINGS_SPAN)) });
  });
  const controls = el("div", "sbu-geometry__controls");
  controls.append(blockPick, outlinePick, homesLabel);

  const plate = el("div", "sbu-geometry__plate");
  const cv = el("canvas", "");
  cv.style.width = "100%";
  cv.style.aspectRatio = "2 / 1";
  cv.setAttribute("role", "img");
  plate.append(cv);

  const read = el("dl", "sbu-geometry__read");
  read.setAttribute("aria-live", "polite");
  const terms = ["Parcel side, s = √(A/n)", "Frontage slots, P/s", "Homes per slot, n ÷ (P/s)",
                 "Homes per hectare, n/A", "Score, their product"];
  const dds = terms.map((t) => {
    const item = el("div", "");
    const dt = el("dt", "");
    dt.textContent = t;
    const dd = el("dd", "");
    item.append(dt, dd);
    read.append(item);
    return dd;
  });
  const note = el("p", "sbu-geometry__note");
  wrap.append(controls, plate, read, note);
  if (caption) host.insertBefore(wrap, caption); else host.append(wrap);

  let size: { width: number; height: number } | null = null;
  const ctx = cv.getContext("2d")!;

  const trace = (view: View, rings: Ring[]): void => {
    ctx.beginPath();
    for (const r of rings) {
      r.forEach(([x, y], i) => {
        const [sx, sy] = toScreen(view, x, y);
        if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.closePath();
    }
  };
  const box = (x0: number, y0: number, x1: number, y1: number): void => {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y0);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x0, y1);
    ctx.closePath();
  };

  /** The row of parcels standing on the longest edge, squares of side s, centred on its middle. */
  const parcels = (rings: Ring[], f: Factors): Ring[] => {
    const e = longestEdge(rings, f.side);
    // As many as fit on the edge between its ticks (a disk's edges hold only a few), at most
    // ROW_PARCELS, centred.
    const fit = Math.max(0, Math.floor((e.len - e.firstTick) / f.side));
    const count = Math.min(ROW_PARCELS, fit);
    const start = e.firstTick + Math.floor((fit - count) / 2) * f.side;
    return Array.from({ length: count }, (_, k) => {
      const a = start + k * f.side, z = a + f.side;
      return [[e.x + e.ux * a, e.y + e.uy * a], [e.x + e.ux * z, e.y + e.uy * z],
              [e.x + e.ux * z + e.nx * f.side, e.y + e.uy * z + e.ny * f.side],
              [e.x + e.ux * a + e.nx * f.side, e.y + e.uy * a + e.ny * f.side]];
    });
  };

  /** The block, its parcels and its slot ticks through `view` (tick length in CSS pixels). */
  const scene = (view: View, rings: Ring[], f: Factors, tickPx: number): void => {
    trace(view, rings);
    ctx.fillStyle = BLOCK_FILL;
    ctx.fill("evenodd");
    for (const p of parcels(rings, f)) {
      trace(view, [p]);
      ctx.fillStyle = PARCEL_FILL;
      ctx.fill();
      ctx.strokeStyle = PARCEL;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    // A tick every s along the street, pointing in: one per frontage slot.
    ctx.strokeStyle = street;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [x, y, nx, ny] of slotTicks(rings, f.side)) {
      const [sx, sy] = toScreen(view, x, y);
      ctx.moveTo(sx, sy);
      ctx.lineTo(sx + nx * tickPx, sy - ny * tickPx);
    }
    ctx.stroke();
    trace(view, rings);
    ctx.strokeStyle = street;
    ctx.lineWidth = 2;
    ctx.stroke();
  };

  const draw = (rings: Ring[], f: Factors): void => {
    if (size === null) return;
    sizeCanvas(cv, size);
    ctx.clearRect(0, 0, size.width, size.height);
    // The inset's square, top right; the block is fitted into the width left of it.
    const side = Math.round(Math.min(size.height * 0.62, size.width * 0.3));
    const ix = size.width - side - 8, iy = 8;
    const view = fitBbox(bbox(rings), size.width - side - 16, size.height, 0.05);
    scene(view, rings, f, 5);
    // The inset's window: INSET_SLOTS parcel sides across, centred just inside the longest edge,
    // marked on the block so the reader can see where it is.
    const e = longestEdge(rings, f.side);
    const cx = e.x + e.ux * (e.len / 2) + e.nx * f.side * 2;
    const cy = e.y + e.uy * (e.len / 2) + e.ny * f.side * 2;
    const half = (INSET_SLOTS * f.side) / 2;
    const [mx0, my0] = toScreen(view, cx - half, cy + half);
    const [mx1, my1] = toScreen(view, cx + half, cy - half);
    box(mx0, my0, mx1, my1);
    ctx.strokeStyle = PARCEL;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    box(ix, iy, ix + side, iy + side);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.save();
    box(ix, iy, ix + side, iy + side);
    ctx.clip();
    const zoom = fitBbox({ minX: cx - half, minY: cy - half, maxX: cx + half, maxY: cy + half },
                         side, side, 0);
    scene({ ...zoom, tx: zoom.tx + ix, ty: zoom.ty + iy }, rings, f,
          Math.min(side / INSET_SLOTS, 14));
    ctx.restore();
    box(ix, iy, ix + side, iy + side);
    ctx.strokeStyle = PARCEL;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.font = "600 12px sans-serif";
    ctx.fillText(`s = ${fmt(f.side, 2)} m`, ix + 8, iy + side - 10);
  };

  const render = (): void => {
    const s = state.get();
    const blk = blocks[s.block];
    const n = s.homes ?? blk.n;
    const f = factors(blk.area_m2, perimeterOf(s.outline, blk), n);
    homesIn.value = String(toStep(n, blk.n / BUILDINGS_SPAN, blk.n * BUILDINGS_SPAN));
    homesOut.textContent = n.toLocaleString("en-US");
    for (const [g, btn] of blockButtons) btn.setAttribute("aria-pressed", String(g === s.block));
    for (const [o, btn] of outlineButtons) {
      btn.setAttribute("aria-pressed", String(o === s.outline));
    }
    const [dSide, dSlots, dPer, dHa, dScore] = dds;
    dSide!.textContent = `${fmt(f.side, 2)} m`;
    dSlots!.textContent = fmt(f.slots, 3);
    dPer!.textContent = fmt(f.perSlot, 3);
    dHa!.textContent = fmt(f.crowding * 10_000, 3);
    dScore!.textContent = `${f.score.toPrecision(3)} · ${fmt(f.score / floor, 2)}× the floor`;
    const what = s.outline === "own" ? `Block ${blk.block_id}'s own outline`
      : `A ${outlineLabel(s.outline).toLowerCase()} of block ${blk.block_id}'s area`;
    note.textContent = `${what}: ${n.toLocaleString("en-US")} homes share `
      + `${fmt(f.slots, 3)} slots of street, about ${fmt(f.perSlot, 2)} behind each; `
      + (f.score >= floor ? "it clears the shipped floor." : "it falls under the shipped floor.")
      + (s.outline === "own" && s.homes === null ? " These are the block's own numbers." : "");
    cv.setAttribute("aria-label", `${what}, ticked every ${fmt(f.side, 2)} m along its street: `
      + `${fmt(f.slots, 3)} frontage slots.`);
    draw(ringsOf(s.outline, blk), f);
  };

  let first = true;
  observeSize(cv, (measured) => runOrReport(host, LABEL, () => {
    size = measured;
    render();
    if (first) {
      first = false;
      state.subscribe(render);
      removeFallbackImage(host);
    }
  }));
}
