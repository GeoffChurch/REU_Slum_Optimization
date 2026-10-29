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
import { fitBbox, toScreen } from "../view/transform.js";

/** The geometry behind the shipped screen, depth_density_proxy = √(nA)/P · n/A, drawn on a
 * rectangle a reader can reshape.
 *
 * A/P is a length -- how thick the block is (a strip of width w gives about w/2, a square of side
 * L gives L/4) -- and √(A/n) is the side of one building's share of the ground, a parcel. So
 * √(nA)/P = (A/P) / √(A/n) counts roughly how many parcels deep the block is: how many rows of
 * homes lie between its middle and a street. n/A is how crowded it is. The score is their
 * product: deep AND crowded.
 *
 * The rectangle opens as the block's own: the one with its area, its perimeter and its buildings
 * -- so its three inputs, and therefore its score, are exactly the block's. Then the reader
 * changes its shape (the long side over the short, at the same area) or its buildings, and every
 * quantity is recomputed here from the formula. The drawing cuts the rectangle into n parcels and
 * shades each by how many rows of parcels separate it from the nearest street, every side being
 * street; those rows are what the score's first factor estimates.
 */
const LABEL = "ScreenGeometry";

/** Which of the bundle's two blocks the rectangle starts from. */
export const GEO_BLOCKS = ["followed", "typical"] as const;
export type GeoBlock = (typeof GEO_BLOCKS)[number];

/** `homes` and `shape` are `null` for "the block's own" -- screen-map.ts's `floor: null`
 * reasoning: a switch of block then lands on that block's own rectangle instead of carrying the
 * last block's numbers into a block they were never measured on, and a reader who only switched
 * block publishes no `?homes=`/`?shape=`. */
export interface GeometryState {
  block: GeoBlock;
  homes: number | null;
  shape: number | null;
}

export const SCREEN_GEOMETRY_URL: UrlCodec<GeometryState> = {
  block: enumParam("geo", GEO_BLOCKS),
  homes: nullableNumberParam("homes", 6),
  shape: nullableNumberParam("shape", 3),
};

/** The rectangle with a block's area and perimeter: sides a >= b with a + b = P/2 and ab = A, so
 * its elongation a/b. A block whose perimeter is shorter than a square's of its area (a roundish
 * block) has no such rectangle; the two blocks the bundle carries have one, and a bundle that
 * broke that would be drawn as a different block, so it throws. */
export function ownShape(blk: DemoScreenBlock): number {
  const h = blk.perimeter_m / 2;
  const disc = h * h - 4 * blk.area_m2;
  if (!(disc >= 0)) {
    throw new Error(`${LABEL}: block ${blk.block_id} has no rectangle of its area and perimeter`);
  }
  return (h + Math.sqrt(disc)) / (h - Math.sqrt(disc));
}

/** Everything the readouts and the drawing need, for a rectangle of area `area` (m²), long side
 * over short side `shape`, with `n` buildings. */
export interface Geometry {
  width: number;        // m, the long side
  height: number;       // m
  thickness: number;    // A/P, m
  parcel: number;       // √(A/n), m
  deep: number;         // √(nA)/P: parcels deep
  crowding: number;     // n/A, per m²
  score: number;        // deep × crowding
  cols: number;         // the parcel grid drawn
  rows: number;
  deepest: number;      // rows of parcels from the middle of the grid to a street
}

export function geometry(area: number, shape: number, n: number): Geometry {
  const height = Math.sqrt(area / shape);
  const width = area / height;
  const perimeter = 2 * (width + height);
  const parcel = Math.sqrt(area / n);
  const thickness = area / perimeter;
  const deep = thickness / parcel;
  const crowding = n / area;
  const cols = Math.max(1, Math.round(width / parcel));
  const rows = Math.max(1, Math.round(height / parcel));
  return {
    width, height, thickness, parcel, deep, crowding, score: deep * crowding, cols, rows,
    deepest: Math.ceil(Math.min(cols, rows) / 2),
  };
}

/** The slider ranges, relative to the block's own values: shapes from a square to a strip
 * SHAPE_MAX times as long as it is wide, buildings from a HOMES_SPAN-th of the block's to
 * HOMES_SPAN times them. Both sliders are logarithmic: the score goes as n^1.5 and as a power of
 * the shape, so equal steps of the thumb are equal ratios. */
const SHAPE_MAX = 32;
const HOMES_SPAN = 8;
const STEPS = 1000;

const toStep = (v: number, lo: number, hi: number): number =>
  Math.round((STEPS * Math.log(v / lo)) / Math.log(hi / lo));
const fromStep = (k: number, lo: number, hi: number): number =>
  lo * Math.exp((k / STEPS) * Math.log(hi / lo));

function ramp(stops: readonly string[], t: number): string {
  const i = Math.min(stops.length - 1, Math.max(0, Math.round(t * (stops.length - 1))));
  return stops[i]!;
}

const fmt = (v: number, sig: number): string => Number(v.toPrecision(sig)).toLocaleString("en-US");

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

function boot(host: HTMLElement, makeState: StateFactory<GeometryState>, b: DemoBundle): void {
  // `data-red-rows`: the rows deep at which the shading reaches the ramp's red. The page's
  // generator draws the no-JS fallback with the same number, so the two pictures agree.
  const redRows = Number(requireAttr(host.dataset.redRows, "data-red-rows", LABEL));
  if (!(Number.isInteger(redRows) && redRows > 1)) {
    throw new Error(`${LABEL}: data-red-rows must be an integer above one, got ${redRows}`);
  }
  const blocks: Record<GeoBlock, DemoScreenBlock> = {
    followed: b.screen.followed, typical: b.screen.typical,
  };
  const shapes: Record<GeoBlock, number> = {
    followed: ownShape(blocks.followed), typical: ownShape(blocks.typical),
  };
  const stops = b.encoding.escape_ramp;
  const floor = b.screen.floor;
  const state: StateSource<GeometryState> = makeState({
    block: "followed", homes: null, shape: null,
  });

  const caption = host.querySelector("figcaption");
  const wrap = el("div", "sbu-geometry");

  const picker = el("div", "sbu-geometry__blocks");
  picker.setAttribute("role", "group");
  picker.setAttribute("aria-label", "Start from");
  const pick = GEO_BLOCKS.map((g) => {
    const btn = el("button", "");
    btn.type = "button";
    btn.textContent = g === "followed" ? `Block ${blocks.followed.block_id}` : "A typical block";
    btn.addEventListener("click", () => state.set({ block: g, homes: null, shape: null }));
    picker.append(btn);
    return [g, btn] as const;
  });

  const slider = (text: string): [HTMLLabelElement, HTMLInputElement, HTMLOutputElement] => {
    const label = el("label", "sbu-geometry__slider");
    const input = el("input", "");
    input.type = "range";
    input.min = "0";
    input.max = String(STEPS);
    input.step = "1";
    const out = el("output", "");
    label.append(`${text} `, input, out);
    return [label, input, out];
  };
  const [shapeLabel, shapeIn, shapeOut] = slider("Shape");
  const [homesLabel, homesIn, homesOut] = slider("Buildings");
  const controls = el("div", "sbu-geometry__controls");
  controls.append(picker, shapeLabel, homesLabel);

  const plate = el("div", "sbu-geometry__plate");
  const cv = el("canvas", "");
  cv.style.width = "100%";
  cv.style.aspectRatio = "5 / 2";
  cv.setAttribute("role", "img");
  plate.append(cv);

  const read = el("dl", "sbu-geometry__read");
  read.setAttribute("aria-live", "polite");
  const terms = ["Thickness, A/P", "Parcel side, √(A/n)", "Parcels deep, √(nA)/P",
                 "Crowding, n/A", "Score, their product"] as const;
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

  const key = el("div", "sbu-geometry__key");
  const bar = el("div", "sbu-geometry__ramp");
  bar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
  const ticks = el("div", "sbu-geometry__ticks");
  for (const t of ["1 row from a street", `${redRows}+ rows`]) {
    const span = el("span", "");
    span.textContent = t;
    ticks.append(span);
  }
  key.append(bar, ticks);

  wrap.append(controls, plate, key, read, note);
  if (caption) host.insertBefore(wrap, caption); else host.append(wrap);

  shapeIn.addEventListener("input", () => {
    state.set({ shape: fromStep(Number(shapeIn.value), 1, SHAPE_MAX) });
  });
  homesIn.addEventListener("input", () => {
    const own = blocks[state.get().block].n;
    state.set({ homes: Math.round(fromStep(Number(homesIn.value), own / HOMES_SPAN,
                                              own * HOMES_SPAN)) });
  });

  let size: { width: number; height: number } | null = null;
  const ctx = cv.getContext("2d")!;

  /** A rectangle as a path (corners in screen space), so a fill or a stroke is one recorded call. */
  const rect = (x0: number, y0: number, x1: number, y1: number): void => {
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y0);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x0, y1);
    ctx.closePath();
  };

  const draw = (g: Geometry): void => {
    if (size === null) return;
    sizeCanvas(cv, size);
    ctx.clearRect(0, 0, size.width, size.height);
    const view = fitBbox({ minX: 0, minY: 0, maxX: g.width, maxY: g.height },
                         size.width, size.height, 0.06);
    const sx = g.width / g.cols, sy = g.height / g.rows;
    // Ring k (1 = the parcels on the street) is the grid inset by k - 1 parcels on every side,
    // painted outermost first, each over the last: one rectangle per ring, not one per parcel.
    for (let k = 1; k <= g.deepest; k++) {
      const [x0, y0] = toScreen(view, (k - 1) * sx, g.height - (k - 1) * sy);
      const [x1, y1] = toScreen(view, g.width - (k - 1) * sx, (k - 1) * sy);
      ctx.fillStyle = ramp(stops, (k - 1) / (redRows - 1));
      rect(x0, y0, x1, y1);
      ctx.fill();
    }
    // The parcels, where they are big enough on screen to be told apart.
    const cell = sx * view.scaleX;
    if (cell >= 4) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < g.cols; i++) {
        const [x, ya] = toScreen(view, i * sx, 0);
        const [, yb] = toScreen(view, i * sx, g.height);
        ctx.moveTo(x, ya);
        ctx.lineTo(x, yb);
      }
      for (let j = 1; j < g.rows; j++) {
        const [xa, y] = toScreen(view, 0, j * sy);
        const [xb] = toScreen(view, g.width, j * sy);
        ctx.moveTo(xa, y);
        ctx.lineTo(xb, y);
      }
      ctx.stroke();
    }
    // The street, all the way round.
    const [ox, oy] = toScreen(view, 0, g.height);
    const [ex, ey] = toScreen(view, g.width, 0);
    ctx.strokeStyle = b.encoding.street;
    ctx.lineWidth = 3;
    rect(ox, oy, ex, ey);
    ctx.stroke();
    // A/P, measured in from the bottom street at the left quarter: the thickness the score
    // divides by the parcel side.
    const [tx, tyBottom] = toScreen(view, g.width / 4, 0);
    const [, tyTop] = toScreen(view, g.width / 4, g.thickness);
    ctx.strokeStyle = "#111111";
    ctx.fillStyle = "#111111";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(tx, tyBottom);
    ctx.lineTo(tx, tyTop);
    ctx.moveTo(tx - 5, tyTop);
    ctx.lineTo(tx + 5, tyTop);
    ctx.stroke();
    ctx.font = "600 13px sans-serif";
    ctx.fillText("A/P", tx + 7, (tyBottom + tyTop) / 2 + 4);
  };

  const render = (): void => {
    const s = state.get();
    const blk = blocks[s.block];
    const shape = s.shape ?? shapes[s.block];
    const n = s.homes ?? blk.n;
    const g = geometry(blk.area_m2, shape, n);
    shapeIn.value = String(toStep(shape, 1, SHAPE_MAX));
    shapeOut.textContent = `${fmt(shape, 2)} : 1`;
    homesIn.value = String(toStep(n, blk.n / HOMES_SPAN, blk.n * HOMES_SPAN));
    homesOut.textContent = n.toLocaleString("en-US");
    for (const [k, btn] of pick) btn.setAttribute("aria-pressed", String(k === s.block));
    const [dThick, dParcel, dDeep, dCrowd, dScore] = dds;
    dThick!.textContent = `${fmt(g.thickness, 3)} m`;
    dParcel!.textContent = `${fmt(g.parcel, 2)} m`;
    dDeep!.textContent = fmt(g.deep, 3);
    dCrowd!.textContent = `${fmt(g.crowding * 10_000, 3)} per ha`;
    const times = g.score / floor;
    dScore!.textContent = `${g.score.toPrecision(3)} · ${fmt(times, 2)}× the floor`;
    const own = s.shape === null && s.homes === null;
    note.textContent = (own
      ? `This rectangle has block ${blk.block_id}'s own area, perimeter and `
        + `${blk.n.toLocaleString("en-US")} buildings, so its score is the block's. `
      : `Block ${blk.block_id}'s area, reshaped. `)
      + `Its deepest parcels are ${g.deepest} ${g.deepest === 1 ? "row" : "rows"} from a street; `
      + (g.score >= floor ? "it clears the shipped floor." : "it falls under the shipped floor.");
    cv.setAttribute("aria-label", `A ${fmt(shape, 2)} to 1 rectangle of ${n} parcels, shaded by `
      + `rows from the street, ${g.deepest} at the deepest.`);
    draw(g);
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
