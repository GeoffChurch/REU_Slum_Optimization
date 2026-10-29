/** A canvas that RASTERIZES, for the few tests that must look at pixels rather than at the call
 * log `harness.ts`'s `RecordingContext` keeps: what colour a block actually comes out, and whether
 * a sequence of frames leaves anything behind. Only the subset `render/city.ts` draws with:
 * `setTransform` (scale and translate), paths of `moveTo`/`lineTo`/`closePath`, `fill` (even-odd
 * or nonzero, here both even-odd: city.ts fills even-odd only), `stroke`, `arc` (stroked as a
 * polygon), `clearRect` of the whole canvas, and the 3-argument `drawImage` of another of these.
 *
 * Antialiasing as a browser does it, in the respect that matters here: each paint computes a
 * pixel's COVERAGE (here from an 8x8 grid of samples) and composites source-over with it, in
 * premultiplied RGBA -- so a shape covering part of a pixel blends with what is under it, which is
 * where both a sub-pixel block's colour and any frame-to-frame residue come from. Exact match to
 * Skia is not the point; the compositing rule is. */

const SS = 8;                                  // samples per pixel side

type Pt = [number, number];

function parse(color: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(color);
  if (m === null) throw new Error(`raster: only #rrggbb colours, not ${color}`);
  const v = parseInt(m[1]!, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function insideEvenOdd(polys: Pt[][], x: number, y: number): boolean {
  let inside = false;
  for (const poly of polys) {
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i]!;
      const [xj, yj] = poly[j]!;
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

function nearSegments(polys: Pt[][], closed: boolean[], x: number, y: number, r: number): boolean {
  const r2 = r * r;
  for (let p = 0; p < polys.length; p++) {
    const poly = polys[p]!;
    const n = closed[p] ? poly.length : poly.length - 1;
    for (let i = 0; i < n; i++) {
      const [ax, ay] = poly[i]!;
      const [bx, by] = poly[(i + 1) % poly.length]!;
      const dx = bx - ax, dy = by - ay;
      const len2 = dx * dx + dy * dy;
      const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
      const ex = ax + t * dx - x, ey = ay + t * dy - y;
      if (ex * ex + ey * ey <= r2) return true;
    }
  }
  return false;
}

export class PixelContext {
  /** Premultiplied RGBA, 0..1, row-major. */
  data: Float64Array;
  fillStyle = "#000000";
  strokeStyle = "#000000";
  lineWidth = 1;
  private m = [1, 0, 0, 1, 0, 0];
  private polys: Pt[][] = [];
  private closed: boolean[] = [];

  constructor(readonly canvas: PixelCanvas) {
    this.data = new Float64Array(canvas.width * canvas.height * 4);
  }

  resize(): void { this.data = new Float64Array(this.canvas.width * this.canvas.height * 4); }

  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void {
    if (b !== 0 || c !== 0) throw new Error("raster: scale and translate only");
    this.m = [a, b, c, d, e, f];
  }
  private dev(x: number, y: number): Pt {
    return [this.m[0]! * x + this.m[4]!, this.m[3]! * y + this.m[5]!];
  }

  beginPath(): void { this.polys = []; this.closed = []; }
  moveTo(x: number, y: number): void { this.polys.push([this.dev(x, y)]); this.closed.push(false); }
  lineTo(x: number, y: number): void { this.polys.at(-1)!.push(this.dev(x, y)); }
  closePath(): void { this.closed[this.closed.length - 1] = true; }
  arc(x: number, y: number, r: number, a0: number, a1: number): void {
    const n = 48;
    const pts: Pt[] = [];
    for (let i = 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      pts.push(this.dev(x + r * Math.cos(a), y + r * Math.sin(a)));
    }
    this.polys.push(pts);
    this.closed.push(false);
  }

  clearRect(): void { this.data.fill(0); }

  fill(_rule?: CanvasFillRule): void {
    this.paint(parse(this.fillStyle), 0, (x, y) => insideEvenOdd(this.polys, x, y));
  }

  stroke(): void {
    const r = (this.lineWidth * this.m[0]!) / 2;
    this.paint(parse(this.strokeStyle), r,
               (x, y) => nearSegments(this.polys, this.closed, x, y, r));
  }

  drawImage(image: PixelCanvas, _dx: number, _dy: number): void {
    const src = image.ctx.data;
    for (let i = 0; i < this.data.length; i += 4) {
      this.over(i, src[i]!, src[i + 1]!, src[i + 2]!, src[i + 3]!);
    }
  }

  /** The colour at device pixel (x, y), un-premultiplied to 0..255, with its alpha 0..1. */
  pixel(x: number, y: number): { rgb: [number, number, number]; a: number } {
    const i = (y * this.canvas.width + x) * 4;
    const a = this.data[i + 3]!;
    const un = (v: number): number => (a === 0 ? 0 : (v / a) * 255);
    return { rgb: [un(this.data[i]!), un(this.data[i + 1]!), un(this.data[i + 2]!)], a };
  }

  private over(i: number, r: number, g: number, b: number, a: number): void {
    const k = 1 - a;
    this.data[i] = r + this.data[i]! * k;
    this.data[i + 1] = g + this.data[i + 1]! * k;
    this.data[i + 2] = b + this.data[i + 2]! * k;
    this.data[i + 3] = a + this.data[i + 3]! * k;
  }

  private paint(rgb: [number, number, number], pad: number,
                hit: (x: number, y: number) => boolean): void {
    const all = this.polys.flat();
    if (all.length === 0) return;
    const W = this.canvas.width, H = this.canvas.height;
    const x0 = Math.max(0, Math.floor(Math.min(...all.map((p) => p[0])) - pad));
    const x1 = Math.min(W - 1, Math.ceil(Math.max(...all.map((p) => p[0])) + pad));
    const y0 = Math.max(0, Math.floor(Math.min(...all.map((p) => p[1])) - pad));
    const y1 = Math.min(H - 1, Math.ceil(Math.max(...all.map((p) => p[1])) + pad));
    const [r, g, b] = rgb.map((v) => v / 255) as [number, number, number];
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        let n = 0;
        for (let sy = 0; sy < SS; sy++) {
          for (let sx = 0; sx < SS; sx++) {
            if (hit(px + (sx + 0.5) / SS, py + (sy + 0.5) / SS)) n++;
          }
        }
        if (n === 0) continue;
        const cov = n / (SS * SS);
        this.over((py * W + px) * 4, r * cov, g * cov, b * cov, cov);
      }
    }
  }
}

/** The element `document.createElement("canvas")` returns while a raster test runs: resizing its
 * backing store (`width`/`height`) reallocates its pixels, as a real canvas's does. */
export class PixelCanvas {
  readonly style: Record<string, string> = {};
  private w = 0;
  private h = 0;
  readonly ctx: PixelContext;
  constructor() { this.ctx = new PixelContext(this); }
  get width(): number { return this.w; }
  set width(v: number) { this.w = v; this.ctx.resize(); }
  get height(): number { return this.h; }
  set height(v: number) { this.h = v; this.ctx.resize(); }
  getContext(): PixelContext { return this.ctx; }
}
