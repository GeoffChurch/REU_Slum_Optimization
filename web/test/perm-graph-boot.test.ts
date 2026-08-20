import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { Bundle } from "../src/bundle.js";

/** PermGraph is the widget that has been live on the public site the longest and the only one with
 * no TypeScript-level coverage at all: `widgets-bundle.test.ts` never reaches its `boot()` (it dies
 * at `fetch is not defined`, measured), and nothing else imported it. So the change this task makes
 * to it -- the fallback <img> now goes after the first successful draw instead of the moment the
 * canvas is inserted -- had nothing that could redden. This file is that guard.
 *
 * Same minimal-stub spirit as frontier-boot.test.ts and svg.test.ts: no jsdom, one fake element
 * class, and a recording 2D context standing in for the whole canvas API `render/canvas.ts` uses
 * (all of it void methods and assignable properties -- there is nothing to return).
 */
let clock = 0;

/** Armed by `mount(host, width, drawFailure)` to make the FIRST draw throw. It has to be a module
 * variable rather than a field set on the element: the canvas is created inside `boot()`, which runs
 * a microtask after `permGraph` returns, so no test holds a reference to it in time. */
let NEXT_DRAW_FAILURE: string | null = null;

/** Every canvas call, in order. `draw()` is a long sequence of strokes and fills with no return
 * value anywhere, so "did it draw" is a question about this list and nothing else. */
class RecordingContext {
  readonly calls: string[] = [];
  /** When set, the next drawing call throws this. The only way to make `draw()` fail from a test:
   * every path through it is canvas calls, so there is nothing else to reach in. */
  failWith: string | null = NEXT_DRAW_FAILURE;
  fillStyle = "";
  strokeStyle = "";
  lineWidth = 0;
  lineCap = "";
  globalAlpha = 1;
  readonly transforms: number[][] = [];
  setTransform(a: number, b: number, c: number, d: number, e: number, f: number): void {
    this.transforms.push([a, b, c, d, e, f]);
  }
  /** The instant of the first drawing call, on the same counter `FakeElement` stamps creations and
   * removals with -- so a test can assert the fallback image went after a PICTURE existed, not
   * merely after a canvas element did. The element is created either way; the drawing is not. */
  firstDrawAt: number | null = null;
  private record(name: string): void {
    if (this.failWith !== null) throw new Error(this.failWith);
    this.calls.push(name);
    this.firstDrawAt ??= ++clock;
  }
  clearRect(): void { this.record("clearRect"); }
  beginPath(): void { this.record("beginPath"); }
  closePath(): void { this.record("closePath"); }
  moveTo(): void { this.record("moveTo"); }
  lineTo(): void { this.record("lineTo"); }
  arc(): void { this.record("arc"); }
  fill(): void { this.record("fill"); }
  stroke(): void { this.record("stroke"); }
}

class FakeElement {
  readonly tagName: string;
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  readonly listeners: string[] = [];
  textContent = "";
  /** Only a canvas has one; the widget reads it back through `getContext`. */
  readonly ctx = new RecordingContext();
  /** The backing store, which `sizeCanvas` sets from the observed box times devicePixelRatio. */
  width = 0;
  height = 0;
  title = "";
  readonly createdAt = ++clock;
  removedAt: number | null = null;

  /** UPPERCASE, as a real element reports it -- `dom/fallback.ts` decides whether the wrapper around
   * the fallback image is the glightbox anchor by testing `tagName === "A"`. */
  constructor(tagName: string) { this.tagName = tagName.toUpperCase(); }
  getContext(): RecordingContext { return this.ctx; }
  append(...nodes: (FakeElement | string)[]): void {
    for (const n of nodes) {
      if (typeof n === "string") {
        const text = new FakeElement("#TEXT");
        text.textContent = n;
        text.parent = this;
        this.children.push(text);
      } else {
        n.parent = this;
        this.children.push(n);
      }
    }
  }
  insertBefore(node: FakeElement, _ref: FakeElement | null): void {
    node.parent = this;
    this.children.push(node);
  }
  /** Detaches, as a real `remove()` does -- `dom/fallback.ts` asks whether the anchor has element
   * children left after the image goes, and a fake that only timestamped would answer "yes". */
  remove(): void {
    this.removedAt = ++clock;
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }
  get parentElement(): FakeElement | null { return this.parent; }
  addEventListener(name: string): void { this.listeners.push(name); }
  querySelector(selector: string): FakeElement | null {
    return this.descendants().find((c) => c.tagName === selector.toUpperCase()) ?? null;
  }
  descendants(): FakeElement[] { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  all(tagName: string): FakeElement[] {
    return this.descendants().filter((c) => c.tagName === tagName.toUpperCase());
  }
  text(): string {
    return [this.textContent, ...this.descendants().map((c) => c.textContent)].join(" ");
  }
}

(globalThis as Record<string, unknown>).document = {
  createElement: (tag: string): FakeElement => new FakeElement(tag),
};
/** `render/canvas.ts` reads `devicePixelRatio` off it, and 2 rather than 1 is what makes the
 * backing-store assertion below able to fail: at 1 the scaled size and the CSS size coincide. */
(globalThis as Record<string, unknown>).window = { devicePixelRatio: 2 };

/** The observer the widget lays itself out from. `observe()` delivers an initial observation the way
 * a real one does, taking the width from `NEXT_WIDTH` -- the canvas has no layout engine here, and
 * a zero is the case that matters (a hidden container, a collapsed <details>).
 *
 * That delivery is DEFERRED to a microtask, and the deferral is the whole point rather than a
 * detail. A real ResizeObserver never calls back from inside `observe()`; it delivers from the
 * browser's own dispatch, after the code that called `observe()` has returned. Firing synchronously
 * instead put the first draw INSIDE the mount's `fetch().then(boot).catch(showWidgetError)` chain,
 * where that chain's `.catch` absorbs any throw -- so `runOrReport` was redundant in the fake and in
 * the fake only, and a test proving the throw reaches the page passed with the wrapper deleted. A
 * microtask scheduled from within a `.then` handler runs after the handler returns, so it is outside
 * the chain exactly as the browser's dispatch is, while still landing before the `setTimeout(0)`
 * `mount()` awaits -- which is what keeps "the chart is drawn by the time mount() resolves" true. */
let NEXT_WIDTH = 0;
class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  constructor(private readonly cb:
              (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(): void {
    const [width, height] = [NEXT_WIDTH, NEXT_WIDTH];
    queueMicrotask(() => this.fire(width, height));
  }
  disconnect(): void {}
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}
(globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;

const { localState } = await import("../src/state.js");
const { permGraph } = await import("../src/widgets/perm-graph.js");

const BUNDLE_PATH = "../examples/perm-graph/bundle.json";
const bundle = JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as Bundle;
const DPR = 2;

/** The mount point `scripts/gen_site_pages.py` emits for this widget: the <figure> itself, carrying
 * the bundle URL and the boot prefix, with the fallback <img> inside the <a class="glightbox">
 * mkdocs-glightbox wraps it in and a <figcaption> after it. */
function mountPoint(): FakeElement {
  const figure = new FakeElement("figure");
  const anchor = new FakeElement("a");
  anchor.append(new FakeElement("img"));
  figure.append(anchor, new FakeElement("figcaption"));
  figure.dataset["bundle"] = BUNDLE_PATH;
  figure.dataset["layer"] = "current";
  figure.dataset["prefix"] = String(bundle.lens_b_index);
  return figure;
}

async function mount(host: FakeElement, width: number,
                    drawFailure: string | null = null): Promise<void> {
  NEXT_WIDTH = width;
  NEXT_DRAW_FAILURE = drawFailure;
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(bundle),
  });
  permGraph(host as unknown as HTMLElement, localState);
  // A macrotask, so every microtask -- the fetch chain AND the observer's deferred first delivery --
  // has drained by the time this resolves.
  await new Promise((resolve) => setTimeout(resolve, 0));
  NEXT_DRAW_FAILURE = null;
}

function canvasOf(host: FakeElement): FakeElement {
  const cv = host.children.find((c) => c.tagName === "CANVAS");
  assert.ok(cv !== undefined, "no canvas was inserted into the figure");
  return cv;
}

test("the widget mounts, draws, and only then drops the fallback image and its lightbox link",
  async () => {
    const host = mountPoint();
    const img = host.querySelector("img")!;
    const anchor = host.querySelector("a")!;
    await mount(host, 640);

    assert.ok(!host.querySelector("figcaption")!.textContent.includes("could not load"),
      host.querySelector("figcaption")!.textContent);
    const cv = canvasOf(host);
    assert.ok(cv.ctx.calls.filter((c) => c === "stroke").length > 0, "nothing was stroked");
    assert.ok(cv.ctx.calls.filter((c) => c === "fill").length > 0, "nothing was filled");
    // The backing store is the observed box times devicePixelRatio -- the whole reason a container
    // resize has to reach this widget at all: a stretched canvas with a stale backing store is a
    // blurred picture that nothing reports.
    assert.equal(cv.width, 640 * DPR);
    assert.equal(cv.height, 640 * DPR);
    assert.deepEqual(cv.ctx.transforms.at(-1), [DPR, 0, 0, DPR, 0, 0]);
    // The canvas pans on a pointer drag; without this a touch drag scrolls the page and the browser
    // cancels the pointer stream mid-gesture. The pointer CAPTURE that goes with it is deliberately
    // not asserted anywhere -- there is no browser here, so no assertion could tell a captured drag
    // from an uncaptured one, and field-boot.test.ts says the same of its own pair.
    assert.equal(cv.style["touchAction"], "none");

    // ORDER, not merely both. This is the change: the <img> used to go the instant the canvas was
    // inserted, before a single pixel had been drawn.
    assert.notEqual(img.removedAt, null, "the fallback image outlived the drawing");
    assert.ok(img.removedAt! > cv.ctx.firstDrawAt!,
      `fallback removed at ${img.removedAt} but nothing was drawn until ${cv.ctx.firstDrawAt}`);
    // And the glightbox <a> goes with it: an anchor emptied of its image is invisible, still
    // focusable, and announced as a link with no text. PermGraph shipped exactly that.
    assert.notEqual(anchor.removedAt, null, "the glightbox anchor outlived its image");
    assert.equal(host.all("a").length, 0, "an <a> survives inside the mounted figure");

    // Every number the picture shows is also on the page as text, at the baked values for the boot
    // prefix -- read off the bundle, so this cannot drift into asserting a typed number.
    const i = bundle.lens_b_index;
    assert.match(host.text(),
      new RegExp(`${bundle.prefix.road_m[i]!.toFixed(0)} m of road`));
    assert.match(host.text(),
      new RegExp(`${(bundle.prefix.permeability[i]! * 100).toFixed(1)}% permeability`));
  });

test("a zero-width container draws nothing and leaves the static figure in place", async () => {
  // The case the old code could not survive: it removed the <img> on the way in, so a figure that
  // mounted into a collapsed container (a hidden tab, a closed <details>, a print layout) was left
  // blank -- no picture, no message. Now nothing is drawn, so the static picture is still the
  // honest one, and it is still there.
  const host = mountPoint();
  const img = host.querySelector("img")!;
  await mount(host, 0);

  const cv = canvasOf(host);
  assert.deepEqual(cv.ctx.calls, [], "drew into a zero-width canvas");
  assert.equal(cv.width, 0, "sized a backing store for a box with no width");
  assert.equal(img.removedAt, null,
    "the fallback image was removed although nothing was ever drawn in its place");
  assert.equal(host.all("img").length, 1);
  assert.equal(host.all("a").length, 1, "the lightbox link went with an image that is still needed");
  assert.ok(!host.querySelector("figcaption")!.textContent.includes("could not load"),
    "a container that is merely not laid out yet is not a failure to report");
});

test("the container becoming visible later draws at the width it finally gets", async () => {
  // The other half of skipping zero: skipping is only correct if the widget still draws when the
  // box arrives. A `<details>` opening or a tab switching fires the observer, not a window resize.
  const host = mountPoint();
  const img = host.querySelector("img")!;
  await mount(host, 0);
  FakeResizeObserver.live.at(-1)!.fire(320, 320);

  const cv = canvasOf(host);
  assert.ok(cv.ctx.calls.length > 0, "still blank after the container was laid out");
  assert.equal(cv.width, 320 * DPR);
  assert.notEqual(img.removedAt, null, "the fallback outlived a real drawing");
});

test("a throw while RE-drawing reaches the caption instead of vanishing into the console",
  async () => {
    // The failure path the observer introduced. A resize callback runs from the browser's own
    // dispatch, outside the mount's `fetch().then(boot).catch(showWidgetError)` chain, so a throw in
    // it is an uncaught exception and nothing else -- and by then the fallback <img> is gone, so the
    // reader is left with a blank figure, no message, and a page that still looks laid out. This
    // widget is the one live on the public site, and it had nothing covering that at all.
    const host = mountPoint();
    await mount(host, 640);
    const cv = canvasOf(host);
    assert.ok(cv.ctx.calls.length > 0, "nothing drew, so the re-draw below proves nothing");

    cv.ctx.failWith = "boom while re-drawing";
    FakeResizeObserver.live.at(-1)!.fire(320, 320);

    assert.match(host.querySelector("figcaption")!.textContent,
      /PermGraph could not load interactively .*boom while re-drawing/);
    assert.match(host.querySelector("figcaption")!.textContent,
      /The static image above still applies\./);
  });

test("a throw on the FIRST draw is reported and keeps the static image", async () => {
  // The same route, on the delivery that is not a resize at all: the observer's initial observation.
  // It arrives after `boot()` has returned (a real one always does), so the mount's `.catch` is
  // already spent -- and this is the case where the message it writes has to be true, because the
  // picture it points the reader at is the one that must NOT have been removed.
  const host = mountPoint();
  const img = host.querySelector("img")!;
  await mount(host, 640, "boom on the first draw");

  assert.match(host.querySelector("figcaption")!.textContent,
    /PermGraph could not load interactively .*boom on the first draw/);
  assert.equal(img.removedAt, null,
    "the fallback image was removed although the drawing that replaces it threw");
  assert.equal(host.all("a").length, 1, "the lightbox link went with an image that is still needed");
});
