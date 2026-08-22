import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { Bundle } from "../src/bundle.js";
import type { StateSource } from "../src/state.js";
import type { PermGraphState } from "../src/widgets/perm-graph.js";
import {
  armDrawFailure, canvasOf, DPR, fakeLocation, FakeElement, fireResize, installStubs,
  mountPoint as mountPointBase, writeNow,
} from "./harness.js";

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
installStubs();

const { urlStore } = await import("../src/url/store.js");
const { permGraph, PERM_GRAPH_URL } = await import("../src/widgets/perm-graph.js");

const BUNDLE_PATH = "../examples/perm-graph/bundle.json";
const bundle = JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as Bundle;

/** perm-graph-boot's own additions to the shared mount point: the bundle URL, the layer, and the
 * boot prefix `scripts/gen_site_pages.py` emits as `data-*` attributes. The DOM shape itself
 * (figure, glightbox anchor, image, figcaption) comes from the harness -- it is identical to what
 * field-boot.test.ts mounts against. */
function mountPoint(): FakeElement {
  const figure = mountPointBase();
  figure.dataset["bundle"] = BUNDLE_PATH;
  figure.dataset["layer"] = "current";
  figure.dataset["prefix"] = String(bundle.lens_b_index);
  return figure;
}

/** The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`:
 * `search` defaults to "", which claims no key, decodes nothing and writes nothing, so a caller
 * that passes no search gets the widget's own initial state exactly as `localState` gave it -- and
 * the URL test below gets the real decode path rather than a second, test-only one. */
async function mount(host: FakeElement, width: number, drawFailure: string | null = null,
                     search = ""): Promise<{
                       store: StateSource<PermGraphState>;
                       loc: ReturnType<typeof fakeLocation>;
                     }> {
  armDrawFailure(drawFailure);
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(bundle),
  });
  const loc = fakeLocation(search);
  const urls = urlStore(loc, writeNow);
  let bound: StateSource<PermGraphState> | null = null;
  permGraph(host as unknown as HTMLElement, (initial) => {
    bound = urls.bind(PERM_GRAPH_URL, initial);
    return bound;
  });
  // A macrotask, so the fetch chain has drained by the time this resolves.
  await new Promise((resolve) => setTimeout(resolve, 0));
  // The initial resize, played explicitly rather than auto-fired from inside `observe()` -- see
  // harness.ts's `FakeResizeObserver` for why that is what keeps `runOrReport` load-bearing: by the
  // time this call happens, the widget's own `fetch().then(boot).catch(...)` chain has already
  // fully settled, so there is no `.catch` left standing over a throw in here.
  try {
    fireResize(width, width);
  } finally {
    armDrawFailure(null);
  }
  // `boot`'s first statement is the `makeState` call, and every mount in this file feeds it the
  // committed bundle, so a null here is a widget that failed before it.
  assert.ok(bound !== null, "the widget never asked for a state store");
  return { store: bound, loc };
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
    // `.op`, not a bare string: the harness's shared `RecordingContext` records a `Call` object per
    // call (see harness.ts), where this file's own pre-extraction version recorded the operation
    // name alone. The comparison is otherwise the same one: is there at least one stroke, one fill.
    assert.ok(cv.ctx.calls.filter((c) => c.op === "stroke").length > 0, "nothing was stroked");
    assert.ok(cv.ctx.calls.filter((c) => c.op === "fill").length > 0, "nothing was filled");
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
    assert.equal(host.findAll("a").length, 0, "an <a> survives inside the mounted figure");

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
  assert.equal(host.findAll("img").length, 1);
  assert.equal(host.findAll("a").length, 1, "the lightbox link went with an image that is still needed");
  assert.ok(!host.querySelector("figcaption")!.textContent.includes("could not load"),
    "a container that is merely not laid out yet is not a failure to report");
});

test("the container becoming visible later draws at the width it finally gets", async () => {
  // The other half of skipping zero: skipping is only correct if the widget still draws when the
  // box arrives. A `<details>` opening or a tab switching fires the observer, not a window resize.
  const host = mountPoint();
  const img = host.querySelector("img")!;
  await mount(host, 0);
  fireResize(320, 320);

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
    fireResize(320, 320);

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
  assert.equal(host.findAll("a").length, 1, "the lightbox link went with an image that is still needed");
});

test("a prefix past the last one is clamped, not drawn out of range", async () => {
  // `?prefix=` is a bare non-negative integer -- how many prefixes THIS block has is a property of
  // the fetched bundle, which no codec can know. An out-of-range one must land on the last prefix
  // rather than index past the end of `b.prefix.current`, and the URL must stop carrying it.
  const { store, loc } = await mount(mountPoint(), 640, null, "prefix=99999");
  assert.equal(store.get().prefix, bundle.n_prefixes - 1);
  const written = loc.written.at(-1);
  assert.ok(written !== undefined, "the clamp never wrote a corrected URL");
  assert.ok(!written.includes("prefix=99999"), `the out-of-range prefix survived: "${written}"`);
});
