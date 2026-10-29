import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import {
  clampView, panBy, PEEL_URL, peelZoom, WHOLE, zoomAt, type PeelView,
} from "../src/widgets/peel-zoom.js";
import { fakeLocation, FakeElement, fireResize, installStubs, writeNow } from "./harness.js";

installStubs();

const NATIVE = 3696;
const WIDTH = 700;                         // the frame's CSS width: the cap is 3696 / 700 = 5.28x
const CAP = NATIVE / WIDTH;

/** The window a view shows, as fractions of the image. */
const windowOf = (v: PeelView): [number, number, number, number] =>
  [v.cx - 0.5 / v.zoom, v.cy - 0.5 / v.zoom, v.cx + 0.5 / v.zoom, v.cy + 0.5 / v.zoom];
const inside = (v: PeelView): boolean =>
  windowOf(v).every((t) => t >= -1e-12 && t <= 1 + 1e-12);

test("zoom is clamped between the whole image and its native resolution", () => {
  assert.equal(clampView({ zoom: 50, cx: 0.5, cy: 0.5 }, CAP).zoom, CAP);
  assert.equal(clampView({ zoom: 0.2, cx: 0.5, cy: 0.5 }, CAP).zoom, 1);
  let v = WHOLE;
  for (let i = 0; i < 40; i++) v = zoomAt(v, 1.6, 0.3, 0.7, CAP);
  assert.equal(v.zoom, CAP, "repeated zooming in stops at the cap");
  for (let i = 0; i < 40; i++) v = zoomAt(v, 1 / 1.6, 0.9, 0.1, CAP);
  assert.deepEqual(v, WHOLE, "zooming all the way out is the whole image, centred");
});

test("a pan stays inside the image, however far it is dragged", () => {
  let v = zoomAt(WHOLE, 4, 0.5, 0.5, CAP);
  for (const [dx, dy] of [[5, 5], [-30, 2], [0.3, -40], [100, 100]] as const) {
    v = panBy(v, dx, dy, CAP);
    assert.ok(inside(v), `window ${windowOf(v)} leaves the image`);
  }
  // Unzoomed there is nowhere to go.
  assert.deepEqual(panBy(WHOLE, 0.4, -0.2, CAP), WHOLE);
  // A hand-edited view off the image is pulled back inside.
  assert.ok(inside(clampView({ zoom: 3, cx: -2, cy: 9 }, CAP)));
});

test("zooming about a point keeps that point under it", () => {
  const v = zoomAt(WHOLE, 2, 0.25, 0.25, CAP);
  // The image point at frame (0.25, 0.25) before -- (0.25, 0.25) -- is still there after.
  assert.ok(Math.abs(v.cx + (0.25 - 0.5) / v.zoom - 0.25) < 1e-12);
});

test("the URL round-trips the view", () => {
  const loc = fakeLocation("");
  const first = urlStore(loc, writeNow).reserve().bind(PEEL_URL, WHOLE);
  const view: PeelView = { zoom: 3.25, cx: 0.4123, cy: 0.6017 };
  first.set(view);
  assert.ok(/peel=3.25/.test(loc.search()) && /peelx=0.4123/.test(loc.search()), loc.search());
  const again = urlStore(fakeLocation(loc.search()), writeNow).reserve().bind(PEEL_URL, WHOLE);
  assert.deepEqual(again.get(), view);
  // The whole image writes nothing: an unzoomed page keeps a clean URL.
  first.set(WHOLE);
  assert.equal(loc.search(), "");
});

/** The figure the page's generator writes: a frame holding the image. */
function mount(): { frame: FakeElement; state: () => StateSource<PeelView> } {
  const figure = new FakeElement("figure");
  figure.dataset.nativePx = String(NATIVE);
  const frame = new FakeElement("div");
  frame.append(new FakeElement("img"));
  figure.append(frame);
  (frame as unknown as { getBoundingClientRect: () => DOMRect }).getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: WIDTH, height: WIDTH }) as DOMRect;
  let store: StateSource<PeelView> | null = null;
  const slot = urlStore(fakeLocation(""), writeNow).reserve();
  peelZoom(figure as never, (initial) => { store = slot.bind(PEEL_URL, initial); return store; });
  fireResize(WIDTH, WIDTH);
  return { frame, state: () => store! };
}

const wheel = (frame: FakeElement, mods: { ctrlKey?: boolean; metaKey?: boolean }):
    { prevented: boolean } => {
  const ev = { ctrlKey: false, metaKey: false, ...mods, deltaY: -300, clientX: 200, clientY: 300,
               prevented: false, preventDefault(): void { this.prevented = true; } };
  frame.dispatch("wheel", ev);
  return ev;
};

test("a plain wheel scrolls the page; only Ctrl or Cmd with it zooms", () => {
  const { frame, state } = mount();
  const plain = wheel(frame, {});
  assert.deepEqual(state().get(), WHOLE, "a plain wheel zoomed the image");
  assert.equal(plain.prevented, false, "a plain wheel was kept from scrolling the page");
  const ctrl = wheel(frame, { ctrlKey: true });
  assert.ok(state().get().zoom > 1 && ctrl.prevented);
  const before = state().get().zoom;
  wheel(frame, { metaKey: true });
  assert.ok(state().get().zoom > before);
});

test("unzoomed, a one-finger swipe is the page's; zoomed, the widget takes the drag", () => {
  const { frame, state } = mount();
  assert.equal(frame.style.touchAction, "pan-x pan-y");
  state().set({ zoom: 2, cx: 0.5, cy: 0.5 });
  assert.equal(frame.style.touchAction, "none");
});
