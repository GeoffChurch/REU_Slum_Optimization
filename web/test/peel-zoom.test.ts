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

/** A touch event with these fingers down (client points), the way a browser hands one over. */
function touch(points: [number, number][], changed: [number, number][] = points):
    { prevented: boolean; preventDefault(): void; touches: unknown; changedTouches: unknown;
      timeStamp: number } {
  const list = (ps: [number, number][]): unknown =>
    Object.assign(ps.map(([x, y]) => ({ clientX: x, clientY: y })), { length: ps.length });
  return { prevented: false, preventDefault(): void { this.prevented = true; },
           touches: list(points), changedTouches: list(changed), timeStamp: 0 };
}

test("one finger on a touch screen never pans, zoomed or not: it is the page's scroll", () => {
  const { frame, state } = mount();
  const zoomed: PeelView = { zoom: 3, cx: 0.5, cy: 0.5 };
  state().set(zoomed);
  frame.dispatch("touchstart", touch([[300, 300]]));
  const moves = [touch([[300, 360]]), touch([[220, 420]])];
  for (const ev of moves) frame.dispatch("touchmove", ev);
  frame.dispatch("touchend", touch([], [[220, 420]]));
  assert.deepEqual(state().get(), zoomed, "a one-finger drag moved the image");
  assert.ok(moves.every((ev) => !ev.prevented), "a one-finger drag was kept from scrolling");
  // The browser keeps one finger in both states: nothing in the frame's own style takes it.
  assert.equal(frame.style.touchAction, undefined);
  // And it says how to move the image instead.
  const notice = frame.findAll("div").find((d) => d.textContent.includes("two fingers"));
  assert.equal(notice?.dataset.shown, "yes");
});

test("two fingers on a touch screen pan the image, and keep the page still", () => {
  const { frame, state } = mount();
  state().set({ zoom: 3, cx: 0.5, cy: 0.5 });
  frame.dispatch("touchstart", touch([[300, 300], [400, 300]]));
  const move = touch([[240, 300], [340, 300]]);          // both 60 px left: a pan, no pinch
  frame.dispatch("touchmove", move);
  const v = state().get();
  assert.ok(move.prevented, "a two-finger gesture also scrolled the page");
  assert.ok(Math.abs(v.zoom - 3) < 1e-9);
  assert.ok(Math.abs(v.cx - (0.5 + 60 / WIDTH / 3)) < 1e-9, `cx ${v.cx}`);
  // Spreading them pinches the zoom up.
  frame.dispatch("touchmove", touch([[200, 300], [380, 300]]));
  assert.ok(state().get().zoom > 3);
});
