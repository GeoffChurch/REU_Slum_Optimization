import assert from "node:assert/strict";
import test from "node:test";

/** A ResizeObserver whose callbacks the test fires by hand. Installed on globalThis before the
 * module under test is imported, the same order the other fake-DOM suites use. */
class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  readonly targets: unknown[] = [];
  constructor(private readonly cb: (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(el: unknown): void { this.targets.push(el); }
  disconnect(): void { FakeResizeObserver.live.splice(FakeResizeObserver.live.indexOf(this), 1); }
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}
(globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;

const { observeSize } = await import("../src/dom/resize.js");

test("a zero-width box does not call back, so nothing draws into a hidden container", () => {
  const seen: number[] = [];
  observeSize({} as HTMLElement, (s) => seen.push(s.width));
  FakeResizeObserver.live.at(-1)!.fire(0, 0);
  assert.deepEqual(seen, [], "drew at zero width -- the fallback image is still the honest picture");
  FakeResizeObserver.live.at(-1)!.fire(320, 200);
  assert.deepEqual(seen, [320], "did not draw once the container became visible");
});

test("every positive resize calls back, because a container can narrow without the window moving", () => {
  const seen: number[] = [];
  observeSize({} as HTMLElement, (s) => seen.push(s.width));
  const obs = FakeResizeObserver.live.at(-1)!;
  obs.fire(700, 400);
  obs.fire(320, 200);
  obs.fire(1200, 700);
  assert.deepEqual(seen, [700, 320, 1200]);
});

test("the disposer stops the callbacks", () => {
  const seen: number[] = [];
  const stop = observeSize({} as HTMLElement, (s) => seen.push(s.width));
  const obs = FakeResizeObserver.live.at(-1)!;
  obs.fire(700, 400);
  stop();
  assert.equal(FakeResizeObserver.live.includes(obs), false, "disconnect() was not called");
});

test("the element the widget asked about is the element observed", () => {
  // `observe()` taking the wrong element is the one mistake this module could make that leaves
  // every assertion above green: callbacks would still arrive, at the wrong box's width.
  const el = {} as HTMLElement;
  observeSize(el, () => {});
  assert.deepEqual(FakeResizeObserver.live.at(-1)!.targets, [el]);
});
