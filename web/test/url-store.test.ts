import { strict as assert } from "node:assert";
import { test } from "node:test";
import { boolParam, enumParam, intParam, type UrlCodec } from "../src/url/param.js";
import { debounce, urlStore } from "../src/url/store.js";
import { fakeLocation, fakeTimers, writeNow } from "./harness.js";

interface Demo { prefix: number; layer: "conductance" | "current"; halos: boolean }
const DEMO: UrlCodec<Demo> = {
  prefix: intParam("prefix"),
  layer: enumParam("layer", ["conductance", "current"] as const),
  halos: boolParam("halos"),
};
const INITIAL: Demo = { prefix: 0, layer: "current", halos: true };

test("an absent key leaves the widget's own initial alone, and writes nothing", () => {
  const loc = fakeLocation("");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.deepEqual(s.get(), INITIAL);
  assert.deepEqual(loc.written, []);
});

test("a present key overrides the initial and survives a write unchanged", () => {
  const loc = fakeLocation("prefix=14&layer=conductance");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.equal(s.get().prefix, 14);
  assert.equal(s.get().layer, "conductance");
  s.set({ halos: false });
  assert.equal(loc.written.at(-1), "prefix=14&layer=conductance&halos=0");
});

test("only values DIFFERING from the initial are emitted", () => {
  const loc = fakeLocation("");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  s.set({ prefix: 9 });
  assert.equal(loc.written.at(-1), "prefix=9");
  s.set({ prefix: 0 });
  assert.equal(loc.written.at(-1), "", "back to the initial: the key goes away entirely");
});

test("unclaimed params are preserved verbatim, and keep their original order", () => {
  const loc = fakeLocation("utm_source=paper&prefix=3&ref=abc");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  s.set({ layer: "conductance" });
  assert.equal(loc.written.at(-1), "utm_source=paper&ref=abc&prefix=3&layer=conductance");
});

test("an unusable value self-corrects: the initial is used AND the key is dropped at once", () => {
  const loc = fakeLocation("prefix=-4&layer=current");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.equal(s.get().prefix, 0, "the widget's own initial, not -4");
  assert.equal(loc.written.at(-1), "", "written without waiting for the reader to touch anything");
});

test("two bindings share one query string and one write", () => {
  interface Other { budget: number }
  const OTHER: UrlCodec<Other> = { budget: intParam("budget") };
  const loc = fakeLocation("");
  const store = urlStore(loc, writeNow);
  const a = store.bind(DEMO, INITIAL);
  const b = store.bind(OTHER, { budget: 3000 });
  a.set({ prefix: 2 });
  b.set({ budget: 5000 });
  assert.equal(loc.written.at(-1), "prefix=2&budget=5000");
});

test("subscribers fire on set, exactly like localState's", () => {
  const seen: number[] = [];
  const s = urlStore(fakeLocation(""), writeNow).bind(DEMO, INITIAL);
  s.subscribe((v) => seen.push(v.prefix));
  s.set({ prefix: 1 });
  s.set({ prefix: 2 });
  assert.deepEqual(seen, [1, 2]);
});

test("debounce collapses a burst into ONE write, carrying the last value", () => {
  const timers = fakeTimers();
  const seen: string[] = [];
  const schedule = debounce(300, timers);
  schedule(() => seen.push("a"));
  schedule(() => seen.push("b"));
  schedule(() => seen.push("c"));
  assert.equal(timers.pending(), 1, "the earlier two timers were cleared, not left queued");
  assert.deepEqual(seen, [], "nothing has run before the window elapses");
  timers.run();
  assert.deepEqual(seen, ["c"]);
});

test("a drag through the debounce writes once, not once per state change", () => {
  const timers = fakeTimers();
  const loc = fakeLocation("");
  const s = urlStore(loc, debounce(300, timers)).bind(DEMO, INITIAL);
  for (let i = 1; i <= 40; i++) s.set({ prefix: i });
  assert.deepEqual(loc.written, [], "not one write yet");
  timers.run();
  assert.deepEqual(loc.written, ["prefix=40"]);
});
