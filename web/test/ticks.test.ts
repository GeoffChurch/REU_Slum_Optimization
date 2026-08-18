import { strict as assert } from "node:assert";
import { test } from "node:test";
import { niceTicks } from "../src/view/ticks.js";

test("tick STEPS are round numbers, not raw divisions", () => {
  // CONTROLLER CORRECTION (see note at end of brief): the invariant is on the STEP, not on each
  // tick value. Ticks are multiples of the step, so 0.3 and 0.4 are CORRECT ticks of a 0.1 step and
  // must not be rejected -- the widget's own x axis is [0, 0.4]. The failure being guarded is a
  // non-nice step: an axis labelled 0.07/0.14/0.21 looks plausible and reads as nonsense, and its
  // step has mantissa 7. Several ranges, because a single range lets a bad candidate list escape
  // through the fallback.
  for (const [min, max] of [[0, 0.4], [0, 1], [0, 784], [0.2, 0.9]] as const) {
    const ts = niceTicks(min, max, 5);
    assert.ok(ts.length >= 2, `need at least two ticks to have a step: ${ts}`);
    const step = ts[1]! - ts[0]!;
    const m = step / 10 ** Math.floor(Math.log10(Math.abs(step)));
    assert.ok([1, 2, 2.5, 5, 10].some((k) => Math.abs(m - k) < 1e-9),
      `step ${step} (mantissa ${m}) for [${min}, ${max}]`);
    for (let i = 2; i < ts.length; i++) {
      assert.ok(Math.abs(ts[i]! - ts[i - 1]! - step) < 1e-9, `uneven step at index ${i}: ${ts}`);
    }
  }
});

test("ticks span the range and stay inside it", () => {
  const ts = niceTicks(0, 1, 5);
  assert.ok(ts.length >= 3 && ts.length <= 9, `count ${ts.length}`);
  assert.ok(ts[0]! >= 0 && ts[ts.length - 1]! <= 1);
  assert.deepEqual([...ts].sort((a, b) => a - b), ts);
});

test("a degenerate range does not hang or return NaN", () => {
  const ts = niceTicks(0.5, 0.5, 5);
  assert.ok(ts.every((t) => Number.isFinite(t)));
});
