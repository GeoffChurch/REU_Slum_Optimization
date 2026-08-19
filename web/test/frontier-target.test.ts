import { strict as assert } from "node:assert";
import { test } from "node:test";
import { clipToXMax, leastClearing, parseChart, parseMethodStyles } from "../src/widgets/frontier.js";
import { fitAxes, toScreen, toWorld } from "../src/view/transform.js";

test("leastClearing finds the least index reaching the target", () => {
  // The identical search budget.prefix_to_permeability runs in Python, over the same monotone
  // sequence -- which is why the answer is exact rather than an interpolation off a curve.
  const perm = [0, 0.2, 0.49, 0.61, 0.66, 0.7];
  assert.equal(leastClearing(perm, 0.6), 3);
  assert.equal(leastClearing(perm, 0.2), 1);
  assert.equal(leastClearing(perm, 0), 0);
});

test("leastClearing reports -1 when the target is never reached", () => {
  assert.equal(leastClearing([0, 0.1, 0.2], 0.9), -1);
});

test("leastClearing agrees with a linear scan on random monotone data", () => {
  // The recurrence below is an arbitrary-but-deterministic number sequence, NOT the LCG its
  // constants suggest (the product overflows 2^53, so the modulus is not exact). That does not
  // matter here: `acc += rnd()` is non-decreasing whatever the generator's quality, which is the
  // only property the binary search's precondition needs.
  let seed = 12345;
  const rnd = (): number => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  for (let trial = 0; trial < 200; trial++) {
    const v: number[] = [];
    let acc = 0;
    for (let i = 0; i < 40; i++) { acc += rnd(); v.push(acc); }
    const target = rnd() * acc * 1.1;
    const expect = v.findIndex((x) => x >= target);
    assert.equal(leastClearing(v, target), expect, `trial ${trial}`);
  }
});

test("toWorld inverts toScreen on a fitAxes view whose two scales differ", () => {
  // The drag path, exactly: a pointer's screen x becomes a world displacement through `toWorld`,
  // on this widget's own axis pairing -- displacement in [0, 0.4] against permeability in [0, 1],
  // in a box wider than it is tall, so scaleX comes out several times scaleY. Every OTHER
  // transform test runs on `fitBbox` output, where the two scales enter equal and stay equal, so
  // nothing else in the suite would notice `toWorld` reusing one scale for both axes or
  // transposing them: the guide would simply track the pointer at the wrong rate -- visible, but
  // subtle enough to be "corrected" by eye with a fudge factor rather than recognised as a bug.
  const v = fitAxes([0, 0.4], [0, 1], 640, 360, 0.15);
  // Asserted so this test cannot silently degrade into the equal-scale case it exists to cover.
  assert.notEqual(v.scaleX, v.scaleY, "fitAxes returned equal scales; this test would be vacuous");
  for (const [x, y] of [[0, 0], [0.1, 0.6], [0.4, 1], [0.25, 0.33]] as const) {
    const [sx, sy] = toScreen(v, x, y);
    const [wx, wy] = toWorld(v, sx, sy);
    assert.ok(Math.abs(wx - x) < 1e-9, `x round-trip ${x} -> ${sx} -> ${wx}`);
    assert.ok(Math.abs(wy - y) < 1e-9, `y round-trip ${y} -> ${sy} -> ${wy}`);
  }
});

test("clipToXMax keeps a curve that never leaves the window untouched", () => {
  const xs = [0, 0.05, 0.2];
  const ys = [0, 0.4, 0.9];
  assert.deepEqual(clipToXMax(xs, ys, 0.4), { xs, ys });
});

test("clipToXMax ends a curve that runs past the window ON the window edge", () => {
  // The real shape: clearance_looped reaches 0.83 displacement while the plot shows [0, 0.4].
  // matplotlib's own axis clipping draws the fallback PNG's line to the axis edge; truncating at
  // the last sample INSIDE the window instead would end the line short of the edge, which reads
  // as the method terminating there.
  const { xs, ys } = clipToXMax([0, 0.3, 0.5], [0, 0.6, 0.8], 0.4);
  assert.deepEqual(xs, [0, 0.3, 0.4]);
  assert.equal(ys.length, 3);
  assert.ok(Math.abs(ys[2]! - 0.7) < 1e-12, `interpolated edge y was ${ys[2]}`);
});

test("clipToXMax survives the repeated-x samples the real bundle contains", () => {
  // greedy_arterial_access_displacement's first three samples all sit at displacement 0.0 (a road
  // that grazes no building). A zero-width segment is where a naive interpolation divides by zero,
  // and a single NaN in a <polyline>'s `points` makes the ENTIRE series render as nothing, with no
  // error anywhere -- the failure shape this branch keeps producing.
  const { xs, ys } = clipToXMax([0, 0, 0, 0.5], [0, 0.03, 0.03, 0.9], 0.4);
  assert.deepEqual(xs, [0, 0, 0, 0.4]);
  for (const y of ys) assert.ok(Number.isFinite(y), `non-finite y in ${JSON.stringify(ys)}`);
});

test("parseMethodStyles rejects a bundle method with no label or colour", () => {
  // Reachable by staleness, not by malformed input: labels and colours are baked into the bundle by
  // scripts/gen_frontier_bundle.py, and a deployed page can outlive the artifact it was generated
  // beside. A curve whose style went missing would be stroked with "undefined" and render as
  // nothing -- one method absent from a chart of eight, on a page that still looks entirely
  // correct. The keys come from the bundle itself, so nothing here names a method.
  const good = { clearance: { label: "Least-Cost Tree", colour: "#d9b64c" } };
  assert.deepEqual(parseMethodStyles(good).get("clearance"),
    { label: "Least-Cost Tree", colour: "#d9b64c" });
  const stale = { ...good, osm_footpaths: { road_m: [0], displacement: [0], permeability: [0] } };
  assert.throws(() => parseMethodStyles(stale), /no label\/colour for method osm_footpaths/);
});

test("parseChart rejects a missing, non-numeric or degenerate field instead of drawing with NaN", () => {
  // Same staleness class one level up: every stroke width, pad, tick target and axis label the
  // widget draws with arrives in this one block. A field the baker stopped emitting would otherwise
  // reach `stroke-width="NaN"` or a NaN-padded view -- both of which render nothing while throwing
  // nowhere.
  const full = {
    x_label: "displacement", y_label: "permeability", line_width: 2.5, guide_colour: "gray",
    guide_width: 1, guide_dash: "6 4", marker_radius: 2.5, grid_opacity: 0.12, tick_target: 5,
    pad: 0.15, slider_step: 0.01, permeability_max: 1,
  };
  assert.equal(parseChart(full).pad, 0.15);
  const { pad: _dropped, ...missing } = full;
  assert.throws(() => parseChart(missing), /frontier\.json's chart is missing "pad"/);
  assert.throws(() => parseChart({ ...full, line_width: "thick" }),
    /frontier\.json's chart "line_width" is not a finite number/);
  // Finite is not enough: 0 passes Number.isFinite, and a zero slider_step makes every drag NaN
  // (Math.round(v / 0)) while a zero line_width strokes nothing at all.
  assert.throws(() => parseChart({ ...full, slider_step: 0 }),
    /frontier\.json's chart "slider_step" must be positive/);
  // A zero marker radius draws no dot, which is how the samples became invisible in the first place.
  assert.throws(() => parseChart({ ...full, marker_radius: 0 }),
    /frontier\.json's chart "marker_radius" must be positive/);
  // pad is a fraction of the box applied to both sides, so half the box leaves no plot at all.
  assert.throws(() => parseChart({ ...full, pad: 0.5 }),
    /frontier\.json's chart "pad" must be in \[0, 0\.5\)/);
  // grid_opacity is the one drawn value where 0 is MEANINGFUL ("no gridlines", which is what the
  // fallback figure draws), so it is range-checked rather than required positive.
  assert.equal(parseChart({ ...full, grid_opacity: 0 }).grid_opacity, 0);
  assert.throws(() => parseChart({ ...full, grid_opacity: 1.5 }),
    /frontier\.json's chart "grid_opacity" must be in \[0, 1\]/);
  // Not an object at all -- what a truncated or replaced artifact deserializes to.
  assert.throws(() => parseChart(null), /frontier\.json's chart is not an object/);
});
