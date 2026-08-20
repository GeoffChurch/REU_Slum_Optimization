import { strict as assert } from "node:assert";
import { test } from "node:test";
import { fitAxes, fitBbox, nearest, panned, toScreen, toWorld, zoomed } from "../src/view/transform.js";

const BOX = { minX: 0, minY: 0, maxX: 100, maxY: 50 };

test("fitBbox centres the box and preserves aspect ratio", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  // The box is twice as wide as tall, so width binds: 400/100 = 4.
  assert.equal(v.scaleX, 4);
  const [, topY] = toScreen(v, 0, 50);
  const [, botY] = toScreen(v, 0, 0);
  // Vertically centred: equal slack above and below a 200px-tall drawing in 400px.
  assert.ok(Math.abs(topY - 100) < 1e-9, `topY ${topY}`);
  assert.ok(Math.abs(botY - 300) < 1e-9, `botY ${botY}`);
});

test("y is flipped: world up is screen up", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const [, yLow] = toScreen(v, 0, 0);
  const [, yHigh] = toScreen(v, 0, 50);
  assert.ok(yHigh < yLow, "larger world y must give smaller screen y");
});

test("toWorld inverts toScreen", () => {
  const v = fitBbox(BOX, 400, 400, 0.1);
  for (const [x, y] of [[0, 0], [100, 50], [37.5, 12.25]] as [number, number][]) {
    const [sx, sy] = toScreen(v, x, y);
    const [wx, wy] = toWorld(v, sx, sy);
    assert.ok(Math.abs(wx - x) < 1e-9 && Math.abs(wy - y) < 1e-9, `${wx},${wy} != ${x},${y}`);
  }
});

test("zoom keeps the cursor's world point under the cursor", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const anchor: [number, number] = [123, 210];
  const before = toWorld(v, ...anchor);
  const after = toWorld(zoomed(v, 2.5, ...anchor), ...anchor);
  assert.ok(Math.abs(before[0] - after[0]) < 1e-9, "world x under cursor moved");
  assert.ok(Math.abs(before[1] - after[1]) < 1e-9, "world y under cursor moved");
});

test("pan moves by exactly the screen delta", () => {
  const v = fitBbox(BOX, 400, 400, 0);
  const [x0, y0] = toScreen(v, 10, 10);
  const [x1, y1] = toScreen(panned(v, 25, -8), 10, 10);
  assert.equal(x1 - x0, 25);
  assert.equal(y1 - y0, -8);
});

test("nearest returns the closest index, not merely a close one", () => {
  const xs = [0, 10, 20];
  const ys = [0, 0, 0];
  assert.equal(nearest(xs, ys, 9.4, 0), 1);
  assert.equal(nearest(xs, ys, 5.1, 0), 1);
  assert.equal(nearest(xs, ys, 4.9, 0), 0);
});

test("fitBbox is uniform AND fits: the binding axis is the smaller ratio, at every box shape", () => {
  // Load-bearing, not cosmetic: render/canvas.ts converts metres to pixels through this scale for
  // road widths and node radii, so unequal scales would silently make geographic widths wrong on
  // one axis while everything still drew.
  //
  // `scaleX === scaleY` alone is not that guarantee, and the widths this chart is now laid out at
  // are exactly what would expose the difference: swap the `Math.min` for a `Math.max` and the two
  // scales stay equal -- uniform, so the old assertion held -- while the drawing overflows its box
  // on the binding axis, silently, at every aspect ratio but one. So the VALUE is pinned, computed
  // here the other way round from fitBbox's own expression.
  const bbox = { minX: 0, minY: 0, maxX: 100, maxY: 50 };
  const bw = bbox.maxX - bbox.minX;
  const bh = bbox.maxY - bbox.minY;
  for (const [w, h] of [[400, 400], [800, 200], [123, 457], [320, 320], [1200, 858]] as [number, number][]) {
    for (const pad of [0, 0.04, 0.15]) {
      const v = fitBbox(bbox, w, h, pad);
      const expected = Math.min(w / bw, h / bh) * (1 - 2 * pad);
      assert.equal(v.scaleX, v.scaleY, `${w}x${h} pad ${pad}`);
      assert.ok(Math.abs(v.scaleX - expected) < 1e-9,
        `${w}x${h} pad ${pad}: scale ${v.scaleX}, expected ${expected}`);
      // And the consequence, stated independently of the formula: the drawing fits inside the box.
      assert.ok(bw * v.scaleX <= w + 1e-9 && bh * v.scaleY <= h + 1e-9,
        `${w}x${h} pad ${pad}: ${bw * v.scaleX}x${bh * v.scaleY} overflows`);
    }
  }
});

test("fitAxes fits each axis independently", () => {
  const v = fitAxes([0, 0.4], [0, 1], 400, 200, 0);
  assert.equal(v.scaleX, 1000);            // 400px / 0.4
  assert.equal(v.scaleY, 200);             // 200px / 1
  const [x0, y0] = toScreen(v, 0, 0);
  const [x1, y1] = toScreen(v, 0.4, 1);
  assert.equal(x0, 0);
  assert.equal(y0, 200);
  assert.equal(x1, 400);
  assert.equal(y1, 0);
});

test("toWorld still inverts toScreen under independent scales", () => {
  const v = fitAxes([0, 0.4], [0, 1], 400, 200, 0.05);
  for (const [x, y] of [[0, 0], [0.4, 1], [0.137, 0.628]] as [number, number][]) {
    const [sx, sy] = toScreen(v, x, y);
    const [wx, wy] = toWorld(v, sx, sy);
    assert.ok(Math.abs(wx - x) < 1e-9 && Math.abs(wy - y) < 1e-9, `${wx},${wy}`);
  }
});
