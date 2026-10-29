import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { CityBundle } from "../src/screen_map.js";
import { ranking, scores, selectAt } from "../src/model/screen.js";
import { blocksIn, closeupBox } from "../src/render/closeup.js";
import { localState } from "../src/state.js";
import {
  Call, fireAnimationFrame, FakeElement, FakeResizeObserver, fireResize, installStubs, lastFrame,
  mountPoint,
} from "./harness.js";
import { screenMap } from "../src/widgets/screen-map.js";

installStubs();

const ct = JSON.parse(
  readFileSync("../examples/screen-map/capetown.json", "utf8")) as CityBundle;
const nb = JSON.parse(
  readFileSync("../examples/screen-map/nairobi.json", "utf8")) as CityBundle;
const E = ct.encoding;
const PAD = 450;

(globalThis as Record<string, unknown>).fetch = (url: string): Promise<unknown> =>
  Promise.resolve({
    ok: true, status: 200, statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(url.includes("nairobi") ? nb : ct),
  });

/** The Demo page's mount: ScreenMap with `data-closeup-pad-m` (or `pad`, when given as a string
 * to test the attribute's own validation). Both canvases get a size: the close-up's observer is
 * registered first, so `fireResize` (the most recent observer) is still the city map's. */
async function mount(pad: string | null = String(PAD)): Promise<FakeElement> {
  const host = mountPoint();
  host.dataset.bundleCapetown = "capetown.json";
  host.dataset.bundleNairobi = "nairobi.json";
  if (pad !== null) host.dataset.closeupPadM = pad;
  const before = FakeResizeObserver.live.length;
  screenMap(host as never, localState);
  await new Promise((resolve) => setTimeout(resolve, 0));
  const mine = FakeResizeObserver.live.slice(before);
  if (mine.length === 2) mine[0]!.fire(300, 300);
  if (mine.length > 0) fireResize(500, 500);
  return host;
}

const canvases = (host: FakeElement): FakeElement[] =>
  host.children.filter((c) => c.tagName === "CANVAS");

function strokes(cv: FakeElement, colour: string): Call[] {
  return lastFrame(cv).filter((c) => c.op === "stroke" && c.strokeStyle === colour);
}

test("without the attribute there is one canvas: Screening and Explore are unchanged", async () => {
  const host = await mount(null);
  assert.equal(canvases(host).length, 1);
});

test("the close-up outlines the followed block in the follow colour, last", async () => {
  const host = await mount();
  const [, close] = canvases(host);
  assert.ok(close !== undefined, "no close-up canvas");
  const frame = lastFrame(close);
  const last = frame.at(-1)!;
  assert.equal(last.op, "stroke");
  assert.equal(last.strokeStyle, E.follow_color);
  // One fill per block in the window, the window being the block's own bbox grown by PAD.
  const near = blocksIn(ct, closeupBox(ct, ct.follow!.index, PAD));
  assert.ok(near.length > 1 && near.length < 2000, `implausible window: ${near.length} blocks`);
  assert.equal(frame.filter((c) => c.op === "fill").length, near.length);
});

test("the close-up's red outlines are exactly the selected blocks in its window", async () => {
  const host = await mount();
  const [, close] = canvases(host);
  const s = scores(ct, "depth_density_proxy");
  const order = ranking(ct, "depth_density_proxy");
  const shipped = ct.floors.find((f) => f.metric === "depth_density_proxy")!.value;
  const near = new Set(blocksIn(ct, closeupBox(ct, ct.follow!.index, PAD)));
  const inWindow = (floor: number): number => {
    const k = selectAt(ct, order, s, floor).count;
    let n = 0;
    for (let i = 0; i < k; i++) if (near.has(order[i]!)) n++;
    return n;
  };
  assert.equal(strokes(close!, E.selected_color).length, inWindow(shipped));
  // Lower the floor: more of the window is selected, and the close-up redraws to say so.
  const slider = host.findAll("input").find((i) => i.type === "range")!;
  const lower = shipped / 4;
  slider.value = String(lower);
  slider.dispatch("input");
  fireAnimationFrame();
  assert.ok(inWindow(lower) > inWindow(shipped), "this test needs a floor that changes the window");
  assert.equal(strokes(close!, E.selected_color).length, inWindow(lower));
});

test("a city with no followed block hides the close-up", async () => {
  const host = await mount();
  const [, close] = canvases(host);
  const toggle = host.findAll("input").find((i) => i.type === "checkbox")!;
  toggle.checked = true;
  toggle.dispatch("change");
  fireAnimationFrame();
  assert.equal(close!.style.display, "none");
});

test("a close-up margin that is not a positive number fails into the caption", async () => {
  const host = await mount("wide");
  assert.match(host.find("figcaption")!.textContent, /data-closeup-pad-m must be a positive/);
});
