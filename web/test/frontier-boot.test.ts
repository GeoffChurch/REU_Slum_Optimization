import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { localState } from "../src/state.js";
import { frontier } from "../src/widgets/frontier.js";
import type { FrontierBundle } from "../src/frontier.js";

/** Mounts the widget for real -- against the COMMITTED bundle, not a toy one -- and asserts a chart
 * actually got drawn. Every defect this branch has produced shared one shape: the page still looked
 * fine while the widget was silently dead (a circular import that meant it never mounted; labels
 * rendered outside the viewport; a NaN coordinate that made a polyline draw nothing). None of those
 * threw where a human would see it, and the PNG fallback kept the page looking correct. So this
 * test asks the two questions a human looking at the page cannot: did every method's curve reach
 * the SVG, and is every coordinate in it a number.
 *
 * No browser and no jsdom: the same minimal-stub spirit as svg.test.ts, one class standing in for
 * every element the widget touches. `document` is stubbed on globalThis before any of the widget's
 * functions run (its module body touches no DOM -- only its function bodies do, and those are
 * called from inside the test below).
 */
let clock = 0;

class FakeElement {
  readonly tagName: string;
  readonly attrs = new Map<string, string>();
  children: FakeElement[] = [];
  readonly listeners: string[] = [];
  readonly style: Record<string, string> = {};
  readonly dataset: Record<string, string> = {};
  textContent = "";
  /** Creation and removal instants on one shared counter, so a test can assert the ORDER of two
   * DOM events and not merely that both happened -- which is the whole of "the fallback image goes
   * only once a chart has been drawn in its place". */
  readonly createdAt = ++clock;
  removedAt: number | null = null;

  constructor(tagName: string) { this.tagName = tagName; }
  setAttribute(name: string, value: string): void { this.attrs.set(name, value); }
  getAttribute(name: string): string | null { return this.attrs.get(name) ?? null; }
  append(...nodes: (FakeElement | string)[]): void {
    for (const n of nodes) if (typeof n !== "string") this.children.push(n);
  }
  insertBefore(node: FakeElement, _ref: FakeElement | null): void { this.children.push(node); }
  replaceChildren(...nodes: FakeElement[]): void { this.children = [...nodes]; }
  remove(): void { this.removedAt = ++clock; }
  addEventListener(name: string): void { this.listeners.push(name); }
  setPointerCapture(): void {}
  releasePointerCapture(): void {}
  /** Every fake element reports the same layout width. The widget measures the <div> it inserts
   * into the figure, which in a browser inherits the figure's own content width -- there is no
   * layout engine here to compute that, and the number itself is not what any assertion turns on. */
  getBoundingClientRect(): { width: number; height: number; left: number; top: number } {
    return { width: HOST_WIDTH, height: HOST_WIDTH, left: 0, top: 0 };
  }
  querySelector(selector: string): FakeElement | null {
    return this.descendants().find((c) => c.tagName === selector) ?? null;
  }
  descendants(): FakeElement[] {
    return this.children.flatMap((c) => [c, ...c.descendants()]);
  }
  /** Every element of `tagName` anywhere below this one, in document order. */
  all(tagName: string): FakeElement[] {
    return this.descendants().filter((c) => c.tagName === tagName);
  }
  /** All text this element and its descendants carry -- the widget's readout, as a reader sees it. */
  text(): string {
    return [this.textContent, ...this.descendants().map((c) => c.textContent)].join(" ");
  }
}

(globalThis as Record<string, unknown>).document = {
  createElement: (tag: string): FakeElement => new FakeElement(tag),
  createElementNS: (_ns: string, tag: string): FakeElement => new FakeElement(tag),
};
(globalThis as Record<string, unknown>).window = { addEventListener: (): void => {} };

const HOST_WIDTH = 800;
const BUNDLE_PATH = "../examples/method-comparison/frontier.json";
const bundle = JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as FrontierBundle;

/** The mount point the generator emits, in the shape `scripts/gen_site_pages.py::_frontier_figure`
 * emits it: a `<figure>` carrying the attributes, with the fallback `<img>` and a `<figcaption>`
 * inside it. The style/chart values are this test's own fixture -- what matters is that the widget
 * is given a complete set, and the Python side asserts the generator emits one
 * (tests/test_gen_site_pages.py::test_frontier_chart_config_covers_every_field_the_widget_requires).
 */
function mountPoint(): FakeElement {
  const figure = new FakeElement("figure");
  figure.children.push(new FakeElement("img"), new FakeElement("figcaption"));
  const methods = Object.fromEntries(Object.keys(bundle.methods)
    .map((key, i) => [key, { label: `Method ${i}`, colour: `#00${i}0${i}0` }]));
  figure.dataset["bundle"] = BUNDLE_PATH;
  figure.dataset["methods"] = JSON.stringify(methods);
  figure.dataset["targetDisplacement"] = String(bundle.matched_displacement);
  figure.dataset["targetPermeability"] = String(bundle.matched_permeability);
  figure.dataset["chart"] = JSON.stringify({
    xLabel: "displacement", yLabel: "permeability", lineWidth: 2.5, guideColour: "gray",
    guideDash: "6 4", tickTarget: 5, pad: 0.15, aspect: 4 / 3, sliderDivisions: 100,
    permeabilityMax: 1,
  });
  return figure;
}

/** Mounts the widget and waits for its fetch chain to settle. The widget's own `.catch` turns any
 * boot failure into caption text rather than an unhandled rejection, so this resolves either way --
 * which is exactly why the assertions below check what was DRAWN, never merely that nothing threw.
 */
async function mount(host: FakeElement): Promise<void> {
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<FrontierBundle> => Promise.resolve(bundle),
  });
  frontier(host as unknown as HTMLElement, localState);
  await new Promise((resolve) => setTimeout(resolve, 0));
}

test("the widget mounts and draws one curve per method in the bundle, with no NaN coordinate",
  async () => {
    const host = mountPoint();
    await mount(host);

    const caption = host.querySelector("figcaption")!;
    assert.ok(!caption.textContent.includes("failed to load"),
      `the widget reported a failure instead of mounting: ${caption.textContent}`);

    const svgs = host.all("svg");
    assert.equal(svgs.length, 1, "expected exactly one <svg> on the mount point");
    const polylines = svgs[0]!.all("polyline");
    assert.equal(polylines.length, Object.keys(bundle.methods).length,
      "one polyline per method, or a curve went missing");
    for (const p of polylines) {
      const points = p.getAttribute("points")!;
      assert.ok(points.length > 0, "an empty polyline draws nothing");
      // FINITE, not merely "not NaN": a single NaN in `points` makes the WHOLE series render as
      // nothing, silently -- and so does an Infinity, which is what a divide-by-zero in the
      // clipping arithmetic produces instead of a NaN. Both leave a chart that simply looks like it
      // has fewer methods than it does, with no error anywhere.
      for (const coord of points.split(/[ ,]/)) {
        assert.ok(Number.isFinite(Number(coord)),
          `non-finite coordinate "${coord}" in a polyline: ${points.slice(0, 120)}`);
      }
      assert.ok(Number(p.getAttribute("stroke-width")) > 0, "a zero-width stroke draws nothing");
    }

    // Both target guides, dashed, and the two axis titles: the chrome that makes the picture
    // readable at all. drawAxes emits gridlines as <line> too, so the guides are identified by
    // carrying the dash the mount point asked for.
    const dashed = svgs[0]!.all("line").filter((l) => l.getAttribute("stroke-dasharray") !== null);
    assert.equal(dashed.length, 2, "expected exactly two dashed target guides");
    const labels = svgs[0]!.all("text").map((t) => t.textContent);
    assert.ok(labels.includes("displacement") && labels.includes("permeability"), String(labels));
  });

test("every number the chart draws is also on the page as text", async () => {
  const host = mountPoint();
  await mount(host);
  const text = host.text();

  // The two booted targets, as the caption states them, and the verdict for every method.
  assert.match(text, /60\.0% permeability within 10\.0% displacement/);

  // And the ANSWER, against an independent oracle: a linear scan over the same committed arrays,
  // written the other way round from the widget's binary search. Without this the summary could
  // report any count at all -- "0 of 8" reads as plausibly as the truth to anyone who has not done
  // the arithmetic, which is exactly the kind of wrong number a chart makes invisible.
  const expected = Object.values(bundle.methods).filter((c) => {
    const i = c.permeability.findIndex((p) => p >= bundle.matched_permeability);
    return i !== -1 && c.displacement[i]! <= bundle.matched_displacement;
  }).length;
  assert.match(text, new RegExp(`${expected} of ${Object.keys(bundle.methods).length} methods reach`),
    `summary does not report ${expected} clearing methods`);
  assert.equal(host.all("li").length, Object.keys(bundle.methods).length,
    "one verdict per method, or the readout and the chart disagree about what is drawn");
  // Every verdict is a real sentence about a real method, never an empty <li>.
  for (const li of host.all("li")) {
    assert.match(li.textContent, /^Method \d+: .+\.$/, li.textContent);
  }
  // Both range inputs exist -- the keyboard path to the same two targets the pointer drags.
  assert.equal(host.all("input").length, 2);
  // One legend button per method, each reporting whether it is the isolated one.
  const buttons = host.all("button");
  assert.equal(buttons.length, Object.keys(bundle.methods).length);
  for (const b of buttons) assert.equal(b.getAttribute("aria-pressed"), "false");
});

test("the fallback image survives a boot failure and is removed only on success", async () => {
  const ok = mountPoint();
  await mount(ok);
  const img = ok.querySelector("img")!;
  const svg = ok.all("svg")[0]!;
  assert.notEqual(img.removedAt, null,
    "the fallback image is still there after a successful mount -- the reader sees both");
  // ORDER, not merely both: removing the image before the chart exists would leave a gap on the
  // page for however long the drawing takes, and -- if the drawing then threw -- for good, under an
  // error message that says "the static image above still applies".
  assert.ok(img.removedAt! > svg.createdAt,
    `fallback removed at ${img.removedAt} but the chart was only built at ${svg.createdAt}`);

  // A mount point missing one target: the page and the widget disagree about the standards the
  // caption states, so the widget must fail loudly AND leave the static image in place, because
  // that is the image mount.ts's error text points the reader at.
  const broken = mountPoint();
  delete broken.dataset["targetPermeability"];
  await mount(broken);
  assert.equal(broken.querySelector("img")!.removedAt, null);
  assert.match(broken.querySelector("figcaption")!.textContent,
    /Frontier failed to load: .*data-target-permeability/);
});
