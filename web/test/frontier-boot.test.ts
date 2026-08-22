import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { frontier, FRONTIER_URL, type FrontierState } from "../src/widgets/frontier.js";
import type { FrontierBundle } from "../src/frontier.js";
// Only the two URL seams -- never the harness's own `FakeElement`, which this file deliberately
// does not share (its widget draws SVG, so its fake element carries `createElementNS` and `all()`
// rather than a recording 2D context).
import { fakeLocation, writeNow } from "./harness.js";

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
/** How many fake elements have been created so far -- a render rebuilds the whole SVG, so this
 * counter moving is exactly "a render happened". */
const created = (): number => clock;

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

  /** UPPERCASE, as a real element reports it -- `dom/fallback.ts` tests `tagName === "A"` to decide
   * whether the thing wrapping the fallback image is the glightbox anchor, so a lowercase fake would
   * make that branch untestable here while looking tested. Every selector below is upper-cased at
   * the point of comparison instead, so call sites still read `querySelector("img")`. */
  constructor(tagName: string) { this.tagName = tagName.toUpperCase(); }
  /** Set by `append`/`insertBefore`, read by `remove` and by `dom/fallback.ts`'s `parentElement`. */
  parent: FakeElement | null = null;
  setAttribute(name: string, value: string): void { this.attrs.set(name, value); }
  getAttribute(name: string): string | null { return this.attrs.get(name) ?? null; }
  /** A plain string appended to a real element becomes a text NODE, so the fake keeps it too --
   * otherwise the widget's own prose (its control labels) is invisible to `text()` below, and a
   * test asserting what the page says could only ever see the parts built as elements. Recorded
   * under a tag name no real selector can match, so element counts and querySelector are
   * unaffected. */
  append(...nodes: (FakeElement | string)[]): void {
    for (const n of nodes) {
      if (typeof n === "string") {
        const text = new FakeElement("#TEXT");
        text.textContent = n;
        text.parent = this;
        this.children.push(text);
      } else {
        n.parent = this;
        this.children.push(n);
      }
    }
  }
  insertBefore(node: FakeElement, _ref: FakeElement | null): void {
    node.parent = this;
    this.children.push(node);
  }
  replaceChildren(...nodes: FakeElement[]): void { this.children = [...nodes]; }
  /** DETACHES, as a real `remove()` does -- not merely a timestamp. `dom/fallback.ts` removes the
   * <img> and then asks whether its anchor has any element children left; a fake that only recorded
   * the removal would leave that count at 1 forever and the anchor branch would never be reached. */
  remove(): void {
    this.removedAt = ++clock;
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }
  /** Real `parentElement` is null once removed, which is exactly what `remove()` above leaves. */
  get parentElement(): FakeElement | null { return this.parent; }
  addEventListener(name: string, fn?: (ev: unknown) => void): void {
    this.listeners.push(name);
    if (fn) (this.handlers[name] ??= []).push(fn);
  }
  /** Handlers by event name, so a test can dispatch a real pointer sequence rather than assert that
   * a listener was merely registered. */
  readonly handlers: Record<string, ((ev: unknown) => void)[]> = {};
  dispatch(name: string, ev: unknown): void {
    for (const fn of this.handlers[name] ?? []) fn(ev);
  }
  setPointerCapture(): void {}
  releasePointerCapture(): void {}
  /** Every fake element reports the same layout width. It is what `FakeResizeObserver.observe`
   * delivers as the initial observation of the <div> the widget inserts into the figure, which in a
   * browser inherits the figure's own content width -- there is no layout engine here to compute
   * that, and the number itself is not what any assertion turns on. */
  getBoundingClientRect(): { width: number; height: number; left: number; top: number } {
    return { width: HOST_WIDTH, height: HOST_WIDTH, left: 0, top: 0 };
  }
  querySelector(selector: string): FakeElement | null {
    return this.descendants().find((c) => c.tagName === selector.toUpperCase()) ?? null;
  }
  descendants(): FakeElement[] {
    return this.children.flatMap((c) => [c, ...c.descendants()]);
  }
  /** Every element of `tagName` anywhere below this one, in document order. */
  all(tagName: string): FakeElement[] {
    return this.descendants().filter((c) => c.tagName === tagName.toUpperCase());
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
/** The observer the widget now lays itself out from (src/dom/resize.ts). A real one delivers an
 * initial observation once `observe()` has been called, so the fake does too -- but DEFERRED to a
 * microtask, never synchronously, because a real ResizeObserver delivers from the browser's own
 * dispatch after the caller has returned. Firing synchronously would put the first draw inside the
 * mount's `fetch().then(boot).catch(showWidgetError)` chain, where that chain's `.catch` absorbs any
 * throw -- making `runOrReport` redundant in the fake and in the fake only. A microtask scheduled
 * from within a `.then` handler runs after the handler returns, so it is outside the chain just as
 * the browser's dispatch is, while still landing before the `setTimeout(0)` `mount()` awaits, which
 * is what keeps "the chart is drawn by the time mount() resolves" true for every test below. `fire`
 * then lets a test do the thing this whole task exists for: change the CONTAINER's width with the
 * window untouched.
 *
 * Installed on globalThis at module top level, exactly like the `document` stub above and for the
 * same reason: neither frontier.ts nor resize.ts touches either global in its module body, only
 * inside functions the tests call. */
class FakeResizeObserver {
  static live: FakeResizeObserver[] = [];
  disconnected = false;
  constructor(private readonly cb:
              (entries: { contentRect: { width: number; height: number } }[]) => void) {
    FakeResizeObserver.live.push(this);
  }
  observe(el: FakeElement): void {
    const { width, height } = el.getBoundingClientRect();
    queueMicrotask(() => this.fire(width, height));
  }
  disconnect(): void { this.disconnected = true; }
  fire(width: number, height: number): void { this.cb([{ contentRect: { width, height } }]); }
}
(globalThis as Record<string, unknown>).ResizeObserver = FakeResizeObserver;
// NOTE: there is deliberately no `window` stub here any more. It existed only to absorb the
// `window.addEventListener("resize", ...)` this widget used to register; nothing in frontier.ts or
// anything it imports reads `window` now, so a stub would be scenery standing in for a dependency
// that no longer exists.

const HOST_WIDTH = 800;
const BUNDLE_PATH = "../examples/method-comparison/frontier.json";
const bundle = JSON.parse(readFileSync(BUNDLE_PATH, "utf8")) as FrontierBundle;

/** The mount point the generator emits, in the shape `scripts/gen_site_pages.py::_frontier_figure`
 * emits it: a `<figure>` carrying four scalar attributes, with the fallback `<img>` and a
 * `<figcaption>` inside it. Since fix round 1 the labels, colours and every drawn dimension come
 * from the BUNDLE instead of from `data-*` JSON, so this fixture no longer supplies them -- which
 * means the assertions below run against the real baked chart block and the real legend names, not
 * a fixture's idea of them. */
function mountPoint(): FakeElement {
  const figure = new FakeElement("figure");
  // The <img> sits inside an <a class="glightbox">, because that is what the SERVED page carries:
  // mkdocs-glightbox rewrites every non-skip_lightbox figure image into a lightbox link at build
  // time. A fixture with a bare <img> would let a widget that removes only the image pass, while the
  // live page kept an empty, focusable, screen-reader-announced link where the picture was.
  const anchor = new FakeElement("a");
  anchor.append(new FakeElement("img"));
  figure.append(anchor, new FakeElement("figcaption"));
  figure.dataset["bundle"] = BUNDLE_PATH;
  figure.dataset["targetDisplacement"] = String(bundle.matched_displacement);
  figure.dataset["targetPermeability"] = String(bundle.matched_permeability);
  figure.dataset["aspect"] = String(4 / 3);
  return figure;
}

/** Mounts the widget and waits for its fetch chain to settle. The widget's own `.catch` turns any
 * boot failure into caption text rather than an unhandled rejection, so this resolves either way --
 * which is exactly why the assertions below check what was DRAWN, never merely that nothing threw.
 *
 * The state store is the PRODUCTION one (`urlStore` over a `fakeLocation`), never `localState`:
 * `search` defaults to "" -- an empty URL, so nothing is decoded and nothing is written, and a
 * caller that passes no search gets the widget's own initial state exactly as `localState` gave
 * it. (The store CLAIMS every key of the codec regardless of what the URL carries; what an
 * empty query skips is the decode, not the claim.) The URL tests below then get the real
 * decode path rather than a second, test-only one.
 *
 * `store` is nullable because a REFUSED bundle is a case this file tests: every one of `boot`'s
 * bundle and `data-*` checks runs before it calls `makeState`, so a payload it rejects never asks
 * for a store at all. Asserting non-null in here would turn "the widget correctly refused a broken
 * bundle" into a helper failure. */
async function mount(host: FakeElement, payload: unknown = bundle, search = ""):
    Promise<{ store: StateSource<FrontierState> | null;
              loc: ReturnType<typeof fakeLocation> }> {
  (globalThis as Record<string, unknown>).fetch = (): Promise<unknown> => Promise.resolve({
    ok: true,
    status: 200,
    statusText: "OK",
    json: (): Promise<unknown> => Promise.resolve(payload),
  });
  const loc = fakeLocation(search);
  const urls = urlStore(loc, writeNow);
  let bound: StateSource<FrontierState> | null = null;
  frontier(host as unknown as HTMLElement, (initial) => {
    bound = urls.bind(FRONTIER_URL, initial);
    return bound;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  return { store: bound, loc };
}

test("the widget mounts and draws one curve per method in the bundle, with no NaN coordinate",
  async () => {
    const host = mountPoint();
    await mount(host);

    const caption = host.querySelector("figcaption")!;
    assert.ok(!caption.textContent.includes("could not load interactively"),
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
      // The width the fallback PNG's own curves were drawn with (reblock.emit's FRONTIER_LW,
      // baked into the bundle), not a number this widget chose.
      assert.equal(Number(p.getAttribute("stroke-width")), bundle.chart.line_width);
    }

    // M1: one visible dot per MEASURED sample, as the fallback PNG draws them. Without these the
    // hover readout snaps to prefixes the reader cannot see, and a curve clipped to one sample draws
    // nothing whatsoever. The count is the number of samples inside the displayed window, summed
    // over the eight curves -- and never one more, which is what would happen if the interpolated
    // clip point (a drawing artifact, not a measurement) got a dot too.
    const measured = Object.values(bundle.methods)
      .map((c) => c.displacement.filter((d) => d <= bundle.frontier_xmax).length)
      .reduce((a, b) => a + b, 0);
    const dots = svgs[0]!.all("circle");
    assert.equal(dots.length, measured, "one marker per measured sample inside the window");
    for (const d of dots) {
      assert.equal(Number(d.getAttribute("r")), bundle.chart.marker_radius);
      for (const attr of ["cx", "cy"]) {
        assert.ok(Number.isFinite(Number(d.getAttribute(attr))),
          `non-finite marker ${attr}: ${d.getAttribute(attr)}`);
      }
    }

    // Both target guides, dashed, and the two axis titles: the chrome that makes the picture
    // readable at all. drawAxes emits gridlines as <line> too, so the guides are identified by
    // carrying the dash the mount point asked for.
    const dashed = svgs[0]!.all("line").filter((l) => l.getAttribute("stroke-dasharray") !== null);
    assert.equal(dashed.length, 2, "expected exactly two dashed target guides");
    for (const g of dashed) {
      assert.equal(g.getAttribute("stroke-dasharray"), bundle.chart.guide_dash);
      assert.equal(Number(g.getAttribute("stroke-width")), bundle.chart.guide_width);
      assert.equal(g.getAttribute("stroke"), bundle.chart.guide_colour);
    }
    // N4 (final review): the gridlines' opacity is a bundle-sourced DRAWN value, and it was the only
    // one of them not pinned here -- line_width, marker_radius and every guide_* above are. Hard-coding
    // it to full ink in the widget left all 45 node tests green, which is the same hole as every other
    // "the chart still drew something, so nothing failed" defect on this branch. Gridlines are the
    // <line>s WITHOUT the guides' dash (see the comment above), and there must be some, or the
    // assertion would hold vacuously over an empty list.
    const gridlines = svgs[0]!.all("line").filter((l) => l.getAttribute("stroke-dasharray") === null);
    assert.ok(gridlines.length > 0, "no gridlines emitted, so their opacity is untested");
    for (const g of gridlines) {
      assert.equal(Number(g.getAttribute("stroke-opacity")), bundle.chart.grid_opacity,
        "gridline opacity must come from the bundle, not from a TypeScript literal");
    }
    // M6: no `role="img"`. That plus an aria-label makes the subtree presentational, hiding the
    // <text> tick labels and axis titles from assistive tech -- and those being real, reachable text
    // is half the stated reason this widget draws SVG instead of canvas.
    assert.equal(svgs[0]!.getAttribute("role"), null,
      "role on the <svg> makes its axis text unreachable to a screen reader");

    const labels = svgs[0]!.all("text").map((t) => t.textContent);
    assert.ok(labels.includes(bundle.chart.x_label) && labels.includes(bundle.chart.y_label),
      String(labels));

    // UNITS (fix round 1): the figure this widget replaces puts a PercentFormatter(xmax=1) on both
    // axes, so JS-off reads "60%" -- the widget used to draw the same tick as "0.6", the two
    // readers of one page seeing different charts. Both axis ranges are checked, since the x axis
    // is the bundle's frontier_xmax and the y axis is its permeability_max.
    assert.ok(labels.includes("0%"), `no percent-formatted zero tick: ${String(labels)}`);
    assert.ok(labels.includes(`${bundle.frontier_xmax * 100}%`),
      `x axis does not end on a percent tick: ${String(labels)}`);
    assert.ok(labels.includes(`${bundle.chart.permeability_max * 100}%`),
      `y axis does not end on a percent tick: ${String(labels)}`);
    for (const t of labels) {
      // No bare fraction may survive anywhere in the chrome: a tick reading "0.6" is the exact
      // contradiction this guard exists for.
      assert.ok(!/^0\.\d+$/.test(t), `bare fraction "${t}" among the axis labels`);
    }
  });

test("every number the chart draws is also on the page as text", async () => {
  const host = mountPoint();
  await mount(host);
  const text = host.text();

  // The two booted targets, in the whole-percent form the caption and the fallback PNG's legend
  // both state them in -- read off the bundle, so this cannot drift into asserting a typed number.
  assert.match(text, new RegExp(`${bundle.matched_permeability * 100}% permeability within `
    + `${bundle.matched_displacement * 100}% displacement`));

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
  // Every verdict is a real sentence naming the method by the SAME legend name the fallback PNG
  // uses (`friendly_method_name`, baked into the bundle) -- "Frontage (street-priced)", never the
  // raw key `greedy_arterial_access_displacement`, which no widget could reconstruct it from.
  const verdicts = host.all("li").map((li) => li.textContent);
  for (const [key, curve] of Object.entries(bundle.methods)) {
    assert.ok(verdicts.some((v) => v.startsWith(`${curve.label}: `)),
      `no verdict for ${key} under its baked label "${curve.label}"`);
    assert.ok(!verdicts.some((v) => v.includes(key)), `verdict names the raw key ${key}`);
  }
  for (const v of verdicts) assert.match(v, /\.$/, v);

  // Both guide labels name their target in whole percentage points -- `{:.0%}`, the format the
  // fallback PNG's own legend states the same two standards in.
  // `\\s+`, not a single space: text() joins each node's text with a space, so the label's own
  // trailing space and the join add up. Whitespace between a label and its value is not the subject.
  assert.match(text,
    new RegExp(`Most displacement allowed:\\s+${bundle.matched_displacement * 100}%`));
  assert.match(text,
    new RegExp(`Least permeability required:\\s+${bundle.matched_permeability * 100}%`));
  // Both range inputs exist -- the keyboard path to the same two targets the pointer drags.
  assert.equal(host.all("input").length, 2);
  // One legend button per method, each reporting whether it is the isolated one.
  const buttons = host.all("button");
  assert.equal(buttons.length, Object.keys(bundle.methods).length);
  for (const b of buttons) assert.equal(b.getAttribute("aria-pressed"), "false");
});

test("the fallback image survives a boot failure and is removed only on success", async () => {
  const ok = mountPoint();
  // Captured BEFORE the mount: `remove()` detaches, so after a successful mount there is no <img>
  // left in the tree to query for -- which is the point.
  const img = ok.querySelector("img")!;
  const anchor = ok.querySelector("a")!;
  await mount(ok);
  const svg = ok.all("svg")[0]!;
  assert.notEqual(img.removedAt, null,
    "the fallback image is still there after a successful mount -- the reader sees both");
  // ...and so is the glightbox <a> that wrapped it. An anchor emptied of its image is invisible but
  // still focusable, and a screen reader announces it as a link with no text -- inside the figure
  // whose whole accessibility argument is that its contents are reachable text. PermGraph shipped
  // exactly that to the live site.
  assert.notEqual(anchor.removedAt, null,
    "the glightbox anchor outlived its image: an empty, focusable, unlabelled link");
  assert.equal(ok.all("a").length, 0, "an <a> survives inside the mounted figure");
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
  assert.equal(broken.querySelector("a")!.removedAt, null,
    "the anchor went even though the image it wraps -- the picture the error text points at -- stayed");
  assert.match(broken.querySelector("figcaption")!.textContent,
    /Frontier could not load interactively .*data-target-permeability/);
  // The sentence that only one of the three former copies carried (final review, M7): the reader is
  // told the picture above still stands, which is true exactly because the fallback survives a boot
  // failure -- asserted on the line above.
  assert.match(broken.querySelector("figcaption")!.textContent,
    /The static image above still applies\./);
});

test("a bundle missing a field the whole chart is scaled by fails LOUDLY and keeps the image",
  async () => {
    // Review finding I1, as its own regression guard. `frontier_xmax` scales the x axis and bounds
    // the clip; read unvalidated it made every one of the eight polylines all-NaN -- which renders
    // nothing -- while the widget reported success in the caption and removed the fallback <img>,
    // leaving a blank frame with no message anywhere. The other tests here would catch the NaN; this
    // one catches the part that made it lethal, which is that nothing was reported and the image went
    // anyway. Same shape for `block_id`, which the readout quotes.
    for (const field of ["frontier_xmax", "block_id"] as const) {
      const { [field]: _dropped, ...stale } = bundle;
      const host = mountPoint();
      await mount(host, stale);
      assert.equal(host.querySelector("img")!.removedAt, null,
        `${field} missing: the fallback image was removed anyway`);
      assert.match(host.querySelector("figcaption")!.textContent,
        new RegExp(`Frontier could not load interactively .*${field}`),
        `${field} missing: no visible failure on the page`);
      assert.equal(host.all("polyline").length, 0,
        `${field} missing: a curve was drawn from an unvalidated bundle`);
    }
  });

test("a zero aspect ratio is refused instead of scaling the chart to nothing", async () => {
  // M1: `measure` divides the box width by this, so 0 gives an Infinite height, an Infinite scaleY
  // and a NaN in every polyline -- which renders nothing, reports nothing, and (before I1) removed
  // the fallback image anyway. The last number that scales the whole chart, and the one field I1's
  // sweep did not reach.
  const host = mountPoint();
  host.dataset["aspect"] = "0";
  await mount(host);
  assert.match(host.querySelector("figcaption")!.textContent,
    /Frontier could not load interactively .*data-aspect must be positive/);
  assert.equal(host.querySelector("img")!.removedAt, null);
  assert.equal(host.all("polyline").length, 0);
});

test("a drag that does not move the guide past a step boundary does not re-render", async () => {
  // M2. A render rebuilds the entire SVG -- 8 polylines, 548 markers, 11 gridlines -- so a
  // pointermove that leaves the snapped target where it was used to rebuild all of it to draw the
  // identical picture. Driven through the widget's own pointer handlers, and observed by counting
  // element creations, which is what "a render happened" means in a fake DOM.
  //
  // The box is HOST_WIDTH x HOST_WIDTH/aspect = 800x600 at pad 0.15, so the x axis [0, 0.4] maps with
  // scaleX 1400 and tx 120: the 10% guide sits at screen x = 260, one screen pixel is 0.0007 of the
  // axis, and the step is 0.01. So +1 px cannot cross a step boundary and +40 px must.
  const host = mountPoint();
  await mount(host);
  const chart = host.children.find((c) => c.tagName === "DIV")!;
  assert.ok(host.all("circle").length > 0, "no markers drawn, so a re-render would be unobservable");

  chart.dispatch("pointerdown", { clientX: 260, clientY: 300, pointerId: 1, preventDefault: () => {} });
  const afterGrab = created();

  chart.dispatch("pointermove", { clientX: 261, clientY: 300, pointerId: 1 });
  assert.equal(created(), afterGrab,
    "a pointermove inside one step rebuilt the whole chart to draw the same picture");

  chart.dispatch("pointermove", { clientX: 300, clientY: 300, pointerId: 1 });
  assert.ok(created() > afterGrab,
    "a pointermove that DOES cross a step boundary must still re-render -- otherwise the early "
    + "return is not an optimisation, it is a broken drag");
  assert.match(host.text(), /Most displacement allowed:\s+13%/,
    "the guide did not follow the drag to its new step");
});

test("a boot target outside its own axis is refused rather than drawn off-chart", async () => {
  // M2: finite but off-axis. `drawGuide` would happily place the line outside the plot rect, where
  // the reader cannot see it, while the readout kept answering truthfully about a guide that is not
  // on the chart -- a picture and a caption that disagree, which is what this whole round was about.
  const host = mountPoint();
  host.dataset["targetPermeability"] = String(bundle.chart.permeability_max + 0.5);
  await mount(host);
  assert.match(host.querySelector("figcaption")!.textContent,
    /data-target-permeability \(1\.5\) is outside its axis \[0, 1\]/);
  assert.equal(host.querySelector("img")!.removedAt, null);
});

test("a container narrowing with the window untouched re-lays the chart out at the new width",
  async () => {
    // The defect this task exists for. `cv.style.width`/`getBoundingClientRect` made the box right
    // AT MOUNT, and a `window` resize listener made it right again when the WINDOW moved -- neither
    // covers a container narrowing on its own (Material's nav drawer at a breakpoint, a <details>
    // opening, a tab panel, print), which is where this absolute-pixel SVG overflowed its figure.
    const host = mountPoint();
    await mount(host);
    const before = host.all("svg")[0]!;
    assert.equal(before.getAttribute("width"), String(HOST_WIDTH));

    FakeResizeObserver.live.at(-1)!.fire(320, 200);

    const after = host.all("svg")[0]!;
    assert.equal(host.all("svg").length, 1, "the old chart was left on the page beside the new one");
    assert.equal(after.getAttribute("width"), "320", "the chart kept the width it mounted at");
    // The INLINE style, not the presentation attribute: Material ships
    // `.md-typeset svg{height:auto;max-width:100%}`, which beats a zero-specificity attribute.
    assert.equal(after.style["width"], "320px");
    assert.equal(after.style["height"], `${320 / (4 / 3)}px`);
    // Re-laid out, not scaled: the labels are still real text at their designed size, which is the
    // whole reason this is a re-render and not a viewBox.
    assert.ok(after.all("text").length >= 4, "the narrowed chart lost its axis chrome");
    assert.ok(!host.querySelector("figcaption")!.textContent.includes("could not load"),
      host.querySelector("figcaption")!.textContent);
  });

test("a resize that does not change the width does not rebuild the chart", async () => {
  // Our own render changes the box being observed: `chartHost` is 0 px tall until the SVG lands in
  // it, so the first draw fires a second callback carrying the SAME width. Without the guard that
  // rebuilds 8 polylines and 548 markers to draw the identical picture, on every resize.
  const host = mountPoint();
  await mount(host);
  const obs = FakeResizeObserver.live.at(-1)!;
  const afterMount = created();
  obs.fire(HOST_WIDTH, 613);
  assert.equal(created(), afterMount,
    "a height-only resize rebuilt the whole chart -- and each rebuild changes the height again");
  obs.fire(321, 613);
  assert.ok(created() > afterMount, "a real width change must still re-render");
});

test("a throw while re-laying out reaches the caption instead of vanishing into the console",
  async () => {
    // The failure path this task introduced. Everything up to and including the first draw used to
    // run inside the mount's `fetch().then().catch(showWidgetError)` chain; drawing now happens in a
    // ResizeObserver callback, which is OUTSIDE it -- a throw there is an unhandled rejection and
    // nothing else, and by then the fallback <img> is already gone, so the reader is left with a
    // blank figure and no message. That is this branch's signature defect, so it gets its own route
    // back to the page (`runOrReport`, dom/error.ts).
    const host = mountPoint();
    await mount(host);
    const obs = FakeResizeObserver.live.at(-1)!;
    const doc = (globalThis as Record<string, unknown>)["document"] as
      { createElementNS: (ns: string, tag: string) => FakeElement };
    const real = doc.createElementNS;
    doc.createElementNS = (): FakeElement => { throw new Error("boom while re-laying out"); };
    try {
      obs.fire(320, 200);
    } finally {
      doc.createElementNS = real;
    }
    assert.match(host.querySelector("figcaption")!.textContent,
      /Frontier could not load interactively .*boom while re-laying out/);
    assert.match(host.querySelector("figcaption")!.textContent,
      /The static image above still applies\./);
  });

test("an unknown ?method= draws EVERY curve, not an empty chart", async () => {
  // `?method=` names a curve in THIS bundle, which no codec can check. An unknown one is not inert:
  // the draw loop skips every key that is not the isolated one, so an isolated name that matches
  // nothing filters out all of them and the reader gets axes with no data on them.
  const host = mountPoint();
  const { store } = await mount(host, bundle, "method=not_a_method");
  assert.ok(store !== null, "the widget never asked for a state store");
  assert.equal(store.get().isolated, null);
  assert.equal(host.all("polyline").length, Object.keys(bundle.methods).length,
    "isolating a method that does not exist filtered out all of them");
});

test("a prototype key is not a method name", async () => {
  // `"toString" in b.methods` is true for every object, so an `in`-based membership test accepts a
  // prototype key as a method name -- and the chart then goes empty exactly as above.
  const host = mountPoint();
  const { store } = await mount(host, bundle, "method=toString");
  assert.ok(store !== null, "the widget never asked for a state store");
  assert.equal(store.get().isolated, null);
  assert.equal(host.all("polyline").length, Object.keys(bundle.methods).length,
    "a prototype key was accepted as a method name and the chart went empty");
});
