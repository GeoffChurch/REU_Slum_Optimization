/** Draw a road on the site's spine block and get its REAL permeability, solved in the browser.
 *
 * The only widget on this site that computes a number instead of looking one up, and the only one
 * that boots a Python runtime to do it (design §4). Everything about it follows from that: the
 * runtime is injected rather than constructed here (`drawRoadWidget` below), booting is an opt-in
 * the reader presses and the control says what pressing it downloads, and the picture is drawn
 * from the two arrays Pyodide hands back rather than from a baked prefix table.
 */
import type { AuthoringBlock } from "../authoring.js";
import { requireAttr } from "../dom/attrs.js";
import { runOrReport, showWidgetError } from "../dom/error.js";
import { removeFallbackImage } from "../dom/fallback.js";
import { observeSize } from "../dom/resize.js";
// Type-only: erased at compile time, so this module has NO runtime import of mount.js. A runtime
// one would recreate the import cycle that once made the whole bundle throw during module
// evaluation while the page still looked fine (see mount.ts's registration comment). This file
// must never import `register`.
import type { Widget } from "../mount.js";
import { pyodideRuntime, type PyResult, type PyRuntime } from "../py/runtime.js";
import { draw, sizeCanvas, type Drawable } from "../render/canvas.js";
import { GRAPH_ENCODING, GRAPH_ENCODING_BLOCK_ID } from "../render/graph-encoding.generated.js";
import type { StateFactory, StateSource } from "../state.js";
import { polylineParam, type UrlCodec } from "../url/param.js";
import {
  fitBbox, nearest, toScreen, toWorld, type Bbox, type View,
} from "../view/transform.js";

/** The name every failure of this widget is reported under -- see region-grow.ts's own `LABEL`
 * for why one constant rather than a string repeated at each call site. */
const LABEL = "DrawRoad";

/** The road the reader drew, in the bundle's own projected metres (EPSG:32734 on this block), as
 * a polyline of at least two points -- or empty, which is "no road drawn yet" and is what the
 * widget boots at.
 *
 * A polyline, not `field.ts`'s two fixed two-point `Road`s: an arbitrary road is the entire point
 * of this widget, and it is why no baked prefix table can answer the question it asks. */
export interface DrawRoadState {
  road: [number, number][];
}

export const DRAW_ROAD_URL: UrlCodec<DrawRoadState> = {
  // One key, at `roadsParam`'s own 0.1 m (url/param.ts's `COORD_DP`). `polylineParam` refuses a
  // road it cannot spell back -- odd coordinate count, non-finite value, fewer than two points --
  // and the store then drops `?road=` and falls back to the initial, so a hand-edited URL that
  // cannot be solved never reaches the runtime.
  road: polylineParam("road"),
};

/** What the boot control tells the reader it will download.
 *
 * The figure is the parent design's (`2026-08-13-site-redesign-design.md` §2: "The ~25-35 MB is
 * paid by the reader who opted in, never by a visitor reading prose"), and it is the distribution
 * plus packages, not something this file measured. It is on the button because this widget lands
 * on a prose page a reader may reach with no intention of computing anything (design §3), so the
 * control has to read as an explicit opt-in rather than as decoration. */
const BOOT_LABEL = "Load the Python runtime";
const BOOT_COST = "~25-35 MB download";
const BOOT_IDLE = `${BOOT_LABEL} (${BOOT_COST})`;
const BOOT_LOADING = "Loading the Python runtime...";
const BOOT_READY = "Python runtime loaded";
const BOOT_FAILED = "The Python runtime did not load";

/** The three things the readout says when there is no answer to report.
 *
 * `NEEDS_TWO_POINTS` is a REFUSAL and nothing else -- it appears only when a road that is not a
 * road was settled, never as an idle prompt, so seeing it is evidence that a settle was refused.
 * `web/src/py/solve.py` refuses the same arity on its own side ("a road needs at least two
 * points") for callers that skip this one. */
const NEEDS_TWO_POINTS =
  "That is not a road yet: it needs at least two points. Click another point, then press Enter or "
  + "double-click to solve it.";

const DRAW_A_ROAD =
  "Click points on the block to draw a road, then press Enter or double-click to solve it.";

/** Quotes `BOOT_LABEL` rather than repeating it: the readout points at a button, and a hand-typed
 * copy of its words would go on naming a button that no longer says that. */
const NEEDS_RUNTIME = `Press “${BOOT_LABEL}” above to solve a road you have drawn.`;

/** Builds a `PyRuntime` for a fetched bundle. Injected into `drawRoadWidget` rather than reached
 * for, exactly as `StateFactory` is (state.ts): the widget cannot tell whether it holds real
 * Pyodide or a hand-written fake, which is what lets `web/test/draw-road-boot.test.ts` drive the
 * whole interaction with no network and no download at all. */
export type PyRuntimeFactory = (bundle: AuthoringBlock, wheelUrl: string) => PyRuntime;

/** At least two points, and that is the whole rule.
 *
 * One predicate with two callers -- `settle` (may this road be published to the URL?) and
 * `pumpSolve` (may this road be handed to the runtime?) -- so the arity a road must have is
 * written once rather than spelled twice and drifting. */
function isSolvable(road: [number, number][]): boolean {
  return road.length >= 2;
}

/** Consecutive duplicate vertices dropped.
 *
 * A double-click settles the road AND delivers its own `pointerdown` on the spot the click before
 * it landed on, so a double-click-settled road ends with the same point twice -- a zero-length
 * final segment, which design §6 lists among the degeneracies to refuse. Dropping the repeat
 * settles the road the reader actually drew instead of refusing it. Exact equality, not a
 * tolerance: both coordinates come out of `toWorld` on the same integer offsets, so the repeat is
 * bit-identical rather than merely close. */
function withoutRepeats(road: [number, number][]): [number, number][] {
  return road.filter((p, i) => {
    const prev = road[i - 1];
    return prev === undefined || prev[0] !== p[0] || prev[1] !== p[1];
  });
}

function bboxOf(ring: [number, number][]): Bbox {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of ring) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

/** The widget, over an injected runtime factory. `drawRoad` just below is this applied to the real
 * one; the tests apply it to a fake. */
export function drawRoadWidget(makeRuntime: PyRuntimeFactory): Widget<DrawRoadState> {
  return (host, makeState) => {
    // Not `host.dataset.bundle!`: a missing attribute then reaches `fetch(undefined)` and surfaces
    // as "fetch undefined failed: 404", which sends the reader looking for a missing FILE rather
    // than the missing ATTRIBUTE that is actually wrong (perm-graph.ts's own reasoning).
    const src = requireAttr(host.dataset.bundle, "data-bundle", LABEL);
    // The `reblock` wheel `micropip` installs at boot. Read here, at mount, though nothing fetches
    // it until the reader presses the button: a page that forgot the attribute should say so where
    // every other mount failure is said, not many seconds into a download.
    const wheelUrl = requireAttr(host.dataset.wheel, "data-wheel", LABEL);
    void fetch(src)
      .then((r) => {
        if (!r.ok) throw new Error(`fetch ${src} failed: ${r.status} ${r.statusText}`);
        return r.json() as Promise<AuthoringBlock>;
      })
      .then((b) => boot(host, makeState, b, makeRuntime(b, wheelUrl)))
      .catch((err: unknown) => showWidgetError(host, LABEL, err));
  };
}

export const drawRoad: Widget<DrawRoadState> = drawRoadWidget(pyodideRuntime);

function boot(host: HTMLElement, makeState: StateFactory<DrawRoadState>, ab: AuthoringBlock,
              runtime: PyRuntime): void {
  // `GRAPH_ENCODING` is examples/perm-graph/bundle.json's own encoding, read at build time
  // (web/scripts/gen-graph-encoding.mjs), and half of it -- `width_norm` -- is a property of that
  // bundle's MESH, not a style choice: it is the scale every edge width on the picture is drawn
  // against. Both bundles are baked from `scripts/_example_block.py`'s one pinned block, but by
  // two scripts run by two separate commands, so re-baking and committing one without the other
  // is the way they come apart -- and nothing in either file's type would notice.
  if (ab.block_id !== GRAPH_ENCODING_BLOCK_ID) {
    throw new Error(`${LABEL}: the drawing scale was baked for block ${GRAPH_ENCODING_BLOCK_ID}, `
      + `but this bundle is block ${ab.block_id}`);
  }
  // EVERY ring, passed through. This used to take `ab.boundary[0]` and throw on anything else,
  // mirroring `polygon_ring`'s own refusal to drop a hole silently -- correct while `draw` stroked
  // a single ring, which it no longer does (`Drawable.boundary` is a ring list as of 2026-09-20,
  // and every render bundle now bakes one). The refusal was not hypothetical: the block this
  // widget was repinned to, `ZAF.9.3.1_1_5810`, HAS an interior ring, so it would have thrown at
  // boot. What replaces it is not a quieter rule but a stricter one -- the hole is now drawn.
  //
  // An EMPTY list is still refused. A block with no exterior is not a block, and the flat-list
  // rendering below would silently draw nothing at all for it.
  if (ab.boundary.length === 0) {
    throw new Error(`${LABEL}: the block boundary has no rings`);
  }
  const boundary = ab.boundary;

  const caption = host.querySelector("figcaption");

  const cv = document.createElement("canvas");
  // Inline styles, never presentation attributes -- Material's `.md-typeset svg{height:auto;
  // max-width:100%}` beats a presentation attribute (region-grow.ts's own reasoning).
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  // Drawing is a pointer gesture, so without this a touch press scrolls the page and the browser
  // CANCELS the pointer stream -- the same line perm-graph.ts carries for its pan.
  cv.style.touchAction = "none";
  // Enter settles the road and Escape clears it, and a canvas is not focusable by default, so
  // neither key can reach it without this. It is also what lets a keyboard user get to the
  // picture at all.
  cv.tabIndex = 0;

  const controls = document.createElement("div");
  const bootButton = document.createElement("button");
  bootButton.textContent = BOOT_IDLE;

  const readout = document.createElement("p");
  // The line that changes on every edit -- the permeability of the road just solved, the refusal
  // of one that is not a road yet, the failure of a solve. A canvas carries no accessible text at
  // all, so without this a screen-reader user drawing a road hears nothing about the answer
  // (screen-map.ts's own reasoning for its readout). `polite`, not `assertive`: a fast sequence of
  // edits must not interrupt itself.
  readout.setAttribute("aria-live", "polite");
  readout.textContent = NEEDS_RUNTIME;

  controls.append(bootButton, readout);

  // Canvas and controls go BEFORE any <figcaption>, so the mount point keeps picture-then-caption
  // reading order. The fallback <img> is NOT removed here: it goes after the first successful
  // draw, below, so a canvas that mounts into a zero-width container or a draw that throws leaves
  // the static picture the error text points the reader at.
  if (caption) {
    host.insertBefore(cv, caption);
    host.insertBefore(controls, caption);
  } else {
    host.append(cv, controls);
  }

  const state: StateSource<DrawRoadState> = makeState({ road: [] });

  /** `solve_egress(ctx, None)`'s own answer, in the shape a solve returns -- not a placeholder.
   *
   * `permeability` is `1 - p0/p0`, which is 0 by construction; `roadMetres` is 0 because there is
   * no road; `potential` is the baked baseline; and `conductance` is `footpath_g` because
   * `edge_conductances` returns the footpath array untouched when `roads is None`
   * (src/reblock/permeability.py). So the no-road picture comes out of the same `pictureOf` call
   * a solved one does, and nothing below branches on whether a road has been solved yet. */
  const BASELINE: PyResult = {
    permeability: 0,
    roadMetres: 0,
    potential: ab.baseline.potential,
    conductance: ab.edges.footpath_g,
  };

  /** Per-edge current, `conductance[k] * (potential[rows[k]] - potential[cols[k]])` -- the
   * expression `reblock.perm_graph.permeability_graph` computes for the PNG and
   * `gen_web_bundle.py` bakes for PermGraph. The runtime returns the two arrays it is built from
   * rather than the product (design §1.6), so it is formed here. */
  const currentOf = (r: PyResult): number[] =>
    ab.edges.rows.map((row, k) =>
      r.conductance[k]! * (r.potential[row]! - r.potential[ab.edges.cols[k]!]!));

  const baselineCurrent = currentOf(BASELINE);
  const parcelRings = ab.parcels.flat();
  // `draw` reads `ground_g` only as `ground_g[i] > 0` (its halo test); the magnitude
  // `permeability_graph` puts there is `params.g_street`, which nothing in `draw` looks at. The
  // authoring bundle carries the same `mesh.ground` mask as a boolean, so 1/0 reproduces every
  // halo the PNG draws and claims nothing about the conductance to ground.
  const groundG = ab.nodes.ground.map((g) => (g ? 1 : 0));

  /** The bundle `draw` consumes, for one solve.
   *
   * TWO prefixes, and the frame below always draws prefix 1. Prefix 0 is the no-road baseline and
   * prefix 1 is `r`, which is what puts the node colour scale where `draw` expects it: it reads
   * `vmax` off `prefix.potential[0]` on purpose ("roads only lower potentials, so this is the
   * shared scale"), so a single-prefix bundle would rescale the ramp on every edit and two roads
   * the reader drew could not be compared by colour.
   *
   * `roads` is empty and the reader's own road is stroked separately (`drawPending` below),
   * because `draw`'s road layer is a corridor of a stated `width_m` and this file does not carry
   * the width `solve.py` hands the solver (`PermeabilityParams().min_road_width_m`). An invented
   * number there would be a claim about the solver that nothing here could check. */
  const pictureOf = (r: PyResult): Drawable => ({
    parcels: parcelRings,
    boundary,
    streets: ab.streets,
    nodes: { cx: ab.nodes.cx, cy: ab.nodes.cy, ground_g: groundG },
    edges: {
      rows: ab.edges.rows,
      cols: ab.edges.cols,
      footpath_g: ab.edges.footpath_g,
      // `conductance > footpath_g` is exactly what `permeability_graph` stores as `upgraded` for
      // the PNG. It recomputes it here because `PyResult` carries the conductances and not the
      // mask -- the same rule applied to the same numbers, not a second opinion about which edges
      // a road raised. -1 is "never raised"; 1 is "raised at prefix 1", the prefix drawn.
      first_upgraded_at: ab.edges.footpath_g.map((g, k) => (r.conductance[k]! > g ? 1 : -1)),
    },
    roads: [],
    prefix: {
      potential: [ab.baseline.potential, r.potential],
      current: [baselineCurrent, currentOf(r)],
    },
    encoding: GRAPH_ENCODING,
  });

  /** The last picture that was drawn from a SUCCESSFUL solve (or the baseline, before there was
   * one). A failed solve leaves it alone, which is what "keeps the last good picture" means in
   * design §6: the reader keeps the answer they had rather than watching the figure blank. */
  let picture = pictureOf(BASELINE);

  // Vertices placed since the last settle, and where the pointer is now. Both are drawing state,
  // not `DrawRoadState`: a half-drawn road is not a road, so it is never published to the URL and
  // never handed to the runtime.
  let pending: [number, number][] = [];
  let hover: [number, number] | null = null;
  /** Which vertex of the settled road the pointer is moving, or null. At this scope rather than
   * inside `wireDrawing` because `clear()` has to be able to end a drag. */
  let dragging: number | null = null;

  let booting: Promise<void> | null = null;
  let booted = false;

  // Assigned only from inside the first sized callback below, exactly like screen-map.ts's own
  // `size`/`view` -- everything that reads them (`render`, the pointer handlers) is wired from
  // there too, so there is no ordering in which they are read before that callback has run.
  let size: { width: number; height: number };
  let view: View;
  const ctx = cv.getContext("2d")!;

  const strokePolyline = (pts: [number, number][]): void => {
    if (pts.length < 2) return;
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(view, x, y);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  };

  /** The reader's road on top of the graph: the settled one, or -- while they are drawing a
   * replacement -- the pending vertices and the segment following the pointer. Only one of the two
   * is shown, so a road being redrawn does not sit on top of the one it replaces.
   *
   * At a fixed screen width rather than the solver's corridor width; see `pictureOf`. The dots are
   * what make a single placed vertex visible at all -- with none, the first click of every road
   * draws nothing. */
  const drawPending = (): void => {
    const e = GRAPH_ENCODING;
    ctx.strokeStyle = e.road_color;
    ctx.fillStyle = e.road_color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    if (pending.length === 0) {
      strokePolyline(state.get().road);
      return;
    }
    strokePolyline(hover === null ? pending : [...pending, hover]);
    for (const [x, y] of pending) {
      const [sx, sy] = toScreen(view, x, y);
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const render = (): void => {
    // Always prefix 1: `picture` puts the drawn solve there and the baseline at 0, so this is the
    // solve being shown and `draw`'s `vmax` still comes off the baseline (see `pictureOf`).
    draw(ctx, picture, { view, prefix: 1, layer: "current", halos: true }, size);
    drawPending();
  };

  // Coalesces every redraw a gesture produces onto the next animation frame, so a pointermove
  // stream cannot queue more frames than the display can render -- each call before the frame
  // fires just moves WHICH state the eventual `render` reads, it never queues a second one.
  // `frameScheduled` is the guard that makes this coalescing rather than merely async:
  // `requestAnimationFrame` on its own queues every call it is given (harness.ts's own comment).
  //
  // The frame drains the SOLVE queue too -- the first of design §4's "two throttles, not one" --
  // so a drag that produces twenty pointermoves between two frames asks for one solve, not twenty.
  // The second is the in-flight guard below.
  let frameScheduled = false;
  const scheduleRender = (): void => {
    if (frameScheduled) return;
    frameScheduled = true;
    // `runOrReport`: an animation-frame callback runs from the browser's own dispatch, as far
    // outside the fetch chain above as the ResizeObserver callback below is -- without this, a
    // throw here is an uncaught exception with a blank figure and no message (dom/error.ts).
    requestAnimationFrame(() => runOrReport(host, LABEL, () => {
      frameScheduled = false;
      render();
      pumpSolve();
    }));
  };

  /** The road waiting to be solved, whether one is running, and which generation it belongs to.
   *
   * Latest-wins, and never two at once. Both are written against `PyRuntime`, which promises no
   * ordering at all, rather than against the one implementation that exists: `pyodideRuntime.solve`
   * is `async` but contains no `await`, so today it blocks the main thread and CANNOT be
   * re-entered -- single-threadedness is what makes the hazard unreachable, not what causes it. It
   * is also exactly why the obvious next move is a worker (a sub-second blocking call on the main
   * thread is the thing a drag cannot afford), and a worker would both let answers land out of
   * order and let a drag pile up roughly sixty solves a second behind the frame coalesce. So the
   * guard is here now, while the interface it defends is the thing being relied on (design §4).
   *
   * `epoch` is the other half, and it is the half `solving` cannot do: a solve already issued
   * cannot be recalled, so `clear()` bumps the generation and the answer that lands afterwards is
   * DISCARDED rather than repainting the graph a reader has just emptied. `requestSolve` bumps it
   * too, so a superseded road's answer never paints over its replacement's. */
  let requested: [number, number][] | null = null;
  let solving = false;
  let epoch = 0;

  const requestSolve = (road: [number, number][]): void => {
    epoch += 1;
    requested = road;
    scheduleRender();
  };

  /** The runtime's two arrays against the mesh they are indexed into.
   *
   * `boot` checked IDENTITY (`block_id`); this is ARITY, and it is the half that fails silently. A
   * short `potential` makes `r.potential[rows[k]]` undefined, so every current is NaN, so
   * `rampColor` indexes the ramp with NaN and returns `undefined` -- and assigning `undefined` to
   * `ctx.fillStyle` is IGNORED by a browser rather than raising. The reader would get a subtly
   * wrong picture with nothing anywhere to say so. Task 5's parity test compares the runtime's
   * NUMBERS against CPython's; it does not exercise this widget's indexing, so it cannot cover
   * this. */
  const checkArity = (r: PyResult): void => {
    if (r.potential.length !== ab.nodes.cx.length
        || r.conductance.length !== ab.edges.rows.length) {
      throw new Error(`${LABEL}: the runtime returned ${r.potential.length} potentials and `
        + `${r.conductance.length} conductances for a mesh of ${ab.nodes.cx.length} parcels and `
        + `${ab.edges.rows.length} edges`);
    }
  };

  /** Hand the pending road to the runtime, or say why not.
   *
   * The `isSolvable` refusal is the widget's half of the arity check `solve.py` also performs; the
   * point of having it here is that the reader is told what is wrong instead of being shown a
   * Python traceback. The `booted` gate is not politeness either: `PyRuntime.solve` assumes
   * `boot()` has already resolved and `pyodideRuntime`'s own implementation refuses a call that
   * arrives before it has (py/runtime.ts -- an `async` method, so the refusal is a rejection,
   * pinned in py-runtime.test.ts). It never boots on the caller's behalf, because a reader who
   * does not press the button must never pay for the download -- and drawing before booting is a
   * thing this widget supports, so this gate is on the ordinary path, not an edge of one. */
  function pumpSolve(): void {
    if (requested === null || solving) return;
    const road = requested;
    requested = null;
    // Captured at ISSUE time: `clear()` or a newer request moves `epoch` on, and the two handlers
    // below compare against it before they touch the picture or the readout.
    const issued = epoch;
    if (!isSolvable(road)) {
      readout.textContent = NEEDS_TWO_POINTS;
      return;
    }
    if (!booted) {
      readout.textContent = NEEDS_RUNTIME;
      return;
    }
    solving = true;
    void runtime.solve(road)
      .then((r) => {
        if (issued !== epoch) return;
        checkArity(r);
        picture = pictureOf(r);
        readout.textContent = `${r.roadMetres.toFixed(0)} m of road · `
          + `${(r.permeability * 100).toFixed(1)}% permeability`;
      })
      .catch((err: unknown) => {
        if (issued !== epoch) return;
        // `picture` is deliberately untouched: the last good GRAPH stays on the canvas and the
        // reader is told what failed, rather than being left with a blank figure (design §6). The
        // road overlay is a different matter -- `settle` has already published the road that
        // failed, so the stroke a reader sees is the road they drew over the graph of the last one
        // that solved, and the sentence says exactly that rather than claiming both are old.
        // A Python traceback is not a reader-facing message, so the cause is quoted inside a
        // sentence that says what it was doing.
        const cause = err instanceof Error ? err.message : String(err);
        readout.textContent = `The solve failed (${cause}). The graph above is the last road `
          + `that solved.`;
      })
      // Whatever happened, the next request may run and the frame that shows the result is due.
      // Only `scheduleRender` here: its own frame calls `pumpSolve`, so a request that arrived
      // while this one was running is picked up there rather than down a second path.
      .finally(() => {
        solving = false;
        scheduleRender();
      });
  }

  const settle = (): void => {
    const road = withoutRepeats(pending);
    pending = [];
    hover = null;
    // Published to the URL only if it is a road -- a one-point "road" spelled into `?road=` is a
    // query `polylineParam` refuses to decode, so the reader would watch it disappear on the next
    // load. It is published BEFORE the solve rather than after, because drawing a road while the
    // runtime is still unloaded is a supported order (the boot handler solves whatever is in hand)
    // and deferring the write would lose the road the reader drew.
    if (isSolvable(road)) state.set({ road });
    requestSolve(road);
  };

  const clear = (): void => {
    pending = [];
    hover = null;
    // Ends a drag in progress, and what it protects is the readout, not an index. MEASURED by
    // removing this line and driving pointerdown-on-vertex -> Escape -> pointermove: nothing
    // throws and no index goes out of range -- `state.get().road.map(...)` over the now-empty road
    // never runs its callback at all -- so an earlier version of this comment named a failure that
    // does not exist. What actually happens is that `dragging` stays set until the pointer is
    // lifted (`release`, below, is what would otherwise end it), and every `pointermove` in that
    // window takes the DRAG branch instead of the hover one: it re-sets the road to the empty
    // array it already is and calls `requestSolve([])`, which `pumpSolve` refuses -- putting
    // `NEEDS_TWO_POINTS` in the readout over whatever was there. That is a refusal shown to a
    // reader who settled nothing, which is exactly what `NEEDS_TWO_POINTS`'s own declaration in
    // this file says cannot happen ("a REFUSAL and nothing else ... it appears only when a road
    // that is not a road was settled"). The window closes at `pointerup` either way -- the hover
    // preview comes back for the next road, also measured -- so this is a corrupted readout for
    // the rest of one gesture, not a stuck widget.
    dragging = null;
    // Both halves of "cancel": drop the road waiting to be solved, and move the generation on so
    // that an answer already in flight is discarded when it lands instead of repainting the graph
    // the reader has just emptied.
    requested = null;
    epoch += 1;
    picture = pictureOf(BASELINE);
    state.set({ road: [] });
    readout.textContent = booted ? DRAW_A_ROAD : NEEDS_RUNTIME;
    scheduleRender();
  };

  bootButton.addEventListener("click", () => {
    // `??=`, and everything that must happen once lives INSIDE it: a reader looking at a button
    // that has been saying "Loading..." for twenty seconds presses it again, and without this
    // each press would start its own follow-on chain -- another status write, another solve of
    // the current road. `PyRuntime.boot()` memoises its own side of this (py/runtime.ts); what is
    // memoised here is this file's own `.then`.
    //
    // A boot that rejects stays rejected, on both sides: `booting` is a settled promise from then
    // on, so a later press does not retry. That matches `boot()`'s own contract -- the CDN being
    // unreachable or the pin 404ing is not something a second press fixes -- and the label stops
    // advertising a download that will not start.
    booting ??= (() => {
      bootButton.textContent = BOOT_LOADING;
      return runtime.boot().then(
        () => {
          booted = true;
          bootButton.textContent = BOOT_READY;
          // A road may already be in hand -- `?road=` decoded at mount, or one drawn before the
          // runtime was up -- and the press answers it immediately. With NO road there is nothing
          // to solve and nothing to refuse, so the readout invites one instead of reporting a
          // refusal the reader did not trigger; `NEEDS_TWO_POINTS` stays a refusal.
          const road = state.get().road;
          if (road.length === 0) readout.textContent = DRAW_A_ROAD;
          else requestSolve(road);
        },
        (err: unknown) => {
          bootButton.textContent = BOOT_FAILED;
          const cause = err instanceof Error ? err.message : String(err);
          readout.textContent = `The Python runtime could not load (${cause}). The graph above `
            + `is this block with no new road.`;
        },
      );
    })();
  });

  /** How near a vertex a press has to land to grab it rather than place a new one, in SCREEN
   * pixels. Screen, not metres: the tolerance a pointer has is a property of the display, and a
   * world-space radius would grab vertices a reader cannot even tell apart at a zoomed-out fit. */
  const GRAB_PX = 8;

  /** Index of the settled road's vertex under `(sx, sy)`, or -1 if the press is not on one.
   *
   * `nearest` (view/transform.ts) does the search in whatever space it is handed -- SCREEN here,
   * exactly as Frontier's hover uses it -- and this adds the radius it deliberately does not
   * have: without one, every press anywhere on the canvas would "hit" the closest vertex and no
   * new vertex could ever be placed. */
  const vertexAt = (sx: number, sy: number): number => {
    const road = state.get().road;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const [x, y] of road) {
      const [px, py] = toScreen(view, x, y);
      xs.push(px);
      ys.push(py);
    }
    const i = nearest(xs, ys, sx, sy);
    if (i < 0) return -1;
    const dx = xs[i]! - sx;
    const dy = ys[i]! - sy;
    return dx * dx + dy * dy <= GRAB_PX * GRAB_PX ? i : -1;
  };

  const wireDrawing = (): void => {
    cv.addEventListener("pointerdown", (ev) => {
      // A press ON an existing vertex moves it (design §4); a press anywhere else places a new
      // one. The hit-test is skipped while a road is part-drawn, so placing a vertex next to one
      // already down is never mistaken for grabbing it.
      const hit = pending.length === 0 ? vertexAt(ev.offsetX, ev.offsetY) : -1;
      if (hit >= 0) {
        dragging = hit;
        // Without capture the drag stops tracking the moment the pointer leaves the canvas and the
        // pointerup never arrives, so the vertex stays glued to the cursor (perm-graph.ts's own
        // reasoning for its pan).
        cv.setPointerCapture(ev.pointerId);
        return;
      }
      pending.push(toWorld(view, ev.offsetX, ev.offsetY));
      hover = null;
      scheduleRender();
    });
    cv.addEventListener("pointermove", (ev) => {
      const at = dragging;
      if (at !== null) {
        const road = state.get().road.map((p, i): [number, number] =>
          (i === at ? toWorld(view, ev.offsetX, ev.offsetY) : p));
        state.set({ road });
        // Re-solved as it moves, not on release: §1.3 measured one solve at 0.06 s expressly to
        // conclude "re-solve on drag, and let the number move with the road". `requestSolve` puts
        // it on the next frame and never runs two at once.
        requestSolve(road);
        return;
      }
      if (pending.length === 0) return;
      hover = toWorld(view, ev.offsetX, ev.offsetY);
      scheduleRender();
    });
    // `pointercancel` as well as `pointerup`: a captured stream can still be cancelled (the OS
    // takes the gesture, the element is removed), and releasing only on `pointerup` would leave
    // the vertex following the pointer for ever after.
    const release = (ev: PointerEvent): void => {
      if (dragging === null) return;
      dragging = null;
      cv.releasePointerCapture(ev.pointerId);
    };
    cv.addEventListener("pointerup", release);
    cv.addEventListener("pointercancel", release);
    cv.addEventListener("dblclick", () => settle());
    cv.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") settle();
      else if (ev.key === "Escape") clear();
    });
  };

  // The observed element is the CANVAS ITSELF: `width: 100%` makes its content box track the
  // container's width, answering "how many CSS pixels am I drawing into?" in one hop
  // (perm-graph.ts's own reasoning).
  //
  // `runOrReport`: a real ResizeObserver delivers from the browser's own dispatch, OUTSIDE the
  // fetch chain above, so that chain's `.catch` cannot see a throw in here -- it would be an
  // uncaught exception with a blank figure and no message (dom/error.ts).
  let firstDraw = true;
  observeSize(cv, (measured) => runOrReport(host, LABEL, () => {
    size = measured;
    sizeCanvas(cv, size);
    // The block's EXTERIOR ring specifically, `boundary[0]`, not every ring: an interior ring
    // lies inside the exterior by definition, so it cannot widen the box, and fitting the union
    // would only invite a reader to wonder whether it could. Non-null because the boot check
    // above refuses an empty ring list.
    //
    // MEASURED on the committed bundle: the parcels' and the streets' own bounding boxes are
    // identical to this one digit for digit, so nothing `draw` paints falls outside the fit. The
    // reader's road can -- they may click anywhere on the canvas, including the padding -- and it
    // deliberately does not move the fit, so the block does not shrink under a road drawn past
    // its edge.
    view = fitBbox(bboxOf(boundary[0]!), size.width, size.height);
    render();
    if (firstDraw) {
      firstDraw = false;
      // Only the SUBSCRIPTION and the pointer wiring are deferred to here, so a later resize can
      // never register a second of either. The pointer handlers have to be: they read `view` to
      // turn an offset into metres, and it does not exist until this callback has run.
      state.subscribe(() => scheduleRender());
      wireDrawing();
      // Only now: the static picture is the honest one until a real one has replaced it.
      removeFallbackImage(host);
    }
  }));
}
