/** The mount contract: a page carries a placeholder and nothing else. */
import { showWidgetError } from "./dom/error.js";
import type { StateFactory } from "./state.js";
import type { UrlCodec } from "./url/param.js";
import {
  browserLocation, debounce, systemTimers, urlStore, type UrlStore,
} from "./url/store.js";

export type Widget<T> = (host: HTMLElement, makeState: StateFactory<T>) => void;

/** A widget with its `T` already erased. `register` captures the generic in `mount`'s closure, so
 * the REGISTRY stays a plain non-generic map while the (widget, codec) pairing is still checked
 * where they are named together. */
interface Registration {
  readonly keys: readonly string[];
  mount(host: HTMLElement, store: UrlStore): void;
}

const REGISTRY = new Map<string, Registration>();

export function register<T>(name: string, w: Widget<T>, codec: UrlCodec<T>): void {
  // Throw rather than replace. With one widget a collision was invisible and harmless; with several
  // it silently disables whichever registered first, and the page still looks fine.
  if (REGISTRY.has(name)) throw new Error(`widget already registered: ${name}`);
  // `Object.values` over a DECLARED mapped type -- a loop over a schema, not a string lookup into
  // a closed set. `as object` mirrors store.ts's own cast on the same generic-mapped-type call
  // (`bind`'s comment there); the cast right after it narrows -- to `{ keys: ... }`, not the full
  // `Param<unknown>` -- because only `.keys` is read below, and the wider cast would silently
  // accept an unsound `p.decode(got, anything)` here if a later task ever added one.
  const keys = (Object.values(codec as object) as readonly { keys: readonly string[] }[])
    .flatMap((p) => [...p.keys]);
  REGISTRY.set(name, {
    keys,
    // `reserve()` runs on THIS line -- inside `mountAll`'s DOM walk -- while `slot.bind` runs
    // whenever the widget gets round to calling `makeState`, which every one of the five does from
    // inside its own `fetch().then(boot)`. Splitting the two is what makes the emitted query's
    // order the page's order instead of the network's: see `UrlSlot` (url/store.ts) for the
    // ordering this replaced, and mount.test.ts's out-of-order test for the guard on it.
    mount: (host, store) => {
      const slot = store.reserve();
      w(host, (initial) => slot.bind(codec, initial));
    },
  });
}

/** 300 ms after the last change. Long enough that a drag writes once; short enough that a reader
 * who stops and copies the address bar gets the view they are looking at. */
export const URL_DEBOUNCE_MS = 300;

function defaultStore(): UrlStore {
  return urlStore(browserLocation(), debounce(URL_DEBOUNCE_MS, systemTimers));
}

export function mountAll(root: ParentNode = document, store: UrlStore = defaultStore()): void {
  // Which widget on THIS page claimed which query key. Two mount points sharing a key would
  // cross-talk silently through one set of values -- including two mount points of the SAME
  // widget, which is why the check is per mount point rather than per registration.
  const claimed = new Map<string, string>();
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-widget]"))) {
    // Per-widget isolation: one widget throwing must not stop the widgets after it from mounting,
    // and the failure must be visible where it happened rather than console-only.
    //
    // The unknown-name lookup is INSIDE this try (fix round 2, review finding M7). It used to throw
    // one line above it, which made the single failure mode that also aborts every LATER mount point
    // the only one with no on-page message: a widget whose registration was lost -- exactly what
    // finding I2 showed nothing tested -- produced a console-only error behind an intact-looking PNG
    // fallback, which is this project's signature defect. Now it renders like any other failure.
    // The URL-key collision below is inside it for the same reason.
    try {
      const name = el.dataset.widget!;
      const widget = REGISTRY.get(name);
      // No default. The name arrives from HTML -- a genuinely open boundary, so a string lookup is
      // right here -- but an unknown one must throw rather than leave a silently empty mount point
      // that looks like a widget which merely failed to draw.
      if (widget === undefined) throw new Error(`unknown data-widget: ${name}`);
      // Checked against the WHOLE key list before any of it is committed to `claimed`.
      // Committing key-by-key would leave a phantom claim behind on a widget that never actually
      // mounted, whenever that widget's OWN key list collides partway through: an earlier key
      // would already be in `claimed` under this widget's name even though the throw below stops
      // `widget.mount` from ever running for it, so a later mount point's collision message would
      // misname the claimant as a widget that holds nothing (fix round 2, M2).
      //
      // `own` is what keeps that fix from also losing the OTHER collision: one codec mapping two
      // different state fields to the same URL key (fix round 3, M-self). `own` and `claimed` are
      // written on DELIBERATELY DIFFERENT schedules -- that asymmetry is the whole mechanism, not
      // an inconsistency. `own` is written PROGRESSIVELY, key-by-key, right here in this same loop
      // (`own.add(k)` below), so a key repeated within one widget's own list finds its earlier
      // occurrence in the very same pass that is still checking it. `claimed` is written only in
      // the separate commit loop below, after this widget's whole list has cleared -- so a
      // mid-list throw never leaves a phantom claim under this widget's name for a later mount
      // point to misreport. Each set does the job the other's schedule cannot; writing `own` on
      // `claimed`'s schedule instead (i.e. only in the commit loop) would silently reintroduce the
      // self-collision bug fix round 3 fixed, since the check pass would then have nothing of
      // THIS widget's own keys to compare a repeat against (fix round 4 -- a prior version of this
      // comment wrongly described `own` as following `claimed`'s schedule; it does not).
      const own = new Set<string>();
      for (const k of widget.keys) {
        const prior = claimed.get(k);
        if (prior !== undefined) {
          throw new Error(`URL key "${k}" is claimed by both ${prior} and ${name} on this page`);
        }
        if (own.has(k)) {
          // A message distinct from the cross-widget one above: "claimed by both X and X" is
          // technically accurate (it IS this widget's own codec colliding with itself) but reads
          // like a typo rather than a bug report, so this names the actual problem instead.
          throw new Error(`URL key "${k}" is claimed twice by ${name}'s own codec`);
        }
        own.add(k);
      }
      for (const k of widget.keys) claimed.set(k, name);
      widget.mount(el, store);
    } catch (err) {
      showMountError(el, err);
    }
  }
}

// One shared renderer for all three failure paths (final review, M7) -- see dom/error.ts for why it
// is its own module and not this one.
function showMountError(el: HTMLElement, err: unknown): void {
  showWidgetError(el, "This figure", err);
}

// Registration lives HERE, not inside the widget module, and that is deliberate: importing the
// widget was previously done for its `register("perm-graph", permGraph)` side effect at the
// widget's own top level, and that widget module imported `register` back from this file to do
// it. ES module evaluation fully resolves a module's imports -- running the imported module's own
// top-level body -- before executing the importing module's body, so with that shape the widget's
// top-level `register(...)` call ran BEFORE this file reached its own `const REGISTRY = new
// Map()` above, throwing `TypeError: Cannot read properties of undefined (reading 'set')` on
// every page load, before the DOMContentLoaded listener below was ever wired up (silent, because
// the PNG fallback stayed visible and the page looked fine). Importing only the plain `permGraph`
// function here -- and registering it explicitly, after REGISTRY exists -- breaks the cycle:
// perm-graph.ts now has no runtime import of this file at all (see its `import type` comment), so
// there is nothing left to reorder. Do not move this back into the widget module.
import { permGraph, PERM_GRAPH_URL } from "./widgets/perm-graph.js";
register("perm-graph", permGraph, PERM_GRAPH_URL);
// Same shape, same reason -- registered HERE, after REGISTRY exists, never from inside the widget
// module (see the paragraph above).
import { frontier, FRONTIER_URL } from "./widgets/frontier.js";
register("frontier", frontier, FRONTIER_URL);
// Third widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { displacementField, FIELD_URL } from "./widgets/displacement-field.js";
register("displacement-field", displacementField, FIELD_URL);
// Fourth widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { regionGrow, REGION_GROW_URL } from "./widgets/region-grow.js";
register("region-grow", regionGrow, REGION_GROW_URL);
// Fifth widget, same shape, same reason -- registered HERE, after REGISTRY exists, never from
// inside the widget module (see the paragraph above).
import { screenMap, SCREEN_MAP_URL } from "./widgets/screen-map.js";
register("screen-map", screenMap, SCREEN_MAP_URL);

// DOMContentLoaded fires once per full page load. That is sufficient only because this project's
// mkdocs.yml does not enable Material's navigation.instant feature (confirmed absent as of this
// writing): with that feature on, page navigations swap content via fetch + DOM replacement
// without a reload, so this listener would fire on the first page and never again. If
// navigation.instant is ever turned on, replace this with Material's `document$` subscription
// (which fires on every instant navigation) so mountAll() keeps running on every page.
document.addEventListener("DOMContentLoaded", () => mountAll());
