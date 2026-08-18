/** The mount contract: a page carries a placeholder and nothing else. */
import { localState, type StateFactory } from "./state.js";

export type Widget = (host: HTMLElement, makeState: StateFactory) => void;

const REGISTRY = new Map<string, Widget>();

export function register(name: string, w: Widget): void {
  // Throw rather than replace. With one widget a collision was invisible and harmless; with several
  // it silently disables whichever registered first, and the page still looks fine.
  if (REGISTRY.has(name)) throw new Error(`widget already registered: ${name}`);
  REGISTRY.set(name, w);
}

export function mountAll(root: ParentNode = document): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-widget]"))) {
    const name = el.dataset.widget!;
    const widget = REGISTRY.get(name);
    // No default. The name arrives from HTML -- a genuinely open boundary, so a string lookup is
    // right here -- but an unknown one must throw rather than leave a silently empty mount point
    // that looks like a widget which merely failed to draw.
    if (widget === undefined) throw new Error(`unknown data-widget: ${name}`);
    // Per-widget isolation: one widget throwing must not stop the widgets after it from mounting,
    // and the failure must be visible where it happened rather than console-only.
    try {
      widget(el, localState);
    } catch (err) {
      showMountError(el, err);
    }
  }
}

function showMountError(el: HTMLElement, err: unknown): void {
  const caption = el.querySelector("figcaption");
  const msg = `This figure could not load interactively (${String(err)}). The static image above still applies.`;
  if (caption) caption.textContent = msg;
  else el.append(Object.assign(document.createElement("p"), { textContent: msg }));
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
import { permGraph } from "./widgets/perm-graph.js";
register("perm-graph", permGraph);

// DOMContentLoaded fires once per full page load. That is sufficient only because this project's
// mkdocs.yml does not enable Material's navigation.instant feature (confirmed absent as of this
// writing): with that feature on, page navigations swap content via fetch + DOM replacement
// without a reload, so this listener would fire on the first page and never again. If
// navigation.instant is ever turned on, replace this with Material's `document$` subscription
// (which fires on every instant navigation) so mountAll() keeps running on every page.
document.addEventListener("DOMContentLoaded", () => mountAll());
