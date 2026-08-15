/** The mount contract: a page carries a placeholder and nothing else. */
import { localState, type StateSource, type WidgetState } from "./state.js";
// Registers itself with `register()` below on import -- this is the one line that puts
// PermGraph into the esbuild bundle. A widget file nothing imports never ships.
import "./widgets/perm-graph.js";

export type Widget = (host: HTMLElement, state: StateSource) => void;

const REGISTRY = new Map<string, Widget>();

export function register(name: string, w: Widget): void {
  REGISTRY.set(name, w);
}

/** Parse `data-prefix` into a non-negative integer, never NaN.
 *
 * The attribute is a string from HTML -- absent, empty, or hand-typed-wrong are all reachable --
 * and `prefix` goes on to index straight into baked arrays (`b.roads.slice(0, prefix)`,
 * `b.prefix.current[prefix]`, ...). `Number(raw)` alone lets a malformed attribute (missing,
 * "abc", "-1", "2.5") become NaN or a value that is numeric but not a valid array index; either
 * would surface downstream as a silent `undefined` read or a `.slice` with a nonsensical bound
 * instead of failing here, at the boundary, where the bad string is still in hand. Falling back to
 * 0 (rather than throwing) matches `layer`/`halos` below, which also coerce a malformed attribute
 * to a valid default instead of rejecting the whole mount over one cosmetic attribute.
 */
function parsePrefix(raw: string | undefined): number {
  const n = raw === undefined ? 0 : Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : 0;
}

function initialState(el: HTMLElement): WidgetState {
  const layer = el.dataset.layer === "conductance" ? "conductance" : "current";
  return { prefix: parsePrefix(el.dataset.prefix), layer, halos: el.dataset.halos !== "false" };
}

export function mountAll(root: ParentNode = document): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-widget]"))) {
    const name = el.dataset.widget!;
    const widget = REGISTRY.get(name);
    // No default. The name arrives from HTML -- a genuinely open boundary, so a string lookup is
    // right here -- but an unknown one must throw rather than leave a silently empty mount point
    // that looks like a widget which merely failed to draw.
    if (widget === undefined) throw new Error(`unknown data-widget: ${name}`);
    widget(el, localState(initialState(el)));
  }
}

// DOMContentLoaded fires once per full page load. That is sufficient only because this project's
// mkdocs.yml does not enable Material's navigation.instant feature (confirmed absent as of this
// writing): with that feature on, page navigations swap content via fetch + DOM replacement
// without a reload, so this listener would fire on the first page and never again. If
// navigation.instant is ever turned on, replace this with Material's `document$` subscription
// (which fires on every instant navigation) so mountAll() keeps running on every page.
document.addEventListener("DOMContentLoaded", () => mountAll());
