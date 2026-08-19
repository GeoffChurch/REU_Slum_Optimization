import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

/** Evaluates the built ../docs/js/widgets.js -- the artifact that actually ships to the page, not
 * just the source modules under src/ that transform.test.ts exercises. A cyclic import between two
 * of those modules once made the BUNDLE throw during ES module evaluation, before any of the
 * source-level unit tests could have caught it: nothing in this project ever evaluated the bundle
 * itself. This test closes that gap.
 *
 * scripts/test.sh builds the bundle immediately before running this file (see that script's own
 * comment), so BUNDLE_PATH always points at a fresh build, never a stale one.
 */
const BUNDLE_PATH = "../docs/js/widgets.js";

/** The widget names a generated page can actually emit, read out of the generator that emits them --
 * `scripts/gen_site_pages.py` is the single source of the `data-widget="…"` strings on the site, so
 * deriving them here means a third widget is covered by the act of putting it on a page. Spec §5 asked
 * for "both widgets register"; a hardcoded pair would have satisfied the letter of that and none of
 * the point, which is the pattern this branch has been punishing all along (final review, I4). */
function emittedWidgetNames(): string[] {
  const generator = readFileSync("../scripts/gen_site_pages.py", "utf8");
  const names = [...generator.matchAll(/data-widget="([a-z-]+)"/g)].map((m) => m[1]!);
  const unique = [...new Set(names)].sort();
  assert.ok(unique.length >= 2,
    `expected at least perm-graph and frontier in the generator, found ${JSON.stringify(unique)}`);
  return unique;
}

test("the built bundle evaluates without throwing", () => {
  const source = readFileSync(BUNDLE_PATH, "utf8");

  // The ONLY global the bundle touches at *evaluation* time (top-level code that runs the moment
  // the script loads, as opposed to code inside an event handler or a fetch callback that runs
  // only once a widget actually mounts) is `document.addEventListener("DOMContentLoaded", ...)`,
  // the very last statement of mount.ts's top level. So a bare `addEventListener` stub is the
  // entire DOM surface this test needs -- not jsdom, not a browser, just enough for the script to
  // finish evaluating. `vm.createContext` gives a fresh realm with its own real Math/Map/Array/etc
  // (those are not DOM, so they need no stub); `document` is the only addition.
  const registeredEvents: string[] = [];
  const context = vm.createContext({
    document: { addEventListener: (name: string) => { registeredEvents.push(name); } },
  });

  // The assertion IS that this does not throw -- node --test fails the case on any uncaught
  // exception, with the real stack trace, so no try/catch is needed here.
  vm.runInContext(source, context, { filename: BUNDLE_PATH });

  // A throw partway through evaluation would leave `registeredEvents` empty just as surely as a
  // silently-truncated bundle would -- assert execution actually reached the bottom of the file
  // and wired up the listener, so a change that made the whole IIFE body a no-op could not pass
  // this test by accident.
  assert.deepEqual(registeredEvents, ["DOMContentLoaded"]);
});

test("the shipped bundle registers every widget name a generated page emits", () => {
  // Spec §5's actual requirement, and the level that matters: piece C's defect was a circular import
  // that made the BUNDLE throw behind an intact PNG, so source-module coverage (mount.test.ts) is not
  // the same guard. This evaluates ../docs/js/widgets.js, then drives its own DOMContentLoaded
  // listener over one mount point per emitted name and checks what each element was told.
  //
  // Neither widget can boot here (no fetch, no layout), so both legitimately report a failure -- what
  // must NOT appear is "unknown data-widget", which is precisely what an unregistered name produces.
  const names = emittedWidgetNames();
  const source = readFileSync(BUNDLE_PATH, "utf8");

  interface FakeNode { textContent: string }
  const mounts = names.map((name) => ({
    dataset: { widget: name },
    appended: [] as FakeNode[],
    style: {} as Record<string, string>,
    querySelector: (): null => null,
    append(...nodes: FakeNode[]): void { this.appended.push(...nodes); },
    insertBefore(node: FakeNode): void { this.appended.push(node); },
    getBoundingClientRect: () => ({ width: 0, height: 0, left: 0, top: 0 }),
    addEventListener: (): void => {},
  }));

  let domReady: (() => void) | undefined;
  const context = vm.createContext({
    document: {
      addEventListener: (event: string, fn: () => void) => {
        if (event === "DOMContentLoaded") domReady = fn;
      },
      querySelectorAll: () => mounts,
      createElement: (): FakeNode => ({ textContent: "" }),
    },
  });
  vm.runInContext(source, context, { filename: BUNDLE_PATH });
  assert.ok(domReady !== undefined, "the bundle never wired its DOMContentLoaded listener");
  domReady();

  for (const [i, name] of names.entries()) {
    const said = mounts[i]!.appended.map((n) => n.textContent).join(" ");
    assert.ok(!said.includes("unknown data-widget"),
      `"${name}" is not registered in the shipped bundle: ${said}`);
    assert.ok(said.length > 0,
      `"${name}" produced no message at all, so mountAll never reached its widget`);
  }
});
