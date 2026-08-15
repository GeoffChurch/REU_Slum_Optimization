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
