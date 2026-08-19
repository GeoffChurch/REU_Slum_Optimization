import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Widget } from "../src/mount.js";

/** Permanent regression coverage for the two hardening fixes register()/mountAll() ship: a
 * duplicate widget name must throw at registration time, and one widget throwing during mount
 * must not prevent a later widget from mounting -- with the failure visible on the page, not
 * merely swallowed. Both behaviours were proved once via a fault-injection ritual during Task 5
 * and then reverted, per the plan; these tests are the permanent guard the plan omitted (Task 5
 * review, fix round 1).
 *
 * The shipped esbuild IIFE bundle (what widgets-bundle.test.ts evaluates) has no exports, so
 * `register`/`mountAll` are reachable only at the source-module level -- scripts/test.sh compiles
 * src/ and test/ together for exactly this reason.
 *
 * mount.ts registers "perm-graph" and calls document.addEventListener as IMPORT-TIME side
 * effects (see its own registration comment), so `document` must be stubbed on globalThis BEFORE
 * the module is imported -- hence `await import(...)` below rather than a static top-level
 * import (a static import is hoisted and evaluated before any of this file's own code runs).
 *
 * Node's test runner isolates test FILES into separate processes but not tests WITHIN a file, so
 * ../src/mount.js -- and the `document` global it reads -- are shared state across both tests
 * below (ESM caches the module: only the first import in the process actually runs its top-level
 * code, including "perm-graph"'s registration). Each test reinstalls its own complete `document`
 * stub immediately before doing anything, rather than relying on whichever test's import happens
 * to run first, so the two stay order-independent regardless of which one node:test picks first.
 */

interface FakeNode { textContent: string }

function stubDocument(): void {
  globalThis.document = {
    addEventListener: () => {},
    createElement: (tag: string): FakeNode => { void tag; return { textContent: "" }; },
  } as unknown as Document;
}

test("register throws on a duplicate widget name", async () => {
  stubDocument();
  const { register } = await import("../src/mount.js");

  // mount.ts's own top-level `register("perm-graph", permGraph)` has already run by the time this
  // await resolves (whether it ran on THIS import or an earlier one in the process -- see the
  // file comment), so "perm-graph" is already taken. Re-registering it under that same name is
  // exactly the regression this guards: with the duplicate check removed, this second call would
  // silently replace the first widget's registration instead of throwing.
  assert.throws(
    () => register("perm-graph", (() => {}) as Widget),
    /widget already registered: perm-graph/,
  );
});

interface FakeMountPoint {
  dataset: { widget: string };
  appended: FakeNode[];
  querySelector(selector: string): FakeNode | null;
  append(...nodes: FakeNode[]): void;
}

function makeMountPoint(widgetName: string): FakeMountPoint {
  const appended: FakeNode[] = [];
  return {
    dataset: { widget: widgetName },
    appended,
    querySelector: () => null, // no <figcaption> -- forces showMountError's append() branch
    append: (...nodes) => { appended.push(...nodes); },
  };
}

test("a widget that throws during mount does not block a later widget, and the failure shows up "
  + "on its own mount point", async () => {
  stubDocument();
  const { register, mountAll } = await import("../src/mount.js");

  // Names unique to this test, so this test's registrations can never collide with the other
  // test's re-registration of "perm-graph" or with the real perm-graph widget, and mountAll never
  // touches the real PermGraph (no fetch, no canvas) -- only these two fakes.
  const failing: Widget = () => { throw new Error("forced synchronous failure"); };
  const okay: Widget = (host) => {
    (host as unknown as FakeMountPoint).append({ textContent: "isolation-ok" });
  };
  register("isolation-fail-widget", failing);
  register("isolation-ok-widget", okay);

  const failEl = makeMountPoint("isolation-fail-widget");
  const okEl = makeMountPoint("isolation-ok-widget");
  const root = { querySelectorAll: () => [failEl, okEl] };
  mountAll(root as unknown as ParentNode);

  // The LATER widget must still have mounted -- proof mountAll kept going past the first widget's
  // throw instead of stopping there.
  assert.ok(
    okEl.appended.some((n) => n.textContent === "isolation-ok"),
    `second widget never mounted: ${JSON.stringify(okEl.appended)}`,
  );

  // The failing widget's OWN mount point must visibly carry the error. Asserting merely that
  // mountAll() didn't throw would also pass if the error were silently swallowed -- a page that
  // still looks fine with a dead widget is the precise failure mode this guard exists to catch,
  // so the assertion has to be on what the DOM stub recorded, not on the absence of an exception.
  assert.ok(
    failEl.appended.some((n) => n.textContent.includes("could not load interactively")
      && n.textContent.includes("forced synchronous failure")),
    `failing widget's mount point shows no visible error: ${JSON.stringify(failEl.appended)}`,
  );
});
