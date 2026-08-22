import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { Widget } from "../src/mount.js";
import { boolParam, type UrlCodec } from "../src/url/param.js";
import { urlStore, type UrlStore } from "../src/url/store.js";
import { fakeLocation, writeNow } from "./harness.js";

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

/** The state of a widget that has none worth naming. `register` now takes a codec, and
 * `UrlCodec<T>`'s mapped type needs a `T` with at least one field to be worth writing, so every
 * fake widget in this file is a `Widget<Nothing>`. */
interface Nothing { on: boolean }

/** `NOTHING` is for registrations that never reach the REGISTRY -- both tests below that use it
 * assert `register` THROWS on the name, so its key is never claimed by anything. Registrations that
 * are actually mounted build their own codec with a key unique to that test instead, because
 * `mountAll` throws when two mount points on one page claim the same key (the test for that is at
 * the bottom of this file). */
const NOTHING: UrlCodec<Nothing> = { on: boolParam("nothing-on") };

/** mountAll's `store` default constructs a real `urlStore`, which reads `window.location.search`
 * immediately -- and `window` does not exist under node:test. Every mountAll call in this file
 * therefore passes a store explicitly.
 *
 * `bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T>` is a generic METHOD, and a non-generic
 * arrow does not satisfy it, so the stub keeps the type parameter. Each `reserve()` hands back a
 * fresh slot that returns each widget its own `initial` unchanged: these tests are about mounting,
 * not about URL round-tripping, which url-store.test.ts covers directly. The ordering test at the
 * bottom of this file is the one that needs a REAL store, and builds its own. */
const noStore: UrlStore = {
  reserve: () => ({
    bind: <T>(_c: UrlCodec<T>, initial: T) =>
      ({ get: () => initial, set: () => {}, subscribe: () => {} }),
  }),
};

test("register throws on a duplicate widget name", async () => {
  stubDocument();
  const { register } = await import("../src/mount.js");

  // mount.ts's own top-level `register("perm-graph", permGraph, PERM_GRAPH_URL)` has already run
  // by the time this await resolves (whether it ran on THIS import or an earlier one in the
  // process -- see the file comment), so "perm-graph" is already taken. Re-registering it under
  // that same name is exactly the regression this guards: with the duplicate check removed, this
  // second call would silently replace the first widget's registration instead of throwing.
  assert.throws(
    () => register("perm-graph", (() => {}) as Widget<Nothing>, NOTHING),
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
  //
  // Their URL keys are distinct for the same reason their names are: both are mounted in the one
  // mountAll call below, and two mount points claiming one key is exactly what mountAll throws on --
  // which would turn this test's second widget into a collision failure rather than the successful
  // mount it is asserting.
  const failing: Widget<Nothing> = () => { throw new Error("forced synchronous failure"); };
  const okay: Widget<Nothing> = (host) => {
    (host as unknown as FakeMountPoint).append({ textContent: "isolation-ok" });
  };
  register("isolation-fail-widget", failing, { on: boolParam("isolation-fail-on") });
  register("isolation-ok-widget", okay, { on: boolParam("isolation-ok-on") });

  const failEl = makeMountPoint("isolation-fail-widget");
  const okEl = makeMountPoint("isolation-ok-widget");
  const root = { querySelectorAll: () => [failEl, okEl] };
  mountAll(root as unknown as ParentNode, noStore);

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

test("every widget name a generated page can emit is registered under exactly that string",
  async () => {
    stubDocument();
    const { register } = await import("../src/mount.js");

    // The one link in the chain with no test on either side of it (Task 7 review finding I2): the
    // generator emits `data-widget="…"` strings, mount.ts registers strings, and nothing paired them
    // -- deleting `register("frontier", frontier)` left the whole suite green while the deployed page
    // showed a console-only error behind an intact PNG. Re-registering must throw, which it can only
    // do if the name is already taken by the real registration in mount.ts's own module body.
    //
    // The names are READ OUT OF THE GENERATOR, not listed here, because the name of this test
    // promises "every widget name a generated page can emit" and a hardcoded pair would not keep that
    // promise for a third widget (final review, I4). Same derivation as widgets-bundle.test.ts, which
    // pins the same property against the shipped artifact rather than the source modules.
    const generator = readFileSync("../scripts/gen_site_pages.py", "utf8");
    const emitted = [...new Set([...generator.matchAll(/data-widget="([a-z-]+)"/g)]
      .map((m) => m[1]!))].sort();
    assert.ok(emitted.length >= 2, `generator emits too few widget names: ${JSON.stringify(emitted)}`);
    for (const name of emitted) {
      assert.throws(
        () => register(name, (() => {}) as Widget<Nothing>, NOTHING),
        new RegExp(`widget already registered: ${name}`),
        `"${name}" is not registered, so a page emitting it mounts nothing`,
      );
    }
  });

test("an unknown widget name renders a visible error rather than a console-only throw",
  async () => {
    stubDocument();
    const { mountAll, register } = await import("../src/mount.js");

    // M7: this throw used to sit OUTSIDE mountAll's per-element try/catch, so the one failure mode
    // that also skips every later mount point was the only one the reader never heard about. It is
    // also what made I2 invisible rather than merely untested.
    // A registered widget AFTER the unknown one, so this also pins that an unknown name no longer
    // aborts the loop. Registered here under a name unique to this test rather than reusing another
    // test's, so the two stay order-independent (node:test shares a module graph within a file).
    const okay: Widget<Nothing> = (host) => {
      (host as unknown as FakeMountPoint).append({ textContent: "mounted-after-unknown" });
    };
    register("m7-later-widget", okay, { on: boolParam("m7-later-on") });
    const el = makeMountPoint("no-such-widget");
    const later = makeMountPoint("m7-later-widget");
    mountAll({ querySelectorAll: () => [el, later] } as unknown as ParentNode, noStore);
    assert.ok(later.appended.some((n) => n.textContent === "mounted-after-unknown"),
      "an unknown name aborted the mount loop instead of failing only its own element");

    assert.ok(
      el.appended.some((n) => n.textContent.includes("could not load interactively")
        && n.textContent.includes("unknown data-widget: no-such-widget")),
      `unknown name produced no on-page message: ${JSON.stringify(el.appended)}`,
    );
  });

test("two mount points claiming one URL key throw, with the message ON THE PAGE", async () => {
  stubDocument();
  const { register, mountAll } = await import("../src/mount.js");

  // One codec, two widgets: the collision mountAll has to catch. Two mount points sharing a query
  // key would read and write one set of values through two independent state sources, so whichever
  // wrote last would silently move the other figure.
  interface A { v: boolean }
  const SHARED: UrlCodec<A> = { v: boolParam("shared-key") };
  register("collide-a", (() => {}) as Widget<A>, SHARED);
  register("collide-b", (() => {}) as Widget<A>, SHARED);
  const a = makeMountPoint("collide-a");
  const b = makeMountPoint("collide-b");
  const root = { querySelectorAll: () => [a, b] } as unknown as ParentNode;
  mountAll(root, noStore);

  // The FIRST mount point is fine; the second is the one that collides, and its failure must be
  // rendered where it happened rather than aborting the page or landing only in the console.
  assert.equal(b.appended.length, 1);
  assert.match(b.appended[0]!.textContent, /shared-key.*collide-a.*collide-b/s);
});

test("a codec that maps two state fields to the SAME URL key throws, with the message ON THE PAGE",
  async () => {
    stubDocument();
    const { register, mountAll } = await import("../src/mount.js");

    // One widget, one codec, two fields sharing a key -- the collision `own` (mount.ts) exists to
    // catch. The two-pass check-then-commit split fix round 2 (M2) made to stop a phantom claim
    // silently lost this property (fix round 3, M-self): with keys checked only against `claimed`,
    // which holds nothing for THIS widget until its own list clears, a key repeated within one
    // codec would sail through, and the two fields would overwrite each other's value in the URL on
    // every write -- output of exactly the right shape, no error anywhere.
    interface SelfCollide { v: boolean; w: boolean }
    const SELF_COLLIDE: UrlCodec<SelfCollide> = {
      v: boolParam("dup-key"),
      w: boolParam("dup-key"),
    };
    register("self-collide", (() => {}) as Widget<SelfCollide>, SELF_COLLIDE);
    const el = makeMountPoint("self-collide");
    const root = { querySelectorAll: () => [el] } as unknown as ParentNode;
    mountAll(root, noStore);

    // A message distinct from the cross-widget collision above ("claimed by both X and X" would
    // read like a typo, not a bug report) -- see mount.ts's own comment on the throw.
    assert.equal(el.appended.length, 1);
    assert.match(el.appended[0]!.textContent, /dup-key.*self-collide.*own codec/s);
  });

test("a widget's own key collision does not phantom-claim its OTHER keys for a later widget",
  async () => {
    stubDocument();
    const { register, mountAll } = await import("../src/mount.js");

    // W1 claims two keys. W2's key list is [kz, ky]: kz first (no collision), then ky, which
    // collides with W1 -- so under key-by-key commits (the bug fix round 2, M2, removed), kz would
    // already sit in `claimed` under W2's name by the time the throw on ky happens, even though
    // W2's own `widget.mount` never runs. W3's only key is kz: it must mount cleanly, since nothing
    // W2 holds is real.
    interface Two { v: boolean; w: boolean }
    interface One { v: boolean }
    const W1_CODEC: UrlCodec<Two> = { v: boolParam("phantom-kx"), w: boolParam("phantom-ky") };
    const W2_CODEC: UrlCodec<Two> = { v: boolParam("phantom-kz"), w: boolParam("phantom-ky") };
    const W3_CODEC: UrlCodec<One> = { v: boolParam("phantom-kz") };

    register("phantom-w1", (() => {}) as Widget<Two>, W1_CODEC);
    register("phantom-w2", (() => {}) as Widget<Two>, W2_CODEC);
    const w3: Widget<One> = (host) => {
      (host as unknown as FakeMountPoint).append({ textContent: "w3-mounted" });
    };
    register("phantom-w3", w3, W3_CODEC);

    const w1el = makeMountPoint("phantom-w1");
    const w2el = makeMountPoint("phantom-w2");
    const w3el = makeMountPoint("phantom-w3");
    const root = { querySelectorAll: () => [w1el, w2el, w3el] } as unknown as ParentNode;
    mountAll(root, noStore);

    // W2 shows its OWN collision, against W1, on the shared key "phantom-ky" -- unchanged
    // behaviour, asserted here so a future regression on this exact trace fails on the right line.
    assert.match(w2el.appended[0]!.textContent, /phantom-ky.*phantom-w1.*phantom-w2/s);
    // W3 must have mounted: "phantom-kz" was never actually claimed by W2, since W2's own
    // `widget.mount` never ran to claim it for real.
    assert.ok(w3el.appended.some((n) => n.textContent === "w3-mounted"),
      `W3 was blocked by a phantom claim: ${JSON.stringify(w3el.appended)}`);
  });

test("the URL key list is pinned -- a key rename breaks published links silently otherwise",
  async () => {
    stubDocument();
    const [{ PERM_GRAPH_URL }, { FRONTIER_URL }, { FIELD_URL }, { REGION_GROW_URL },
      { SCREEN_MAP_URL }] = await Promise.all([
      import("../src/widgets/perm-graph.js"), import("../src/widgets/frontier.js"),
      import("../src/widgets/displacement-field.js"), import("../src/widgets/region-grow.js"),
      import("../src/widgets/screen-map.js"),
    ]);
    const keysOf = (codec: object): string[] =>
      (Object.values(codec) as readonly { keys: readonly string[] }[])
        .flatMap((p) => [...p.keys]).sort();
    assert.deepEqual(keysOf(PERM_GRAPH_URL), ["halos", "layer", "prefix"]);
    assert.deepEqual(keysOf(FRONTIER_URL), ["disp", "method", "perm"]);
    assert.deepEqual(keysOf(FIELD_URL), ["road1", "road2", "road2on", "width"]);
    assert.deepEqual(keysOf(REGION_GROW_URL), ["budget", "seed"]);
    assert.deepEqual(keysOf(SCREEN_MAP_URL), ["city", "floor", "metric"]);

    // And the union is collision-free ACROSS widgets, which is the property mountAll's throw
    // enforces per page and this asserts for the shipped set as a whole.
    const all = [PERM_GRAPH_URL, FRONTIER_URL, FIELD_URL, REGION_GROW_URL, SCREEN_MAP_URL]
      .flatMap(keysOf);
    assert.equal(new Set(all).size, all.length, `duplicate URL key across widgets: ${all}`);
  });

test("the emitted query is in MOUNT order, whatever order the widgets get round to binding in",
  async () => {
    stubDocument();
    const { register, mountAll } = await import("../src/mount.js");

    // Design §2.2 rule 5 -- one view, one URL -- is the citability property this whole piece
    // exists to deliver, and it was false on any page with more than one widget until `register`
    // started reserving a slot at mount time: the store appended a contributor inside `bind`, and
    // every one of the five widgets binds from inside its own `fetch().then(boot)`. On Explore
    // that means screen-map, FIRST in the DOM, awaits both city tiers (7.33 MB) while frontier,
    // LAST in the DOM, has 0.02 MB -- so the emitted order came out roughly reversed, and need not
    // repeat between loads. url-store.test.ts pins the same property one layer down, on the slots
    // themselves;
    // what this adds is the tie from a slot's position to the DOM walk that took it.
    //
    // So neither widget binds while `mountAll` runs. Each stashes its `makeState` and is fired
    // afterwards, in reverse mount order -- which is what a slower first bundle does.
    interface One { v: boolean }
    const pending: (() => void)[] = [];
    const deferred: Widget<One> = (_host, makeState) => {
      pending.push(() => { makeState({ v: false }).set({ v: true }); });
    };
    register("order-first-widget", deferred, { v: boolParam("order-first-on") });
    register("order-second-widget", deferred, { v: boolParam("order-second-on") });

    const loc = fakeLocation("");
    const first = makeMountPoint("order-first-widget");
    const second = makeMountPoint("order-second-widget");
    const root = { querySelectorAll: () => [first, second] } as unknown as ParentNode;
    mountAll(root, urlStore(loc, writeNow));

    assert.equal(pending.length, 2,
      `both widgets must have mounted without binding: ${pending.length} of 2 deferred`);
    assert.deepEqual(loc.written, [], "nothing is bound yet, so nothing can have been written");
    pending[1]!();
    pending[0]!();
    assert.equal(loc.written.at(-1), "order-first-on=1&order-second-on=1",
      "the query came out in bind order (the network's) rather than mount order (the page's)");
  });
