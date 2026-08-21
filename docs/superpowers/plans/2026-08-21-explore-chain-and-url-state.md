# Explore chain + URL-as-state (piece E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every widget's state becomes citable in the URL, and `docs/explore.md` walks one block
(`ZAF.9.3.1_1_40972`) down all five pipeline stages.

**Architecture:** One `UrlStore` per page, built in `mountAll` over injected `UrlLocation` and
`Scheduler` seams. Each widget is registered together with a `UrlCodec<T>` — a **mapped type over its
own state interface**, so a missing or renamed field is a compile error rather than a silently
dropped query key. `register` captures `T` in a closure and stores a non-generic `Registration`, so
`REGISTRY` stays a plain map while the (widget, codec) pairing is checked at each call site.
`history.replaceState` only, behind a 300 ms trailing debounce.

**Tech Stack:** TypeScript 5.9 (no framework, no runtime deps), esbuild, Node's built-in test runner,
Python 3 stdlib + geopandas/matplotlib for the bakers, MkDocs Material.

**Spec:** `docs/superpowers/specs/2026-08-21-explore-chain-and-url-state-design.md`

## Global Constraints

- `scripts/gen_site_pages.py` stays **stdlib-only** and must **never** import `reblock`.
- `docs/js/` and `docs/assets/` are gitignored — **never stage them**.
- Generated bundles and their `.d.ts` are generated and committed, **never hand-edited**.
- No `# type: ignore`, no mypy excludes, no `eslint-disable`, no unreachable guards as fixes.
- Never reach into a closed, known-at-authoring-time set with a runtime string, position or count.
  Dynamic access over a genuinely open set has **no default** — it raises or returns an explicit
  sentinel the caller handles.
- No legacy-compatibility shims. Migrate the data and delete the old path.
- Every number in generated prose is read from an artifact, **never typed**.
- **Do not enable Material's `navigation.instant`** — `mkdocs.yml`'s `features:` list is off limits
  except for the nav entry in Task 8. Two documented failures fire the moment it is on; see
  `web/src/mount.ts`'s `DOMContentLoaded` comment and `web/src/dom/resize.ts`'s disposer comment.
- Fault injection is the acceptance criterion for every guard: break the thing it guards, observe
  RED, restore. An injection that will not redden is **reported**, not tuned.
- Web tests run with `cd web && npm test` (which builds the esbuild bundle first). Type-check with
  `cd web && npm run check`. Python tests run with `pixi run test`; lint with `pixi run lint`.
- This machine runs **Node v24.12.0**, whose default `node --test` reporter is `spec`, not TAP:
  the summary lines read `ℹ pass 139` / `ℹ fail 0`, **not** `# pass 139`. Every grep in this plan
  is written as `grep -E "(pass|fail) [0-9]+"`, which matches either. Verify any pattern you
  substitute against a GREEN baseline first, so a grep that matches nothing is not mistaken for a
  suite that passed.

---

## File Structure

**Create**

| File | Responsibility |
| --- | --- |
| `web/src/url/param.ts` | `Param<V>`, `UrlCodec<T>`, and the seven primitives that build them |
| `web/src/url/store.ts` | `UrlLocation`, `browserLocation`, `Timers`, `systemTimers`, `Scheduler`, `debounce`, `UrlStore`, `urlStore` |
| `web/test/url-param.test.ts` | round-trips, garbage rejection, default omission |
| `web/test/url-store.test.ts` | bind/emit, preservation, self-correction, emission order, debounce |
| `docs/_partials/explore.md` | the walkthrough prose (tracked; markers filled by the generator) |

**Modify**

| File | Change |
| --- | --- |
| `web/src/state.ts` | `StateFactory<T>` narrowed from a universally-quantified factory |
| `web/src/mount.ts` | `Widget<T>`, `register(name, w, codec)`, `Registration`, store construction, URL-key collision throw |
| `web/src/widgets/perm-graph.ts` | export state, codec, prefix clamp |
| `web/src/widgets/frontier.ts` | export state, codec, `isolated` reset |
| `web/src/widgets/displacement-field.ts` | export state, codec, width-slider initial |
| `web/src/widgets/region-grow.ts` | export state, codec, `seed` → block_id, budget-slider initial |
| `web/src/widgets/screen-map.ts` | export state, codec, `floor: number \| null`, `defaultFloorFor`, the follow ring |
| `web/src/render/city.ts` | `paintFrame` draws the follow ring |
| `web/src/screen_map.d.ts` | regenerated: `CityFollow`, `follow?`, `follow_color` |
| `web/test/mount.test.ts` | `register`'s third argument; the collision test |
| `web/test/{region-grow,screen-map,field,frontier,perm-graph}-boot.test.ts` | the new guards |
| `scripts/gen_screen_map.py` | bake `follow` + `follow_color`; draw the ring in the PNG |
| `scripts/gen_site_pages.py` | `_perm_graph_widget_figure`, two new markers, the Explore page, the widget-bundle guard's page list |
| `docs/stylesheets/sbu.css` | `.sbu-stage-rail` |
| `mkdocs.yml` | one nav entry |
| `.gitignore` | `docs/explore.md` |
| `tests/test_screen_map_bundle.py` | the follow bake's guards |
| `tests/test_gen_site_pages.py` | the Explore page's mount points and path rewriting |

---

### Task 1: The codec vocabulary — `web/src/url/param.ts`

**Files:**
- Create: `web/src/url/param.ts`
- Test: `web/test/url-param.test.ts`

**Interfaces:**
- Consumes: `Road` from `web/src/field.d.ts` (`{ coords: [number, number][]; width_m: number }`).
- Produces: `Param<V>`, `UrlCodec<T>`, `intParam`, `numberParam`, `nullableNumberParam`, `boolParam`,
  `enumParam`, `stringParam`, `nullableStringParam`, `roadsParam`.

**Contract refinement over the spec:** §2.1 says an empty `encode` result "means a defect". That is
true for the scalar params and **false** for `nullableNumberParam`/`nullableStringParam`, where
`null` is legitimately spelled by the *absence* of the key. The contract implemented here is
therefore "`encode` returns the keys to emit for this value"; `{}` is a valid answer for a null.
Write that in the docstring.

- [ ] **Step 1: Write the failing test**

Create `web/test/url-param.test.ts`:

```ts
import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  boolParam, enumParam, intParam, nullableNumberParam, nullableStringParam,
  numberParam, roadsParam, stringParam,
} from "../src/url/param.js";
import type { Road } from "../src/field.js";

test("intParam round-trips and rejects everything that is not a non-negative integer", () => {
  const p = intParam("prefix");
  assert.deepEqual(p.keys, ["prefix"]);
  assert.deepEqual(p.encode(14, 0), { prefix: "14" });
  assert.equal(p.decode({ prefix: "14" }, 0), 14);
  for (const bad of ["", " ", "1.5", "-1", "abc", "NaN", "Infinity", "1e3x"]) {
    assert.equal(p.decode({ prefix: bad }, 0), null, bad);
  }
});

test("numberParam keeps six significant figures and rejects non-finite input", () => {
  const p = numberParam("floor", 6);
  assert.deepEqual(p.encode(0.012799999, 0), { floor: "0.0128" });
  assert.equal(p.decode({ floor: "0.0128" }, 0), 0.0128);
  for (const bad of ["", "NaN", "Infinity", "-Infinity", "0x10", "abc"]) {
    assert.equal(p.decode({ floor: bad }, 0), null, bad);
  }
});

test("nullableNumberParam spells null as the ABSENT key, both ways", () => {
  const p = nullableNumberParam("floor", 6);
  assert.deepEqual(p.encode(null, null), {});
  assert.deepEqual(p.encode(0.02, null), { floor: "0.02" });
  assert.equal(p.decode({ floor: "0.02" }, null), 0.02);
  assert.ok(p.same(null, null));
  assert.ok(!p.same(null, 0.02));
});

test("boolParam is 0/1 and refuses anything else -- not a truthiness test", () => {
  const p = boolParam("halos");
  assert.deepEqual(p.encode(false, true), { halos: "0" });
  assert.equal(p.decode({ halos: "1" }, false), true);
  assert.equal(p.decode({ halos: "0" }, true), false);
  for (const bad of ["true", "yes", "", "2", "maybe"]) {
    assert.equal(p.decode({ halos: bad }, false), null, bad);
  }
});

test("enumParam admits exactly its declared members", () => {
  const p = enumParam("city", ["capetown", "nairobi"] as const);
  assert.equal(p.decode({ city: "nairobi" }, "capetown"), "nairobi");
  assert.equal(p.decode({ city: "kampala" }, "capetown"), null);
  // A prototype key must not be admitted by an `in`-style membership test.
  assert.equal(p.decode({ city: "toString" }, "capetown"), null);
});

test("nullableStringParam round-trips a slug and spells null as the absent key", () => {
  const p = nullableStringParam("method");
  assert.deepEqual(p.encode(null, null), {});
  assert.deepEqual(p.encode("clearance", null), { method: "clearance" });
  assert.equal(p.decode({ method: "clearance" }, null), "clearance");
  assert.equal(p.decode({ method: "" }, null), null);
});

test("stringParam round-trips a block id and rejects the empty string", () => {
  const p = stringParam("seed");
  assert.deepEqual(p.encode("ZAF.9.3.1_1_40972", "X"), { seed: "ZAF.9.3.1_1_40972" });
  assert.equal(p.decode({ seed: "ZAF.9.3.1_1_40972" }, "X"), "ZAF.9.3.1_1_40972");
  assert.equal(p.decode({ seed: "" }, "X"), null);
});

const ROADS: Road[] = [
  { coords: [[132.53, 3.82], [40.24, 113.92]], width_m: 7 },
  { coords: [[101.81, 7.77], [26.28, 97.88]], width_m: 7 },
];

test("roadsParam emits ONLY the sub-keys that changed", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.deepEqual(p.keys, ["road1", "road2", "width"]);
  const wider = ROADS.map((r) => ({ ...r, width_m: 12 }));
  assert.deepEqual(p.encode(wider, ROADS), { width: "12" });
  const moved = [{ ...ROADS[0]!, coords: [[1, 2], [3, 4]] as [number, number][] }, ROADS[1]!];
  assert.deepEqual(p.encode(moved, ROADS),
                   { road1: "1,2,3,4", road2: "101.8,7.8,26.3,97.9" });
});

test("roadsParam decodes a width-only URL against the initial geometry", () => {
  const p = roadsParam("road1", "road2", "width");
  const got = p.decode({ width: "12" }, ROADS);
  assert.deepEqual(got?.map((r) => r.width_m), [12, 12]);
  assert.deepEqual(got?.[0]!.coords, ROADS[0]!.coords);
});

test("roadsParam refuses a half-specified pair and any malformed segment", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.equal(p.decode({ road1: "1,2,3,4" }, ROADS), null, "road2 missing");
  assert.equal(p.decode({ road1: "1,2,3", road2: "1,2,3,4" }, ROADS), null, "three numbers");
  assert.equal(p.decode({ road1: "1,2,3,x", road2: "1,2,3,4" }, ROADS), null, "not a number");
  assert.equal(p.decode({ width: "0" }, ROADS), null, "non-positive width");
  assert.equal(p.decode({ width: "-3" }, ROADS), null, "negative width");
});

test("roadsParam.same ignores object identity and sees a moved vertex", () => {
  const p = roadsParam("road1", "road2", "width");
  assert.ok(p.same(ROADS, ROADS.map((r) => ({ coords: [...r.coords], width_m: r.width_m }))));
  assert.ok(!p.same(ROADS, [{ ...ROADS[0]!, coords: [[0, 0], [1, 1]] }, ROADS[1]!]));
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npm test 2>&1 | tail -30`
Expected: FAIL — `Cannot find module '../src/url/param.js'`.

- [ ] **Step 3: Write `web/src/url/param.ts`**

```ts
/** The query-string vocabulary: how one field of one widget's state is spelled in a URL.
 *
 * A `Param` owns its KEYS rather than deriving them from the field name, so renaming a TypeScript
 * field cannot break a URL somebody published. Usually one key; `roadsParam` owns three, because
 * the corridor width is the knob a reader is most likely to hand-edit and deserves its own.
 */
import type { Road } from "../field.js";

export interface Param<V> {
  /** Every query key this param reads and writes. */
  readonly keys: readonly string[];

  /** The keys to emit for `v`, given the widget's own `initial`. Only the ones that DIFFER need
   * appear, which is what keeps a width-only change to `?width=12` instead of the full geometry.
   * `{}` is a legitimate answer for a nullable param whose value is `null`: null is spelled by the
   * key's ABSENCE, and there is nothing to emit. */
  encode(v: V, initial: V): Record<string, string>;

  /** `present` carries only THIS param's keys that the URL actually had -- possibly a subset, so a
   * hand-edited `?width=9` alone is decodable against `initial`.
   *
   * `null` means "the URL said something this widget cannot use". The store then falls back to
   * `initial` and DROPS every key of this param, so the reader watches their typo disappear rather
   * than being handed a broken figure or, worse, a plausible wrong one. */
  decode(present: Readonly<Record<string, string>>, initial: V): V | null;

  /** Value equality. Declared per param rather than `===` because `roads` is an array of objects
   * and identity comparison would rewrite the URL on every render. */
  same(a: V, b: V): boolean;
}

/** One `Param` per field of `T`, checked by the compiler.
 *
 * `-?` strips optionality, so an optional state field still needs a codec entry; adding a field to
 * a state interface without adding one here is a compile error, and renaming one is a compile
 * error at this object rather than a query key that silently stops being written. */
export type UrlCodec<T> = { readonly [K in keyof T]-?: Param<T[K]> };

/** ONE spelling of a number, and nothing else -- the same reasoning `boolParam` carries for 0/1.
 *
 * The regex is not belt-and-braces over `Number.isFinite`: `Number("0x10")` is 16 and finite, and
 * `Number("")` is 0, so a bare `Number()` admits hex and the empty string as valid readings of
 * `?floor=`. Both matter -- `?prefix=` with nothing after it is a real thing a reader can type, and
 * a URL grammar whose acceptance depends on which spelling somebody guessed is not a grammar. */
const DECIMAL = /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/;

function finite(raw: string): number | null {
  if (!DECIMAL.test(raw.trim())) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/** `sig` significant figures, then back through `Number` so 0.0128000 prints as 0.0128. Significant
 * figures rather than decimal places because these values span metric scales that differ by orders
 * of magnitude (`density_compactness` floors sit near 3.55e-4, `budget` near 3e3). */
function sigfig(v: number, sig: number): string {
  return String(Number(v.toPrecision(sig)));
}

export function intParam(key: string): Param<number> {
  return {
    keys: [key],
    encode: (v) => ({ [key]: String(v) }),
    decode: (present) => {
      const n = finite(present[key]!);
      return n !== null && Number.isInteger(n) && n >= 0 ? n : null;
    },
    same: (a, b) => a === b,
  };
}

export function numberParam(key: string, sig: number): Param<number> {
  return {
    keys: [key],
    encode: (v) => ({ [key]: sigfig(v, sig) }),
    decode: (present) => finite(present[key]!),
    same: (a, b) => a === b,
  };
}

export function nullableNumberParam(key: string, sig: number): Param<number | null> {
  return {
    keys: [key],
    encode: (v) => (v === null ? {} : { [key]: sigfig(v, sig) }),
    decode: (present) => finite(present[key]!),
    same: (a, b) => a === b,
  };
}

/** `0`/`1`, and NOTHING else -- deliberately not a truthiness test. "true", "yes" and "" would all
 * be plausible spellings, and admitting some of them silently would make the URL's grammar depend
 * on which one a reader guessed. One spelling, everything else refused and dropped. */
export function boolParam(key: string): Param<boolean> {
  return {
    keys: [key],
    encode: (v) => ({ [key]: v ? "1" : "0" }),
    decode: (present) => {
      const raw = present[key]!;
      return raw === "1" ? true : raw === "0" ? false : null;
    },
    same: (a, b) => a === b,
  };
}

/** Membership by `includes` over the DECLARED list, never `in` on an object: `"toString" in obj` is
 * true for every object, so an `in`-based membership test admits prototype keys as valid enum
 * members. */
export function enumParam<V extends string>(key: string, values: readonly V[]): Param<V> {
  return {
    keys: [key],
    encode: (v) => ({ [key]: v }),
    decode: (present) => {
      const raw = present[key]!;
      return values.includes(raw as V) ? (raw as V) : null;
    },
    same: (a, b) => a === b,
  };
}

export function stringParam(key: string): Param<string> {
  return {
    keys: [key],
    encode: (v) => ({ [key]: v }),
    decode: (present) => (present[key]! === "" ? null : present[key]!),
    same: (a, b) => a === b,
  };
}

export function nullableStringParam(key: string): Param<string | null> {
  return {
    keys: [key],
    encode: (v) => (v === null ? {} : { [key]: v }),
    decode: (present) => (present[key]! === "" ? null : present[key]!),
    same: (a, b) => a === b,
  };
}

/** Coordinate precision: 0.1 m. The bundle ships 2 dp; 10 cm is far below anything visible in a
 * drag, halves the URL's length, and is idempotent after one round trip. */
const COORD_DP = 1;

function segment(raw: string): [[number, number], [number, number]] | null {
  const parts = raw.split(",");
  if (parts.length !== 4) return null;
  const n = parts.map(finite);
  if (n.some((v) => v === null)) return null;
  return [[n[0]!, n[1]!], [n[2]!, n[3]!]];
}

/** `String(Number(v.toFixed(...)))`, never a bare `toFixed`: `(1).toFixed(1)` is "1.0", and a URL
 * whose coordinates all carry a trailing ".0" is longer for nothing. The width key below spells
 * itself the same way, so one param never emits two number formats. */
function fmtCoord(v: number): string {
  return String(Number(v.toFixed(COORD_DP)));
}

function fmtSegment(r: Road): string {
  const [a, b] = [r.coords[0]!, r.coords[1]!];
  return [a[0], a[1], b[0], b[1]].map(fmtCoord).join(",");
}

function sameCoords(a: [number, number][], b: [number, number][]): boolean {
  return a.length === b.length && a.every((p, i) => p[0] === b[i]![0] && p[1] === b[i]![1]);
}

/** DisplacementField's two roads and their shared width, over three keys.
 *
 * Exactly two roads of exactly two points each is an invariant of the bundle that
 * `displacement-field.ts`'s `boot` already validates and `liveIndices` already relies on with
 * literal 0/1. This param asserts the same and returns `null` otherwise, rather than inventing a
 * geometry the widget cannot draw.
 *
 * One shared width, because `displacement-field.ts`'s width slider writes its value onto EVERY
 * road's `width_m` -- deliberately, so the overlap demonstration stays exact. */
export function roadsParam(k1: string, k2: string, kWidth: string): Param<Road[]> {
  const keys = [k1, k2, kWidth] as const;
  return {
    keys,
    encode(v, initial) {
      const out: Record<string, string> = {};
      const movedGeometry = v.some((r, i) => !sameCoords(r.coords, initial[i]!.coords));
      if (movedGeometry) {
        out[k1] = fmtSegment(v[0]!);
        out[k2] = fmtSegment(v[1]!);
      }
      if (v[0]!.width_m !== initial[0]!.width_m) out[kWidth] = fmtCoord(v[0]!.width_m);
      return out;
    },
    decode(present, initial) {
      if (initial.length !== 2) return null;
      const has1 = present[k1] !== undefined;
      const has2 = present[k2] !== undefined;
      // A half-specified pair is not a state: one road from the URL and one from the bundle is a
      // picture nobody asked for.
      if (has1 !== has2) return null;
      let coords: [[number, number], [number, number]][] | null = null;
      if (has1 && has2) {
        const a = segment(present[k1]!);
        const b = segment(present[k2]!);
        if (a === null || b === null) return null;
        coords = [a, b];
      }
      let width: number | null = null;
      if (present[kWidth] !== undefined) {
        width = finite(present[kWidth]!);
        if (width === null || width <= 0) return null;
      }
      return initial.map((r, i) => ({
        coords: coords === null ? r.coords : coords[i]!,
        width_m: width ?? r.width_m,
      }));
    },
    same(a, b) {
      return a.length === b.length
        && a.every((r, i) => r.width_m === b[i]!.width_m && sameCoords(r.coords, b[i]!.coords));
    },
  };
}
```

- [ ] **Step 4: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | tail -20`
Expected: type-check clean; all `url-param` tests PASS.

- [ ] **Step 5: Fault injection**

For each, make the edit, run `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`, confirm a
non-zero fail count, then **restore**:
1. In `finite`, delete the `raw.trim() === ""` line ⇒ `intParam`'s `""` case must redden.
2. In `enumParam.decode`, swap `values.includes(raw as V)` for `raw in values` ⇒ the `toString` case
   must redden.
3. In `roadsParam.decode`, delete the `if (has1 !== has2) return null;` line ⇒ the half-pair case
   must redden.
4. In `roadsParam.encode`, always emit both segments (drop the `movedGeometry` guard) ⇒ the
   width-only case must redden.
5. In `fmtCoord`, return `v.toFixed(COORD_DP)` directly ⇒ the moved-segment case must redden on
   `"1.0,2.0,3.0,4.0"` vs `"1,2,3,4"`.

Record each result in the task report. An injection that will not redden is reported, not tuned.

- [ ] **Step 6: Commit**

```bash
git add web/src/url/param.ts web/test/url-param.test.ts
git commit -m "feat(web): the URL codec vocabulary -- Param<V> and UrlCodec<T>"
```

---

### Task 2: The store — `web/src/url/store.ts`

**Files:**
- Create: `web/src/url/store.ts`
- Test: `web/test/url-store.test.ts`

**Interfaces:**
- Consumes: `Param<V>`, `UrlCodec<T>` (Task 1); `StateSource<T>` from `web/src/state.ts`.
- Produces:
  ```ts
  export interface UrlLocation { search(): string; replace(search: string): void }
  export function browserLocation(): UrlLocation;
  export interface Timers { set(fn: () => void, ms: number): number; clear(id: number): void }
  export const systemTimers: Timers;
  export type Scheduler = (write: () => void) => void;
  export function debounce(ms: number, timers: Timers): Scheduler;
  export interface UrlStore { bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T> }
  export function urlStore(loc: UrlLocation, schedule: Scheduler): UrlStore;
  ```

- [ ] **Step 1: Write the failing test**

Create `web/test/url-store.test.ts`:

```ts
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { boolParam, enumParam, intParam, type UrlCodec } from "../src/url/param.js";
import { debounce, urlStore } from "../src/url/store.js";
import { fakeLocation, fakeTimers, writeNow } from "./harness.js";

interface Demo { prefix: number; layer: "conductance" | "current"; halos: boolean }
const DEMO: UrlCodec<Demo> = {
  prefix: intParam("prefix"),
  layer: enumParam("layer", ["conductance", "current"] as const),
  halos: boolParam("halos"),
};
const INITIAL: Demo = { prefix: 0, layer: "current", halos: true };

`fakeLocation` and `fakeTimers` go in **`web/test/harness.ts`**, not in this file: every boot test in
Tasks 4, 5 and 7 needs `fakeLocation` too, and the spec's own rule is that E's store is tested
through the shared harness rather than through a new fake per file. Add to `harness.ts`:

```ts
/** A `UrlLocation` over a plain string, plus every search string it was ever asked to write. Lives
 * here rather than in one test file because every URL-aware boot test needs it -- the same reason
 * `mountPoint` and `canvasOf` are here. */
export function fakeLocation(search: string): UrlLocation & { written: string[] } {
  const written: string[] = [];
  let current = search;
  return {
    written,
    search: () => current,
    replace(next) { current = next; written.push(next); },
  };
}

/** A `Timers` whose queue a test drives by hand -- nothing fires on its own, exactly as
 * `requestAnimationFrame`'s stub above queues rather than fires. */
export function fakeTimers(): Timers & { run(): void; pending(): number } {
  const queue = new Map<number, () => void>();
  let next = 0;
  return {
    set(fn) { queue.set(++next, fn); return next; },
    clear(id) { queue.delete(id); },
    pending: () => queue.size,
    run() {
      const fns = [...queue.values()];
      queue.clear();
      for (const fn of fns) fn();
    },
  };
}

/** Writes immediately. The debounce is tested on its own; a test that had to drive a timer queue to
 * observe a URL write would be testing two things at once. */
export const writeNow: Scheduler = (write) => { write(); };
```

and in `web/test/url-store.test.ts` import them from `./harness.js`:


test("an absent key leaves the widget's own initial alone, and writes nothing", () => {
  const loc = fakeLocation("");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.deepEqual(s.get(), INITIAL);
  assert.deepEqual(loc.written, []);
});

test("a present key overrides the initial and survives a write unchanged", () => {
  const loc = fakeLocation("prefix=14&layer=conductance");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.equal(s.get().prefix, 14);
  assert.equal(s.get().layer, "conductance");
  s.set({ halos: false });
  assert.equal(loc.written.at(-1), "prefix=14&layer=conductance&halos=0");
});

test("only values DIFFERING from the initial are emitted", () => {
  const loc = fakeLocation("");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  s.set({ prefix: 9 });
  assert.equal(loc.written.at(-1), "prefix=9");
  s.set({ prefix: 0 });
  assert.equal(loc.written.at(-1), "", "back to the initial: the key goes away entirely");
});

test("unclaimed params are preserved verbatim, and keep their original order", () => {
  const loc = fakeLocation("utm_source=paper&prefix=3&ref=abc");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  s.set({ layer: "conductance" });
  assert.equal(loc.written.at(-1), "utm_source=paper&ref=abc&prefix=3&layer=conductance");
});

test("an unusable value self-corrects: the initial is used AND the key is dropped at once", () => {
  const loc = fakeLocation("prefix=-4&layer=current");
  const s = urlStore(loc, writeNow).bind(DEMO, INITIAL);
  assert.equal(s.get().prefix, 0, "the widget's own initial, not -4");
  assert.equal(loc.written.at(-1), "", "written without waiting for the reader to touch anything");
});

test("two bindings share one query string and one write", () => {
  interface Other { budget: number }
  const OTHER: UrlCodec<Other> = { budget: intParam("budget") };
  const loc = fakeLocation("");
  const store = urlStore(loc, writeNow);
  const a = store.bind(DEMO, INITIAL);
  const b = store.bind(OTHER, { budget: 3000 });
  a.set({ prefix: 2 });
  b.set({ budget: 5000 });
  assert.equal(loc.written.at(-1), "prefix=2&budget=5000");
});

test("subscribers fire on set, exactly like localState's", () => {
  const seen: number[] = [];
  const s = urlStore(fakeLocation(""), writeNow).bind(DEMO, INITIAL);
  s.subscribe((v) => seen.push(v.prefix));
  s.set({ prefix: 1 });
  s.set({ prefix: 2 });
  assert.deepEqual(seen, [1, 2]);
});

test("debounce collapses a burst into ONE write, carrying the last value", () => {
  const timers = fakeTimers();
  const seen: string[] = [];
  const schedule = debounce(300, timers);
  schedule(() => seen.push("a"));
  schedule(() => seen.push("b"));
  schedule(() => seen.push("c"));
  assert.equal(timers.pending(), 1, "the earlier two timers were cleared, not left queued");
  assert.deepEqual(seen, [], "nothing has run before the window elapses");
  timers.run();
  assert.deepEqual(seen, ["c"]);
});

test("a drag through the debounce writes once, not once per state change", () => {
  const timers = fakeTimers();
  const loc = fakeLocation("");
  const s = urlStore(loc, debounce(300, timers)).bind(DEMO, INITIAL);
  for (let i = 1; i <= 40; i++) s.set({ prefix: i });
  assert.deepEqual(loc.written, [], "not one write yet");
  timers.run();
  assert.deepEqual(loc.written, ["prefix=40"]);
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npm test 2>&1 | tail -30`
Expected: FAIL — `Cannot find module '../src/url/store.js'`.

- [ ] **Step 3: Write `web/src/url/store.ts`**

```ts
/** The URL as the page's state: one store per page, shared by every widget on it.
 *
 * `replaceState` only, never `pushState`. Back-to-undo-a-slider is not a behaviour anyone expects,
 * and skipping history entries is what lets four widgets keep their one-way "control writes state"
 * wiring instead of each growing a state-to-control write-back for `popstate` (design §2.2). It is
 * also what keeps this under Safari's `replaceState` rate limit -- roughly 100 calls per 30 s,
 * which a write per `pointermove` inside a single drag would exceed on its own.
 */
import type { StateSource } from "../state.js";
import type { Param, UrlCodec } from "./param.js";

/** The `location`/`history` seam. Injected rather than reached for, so the store is testable
 * without stubbing two globals, and so exactly one place in the codebase knows the browser API. */
export interface UrlLocation {
  search(): string;
  /** `search` is the query WITHOUT its leading "?"; "" means "no query at all". */
  replace(search: string): void;
}

export function browserLocation(): UrlLocation {
  return {
    search: () => window.location.search,
    replace: (search) => {
      history.replaceState(null, "", search === "" ? window.location.pathname : `?${search}`);
    },
  };
}

export interface Timers {
  set(fn: () => void, ms: number): number;
  clear(id: number): void;
}

export const systemTimers: Timers = {
  // The browser and Node disagree about setTimeout's return type (number vs Timeout); this seam's
  // contract is a number, and the cast is where that difference is absorbed -- once, here.
  set: (fn, ms) => setTimeout(fn, ms) as unknown as number,
  clear: (id) => { clearTimeout(id); },
};

export type Scheduler = (write: () => void) => void;

/** Trailing-edge debounce: nothing is written while changes keep arriving, and one write lands
 * `ms` after the last one. A drag therefore produces exactly one URL update, when it settles --
 * both cheaper and more correct than a leading-edge or every-Nth-frame write, which would publish
 * intermediate states nobody was ever looking at. */
export function debounce(ms: number, timers: Timers): Scheduler {
  let pending: number | null = null;
  let latest: (() => void) | null = null;
  return (write) => {
    latest = write;
    if (pending !== null) timers.clear(pending);
    pending = timers.set(() => {
      pending = null;
      const fn = latest;
      latest = null;
      if (fn !== null) fn();
    }, ms);
  };
}

export interface UrlStore {
  bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T>;
}

/** Percent-encoding that leaves "," alone. RFC 3986 lists "," as a sub-delim that is legal in a
 * query, and `roadsParam` puts four of them in every segment: `URLSearchParams.toString()` would
 * spell `road1=132.5%2C3.8%2C40.2%2C113.9`, which is correct and unreadable, in a URL whose entire
 * purpose is to be pasted into a review. */
function enc(s: string): string {
  return encodeURIComponent(s).replace(/%2C/g, ",");
}

export function urlStore(loc: UrlLocation, schedule: Scheduler): UrlStore {
  // Parsed ONCE, at construction. A later widget binding must see the same query the first one did,
  // even though the store has by then rewritten `location` -- otherwise the second widget would
  // read a URL from which the first widget's defaults had already been pruned.
  const arrived: [string, string][] = [...new URLSearchParams(loc.search())];
  const claimed = new Set<string>();
  const contributors: (() => Record<string, string>)[] = [];

  const write = (): void => {
    const parts: string[] = [];
    // Unclaimed first, in the order they arrived -- someone else's `utm_source` or `ref` is not
    // this store's to reorder or discard.
    for (const [k, v] of arrived) {
      if (!claimed.has(k)) parts.push(`${enc(k)}=${enc(v)}`);
    }
    // Then everything the widgets on this page contribute, in mount order and then in
    // codec-declaration order -- deterministic, so the same view always produces the same URL.
    for (const emit of contributors) {
      for (const [k, v] of Object.entries(emit())) parts.push(`${enc(k)}=${enc(v)}`);
    }
    loc.replace(parts.join("&"));
  };

  return {
    bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T> {
      // `Object.keys` over a DECLARED mapped type: a loop over a schema, which is the allowed form
      // of dynamic access -- not a string lookup into a closed set. TypeScript cannot express the
      // per-key existential, so the two casts below are the standard cost of iterating a mapped
      // type; nothing outside this loop deals in `keyof T` strings.
      const fields = Object.keys(codec as object) as (keyof T)[];
      const present = new Map(arrived);
      let current = { ...initial };
      let dropped = false;

      for (const field of fields) {
        const p = codec[field] as Param<T[keyof T]>;
        for (const k of p.keys) claimed.add(k);
        const got: Record<string, string> = {};
        for (const k of p.keys) {
          const v = present.get(k);
          if (v !== undefined) got[k] = v;
        }
        if (Object.keys(got).length === 0) continue;
        const decoded = p.decode(got, initial[field]);
        // No default. An unusable value falls back to the widget's own initial AND is dropped, so
        // the URL corrects itself in front of the reader rather than carrying a lie.
        if (decoded === null) { dropped = true; continue; }
        current[field] = decoded;
      }

      contributors.push(() => {
        const out: Record<string, string> = {};
        for (const field of fields) {
          const p = codec[field] as Param<T[keyof T]>;
          if (p.same(current[field], initial[field])) continue;
          Object.assign(out, p.encode(current[field], initial[field]));
        }
        return out;
      });

      // Self-correction does not wait for the reader to touch a control: if anything was dropped,
      // the corrected URL is written now.
      if (dropped) schedule(write);

      const listeners: ((s: T) => void)[] = [];
      return {
        get: () => current,
        set(patch) {
          current = { ...current, ...patch };
          for (const fn of listeners) fn(current);
          schedule(write);
        },
        subscribe(fn) { listeners.push(fn); },
      };
    },
  };
}
```

- [ ] **Step 4: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | tail -20`
Expected: type-check clean; all `url-store` tests PASS.

- [ ] **Step 5: Fault injection**

Make the edit, run the suite, confirm RED, restore. Report each:
1. Delete `if (dropped) schedule(write);` ⇒ the self-correction test must redden.
2. In `write`, drop the `if (!claimed.has(k))` guard ⇒ the preservation test must redden (the
   claimed `prefix=3` would appear twice).
3. In `debounce`, remove `if (pending !== null) timers.clear(pending);` ⇒ the burst test must redden
   on `timers.pending()`.
4. In `bind`, replace the `p.same(...)` skip with `if (false)` ⇒ the default-omission test must
   redden.
5. Change `arrived` to be re-read from `loc.search()` inside `bind` ⇒ the two-bindings test must
   redden.

- [ ] **Step 6: Commit**

```bash
git add web/src/url/store.ts web/test/url-store.test.ts
git commit -m "feat(web): the page-shared URL store, over injected location and scheduler seams"
```

---

### Task 3: The mount contract and the five codecs

**Files:**
- Modify: `web/src/state.ts`, `web/src/mount.ts`
- Modify: `web/src/widgets/{perm-graph,frontier,displacement-field,region-grow,screen-map}.ts`
- Modify: `web/test/mount.test.ts`

**Interfaces:**
- Consumes: `UrlCodec<T>` and the primitives (Task 1); `urlStore`, `browserLocation`, `debounce`,
  `systemTimers`, `UrlStore` (Task 2).
- Produces:
  ```ts
  export type StateFactory<T> = (initial: T) => StateSource<T>;         // state.ts
  export type Widget<T> = (host: HTMLElement, makeState: StateFactory<T>) => void;   // mount.ts
  export function register<T>(name: string, w: Widget<T>, codec: UrlCodec<T>): void;
  export function mountAll(root?: ParentNode, store?: UrlStore): void;
  export const URL_DEBOUNCE_MS = 300;
  // and, from each widget module, its state interface plus:
  export const PERM_GRAPH_URL: UrlCodec<PermGraphState>;
  export const FRONTIER_URL: UrlCodec<FrontierState>;
  export const FIELD_URL: UrlCodec<FieldState>;
  export const REGION_GROW_URL: UrlCodec<RegionGrowState>;
  export const SCREEN_MAP_URL: UrlCodec<ScreenState>;
  ```

**Note for the implementer:** this task is deliberately atomic. `register`'s third argument is
required, so all five codecs must land in the same commit or nothing compiles. `RegionGrowState.seed`
stays a `number` in **this** task and becomes a block_id in Task 4 — do not change it here.
`ScreenState.floor` stays a `number` here and becomes `number | null` in Task 5.

- [ ] **Step 1: Narrow `StateFactory` in `web/src/state.ts`**

Replace line 17 (`export type StateFactory = <T>(initial: T) => StateSource<T>;`) with:

```ts
/** Parameterised by the state it makes, NOT universally quantified over it.
 *
 * The old `<T>(initial: T) => StateSource<T>` let the WIDGET pick `T` at its own call site, which
 * meant nothing could pair a widget with a description of its state -- any URL keying would have
 * had to be a runtime string table. With `T` on the type instead, `register(name, widget, codec)`
 * type-checks the pairing at each of the five call sites, and `UrlCodec<T>`'s mapped type makes a
 * missing or renamed field a compile error. `localState` is still generic, so it satisfies this at
 * every instantiation and every existing boot test keeps working unchanged. */
export type StateFactory<T> = (initial: T) => StateSource<T>;
```

**This task DISCHARGES A DEBT.** `web/src/url/store.ts`'s `arrived` comment (Task 2) justifies
parsing the query once by pointing at `mountAll`'s URL-key collision throw — the case where two
codecs claim one key is the only one where parse-once and re-read-per-bind differ, and that comment
says `mountAll` forbids it. As of Task 2 it does not: `mount.ts` is still the pre-Task-3 version.
Step 2 below is what makes that citation true. If the collision throw is dropped or weakened, go back
and correct `store.ts`'s comment in the same commit — a comment naming a guard that is nowhere is the
defect this branch has now caught three times.

- [ ] **Step 2: Rewrite the top of `web/src/mount.ts`**

Replace lines 1-44 (through the end of `showMountError`) with:

```ts
/** The mount contract: a page carries a placeholder and nothing else. */
import { showWidgetError } from "./dom/error.js";
import type { StateFactory } from "./state.js";
import type { Param, UrlCodec } from "./url/param.js";
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
  // `Object.values` over a DECLARED mapped type -- a loop over a schema, not a string lookup into a
  // closed set. Only `.keys` is read, so the value type is narrowed to exactly that.
  const keys = (Object.values(codec as object) as readonly Param<unknown>[])
    .flatMap((p) => [...p.keys]);
  REGISTRY.set(name, {
    keys,
    mount: (host, store) => { w(host, (initial) => store.bind(codec, initial)); },
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
      for (const k of widget.keys) {
        const prior = claimed.get(k);
        if (prior !== undefined) {
          throw new Error(`URL key "${k}" is claimed by both ${prior} and ${name} on this page`);
        }
        claimed.set(k, name);
      }
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
```

Then update the five registration lines at the bottom of the file to pass each widget's codec, e.g.:

```ts
import { permGraph, PERM_GRAPH_URL } from "./widgets/perm-graph.js";
register("perm-graph", permGraph, PERM_GRAPH_URL);
```

and the same shape for `frontier`/`FRONTIER_URL`, `displacementField`/`FIELD_URL`,
`regionGrow`/`REGION_GROW_URL`, `screenMap`/`SCREEN_MAP_URL`. **Keep every existing comment in that
block** — it records the module-evaluation cycle that once made the whole bundle throw. Keep the
`DOMContentLoaded` listener and its comment exactly as they are.

- [ ] **Step 3: Add the five codecs and export the five state interfaces**

`web/src/widgets/perm-graph.ts` — export the interface at line 14 and add, immediately below it:

```ts
export const PERM_GRAPH_URL: UrlCodec<PermGraphState> = {
  prefix: intParam("prefix"),
  layer: enumParam("layer", ["conductance", "current"] as const),
  halos: boolParam("halos"),
};
```

`web/src/widgets/frontier.ts` — export `FrontierState` (line 221) and add below it:

```ts
export const FRONTIER_URL: UrlCodec<FrontierState> = {
  // Short, readable keys, not the field names: `disp`/`perm` are what the parent design's own
  // example URL uses, and a field rename must not break a published link.
  targetDisplacement: numberParam("disp", 6),
  targetPermeability: numberParam("perm", 6),
  isolated: nullableStringParam("method"),
};
```

`web/src/widgets/displacement-field.ts` — export `FieldState` (line 21) and add below it:

```ts
export const FIELD_URL: UrlCodec<FieldState> = {
  roads: roadsParam("road1", "road2", "width"),
  second: boolParam("road2on"),
};
```

`web/src/widgets/region-grow.ts` — export `RegionGrowState` (line 22) and add below it:

```ts
export const REGION_GROW_URL: UrlCodec<RegionGrowState> = {
  seed: intParam("seed"),
  budget: intParam("budget"),
};
```

`web/src/widgets/screen-map.ts` — export `ScreenState` (line 27) and add below `METRIC_NAMES`:

```ts
export const SCREEN_MAP_URL: UrlCodec<ScreenState> = {
  city: enumParam("city", ["capetown", "nairobi"] as const),
  // METRIC_NAMES, not Object.keys(METRICS): the same closed, spelled-out list the <select> is built
  // from, so a fifth metric added to the model without a line there is a compile error in one place
  // rather than a silently short menu and a silently narrower URL grammar.
  metric: enumParam("metric", METRIC_NAMES),
  floor: numberParam("floor", 6),
};
```

Each widget also changes its exported const's annotation from `Widget` to `Widget<XState>` and its
`boot`'s parameter from `StateFactory` to `StateFactory<XState>`, and adds the imports it now needs
from `../url/param.js`.

- [ ] **Step 4: Update `web/test/mount.test.ts`**

`register`'s call in the duplicate-name test gains a codec:

```ts
import { boolParam, type UrlCodec } from "../src/url/param.js";
interface Nothing { on: boolean }
const NOTHING: UrlCodec<Nothing> = { on: boolParam("nothing-on") };
// ...
assert.throws(
  () => register("perm-graph", (() => {}) as Widget<Nothing>, NOTHING),
  /widget already registered: perm-graph/,
);
```

Add a new test for the collision throw, using the file's existing `makeMountPoint` helper and its
`stubDocument()` ritual:

```ts
test("two mount points claiming one URL key throw, with the message ON THE PAGE", async () => {
  stubDocument();
  const { register, mountAll } = await import("../src/mount.js");
  interface A { v: boolean }
  const SHARED: UrlCodec<A> = { v: boolParam("shared-key") };
  register("collide-a", (() => {}) as Widget<A>, SHARED);
  register("collide-b", (() => {}) as Widget<A>, SHARED);
  const a = makeMountPoint("collide-a");
  const b = makeMountPoint("collide-b");
  const root = { querySelectorAll: () => [a, b] } as unknown as ParentNode;
  // The stub must preserve the generic: `bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T>`
  // is a generic METHOD, and a non-generic arrow does not satisfy it.
  const noStore: UrlStore = {
    bind: <T>(_c: UrlCodec<T>, initial: T) =>
      ({ get: () => initial, set: () => {}, subscribe: () => {} }),
  };
  mountAll(root, noStore);
  // The FIRST mount point is fine; the second is the one that collides, and its failure must be
  // rendered where it happened rather than aborting the page or landing only in the console.
  assert.equal(b.appended.length, 1);
  assert.match(b.appended[0]!.textContent, /shared-key.*collide-a.*collide-b/s);
});
```

- [ ] **Step 5: Pin every widget's URL key list**

A key rename (`intParam("prefix")` → `intParam("prefixx")`) is **not** a compile error and would
silently break every published link. Add to `web/test/mount.test.ts`:

```ts
test("the URL key list is pinned -- a key rename breaks published links silently otherwise", async () => {
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
```

- [ ] **Step 6: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | tail -20`
Expected: type-check clean; every test PASSES, including the pre-existing boot tests untouched.

- [ ] **Step 7: Fault injection**

1. Delete the collision loop in `mountAll` ⇒ the new collision test must redden.
2. Delete one field from `SCREEN_MAP_URL` (e.g. `floor`) ⇒ `npm run check` must fail with a mapped-
   type error, not merely a test failure. Record the exact compiler message.
3. Rename `ScreenState.floor` to `gate` without touching the codec ⇒ `npm run check` must fail.

4. Rename one URL key (`intParam("prefix")` → `intParam("prefixx")`) ⇒ the key-list test must
   redden and `npm run check` must stay CLEAN — that gap is exactly why the test exists.

- [ ] **Step 8: Commit**

```bash
git add web/src/state.ts web/src/mount.ts web/src/widgets/*.ts web/test/mount.test.ts
git commit -m "feat(web): keyed mount contract -- register(name, widget, codec), Widget<T>"
```

---

### Task 4: Control initials and bundle-dependent resets

**Files:**
- Modify: `web/src/widgets/region-grow.ts`, `web/src/widgets/displacement-field.ts`,
  `web/src/widgets/perm-graph.ts`, `web/src/widgets/frontier.ts`
- Test: `web/test/region-grow-boot.test.ts`, `web/test/field-boot.test.ts`,
  `web/test/perm-graph-boot.test.ts`, `web/test/frontier-boot.test.ts`

**Interfaces:**
- Consumes: `REGION_GROW_URL` etc. (Task 3), `urlStore` (Task 2).
- Produces: `RegionGrowState.seed` is now a **block_id string**, so
  `REGION_GROW_URL.seed` becomes `stringParam("seed")`.

**Why these four are one task:** each is the same defect in a different widget — a value the URL can
now set that the widget's own DOM or model does not honour — and each is a handful of lines. A
reviewer would accept or reject them together.

- [ ] **Step 1: Write the failing tests**

In `web/test/region-grow-boot.test.ts`, add (the file already has a `mount the widget over a fake
bundle` helper; reuse it, and mount through a `urlStore` over a `fakeLocation` rather than
`localState`):

```ts
test("a URL budget reaches the SLIDER, not only the canvas", async () => {
  const { host, slider } = await mountWithSearch("budget=5000");
  assert.equal(slider.value, "5000",
    "the slider initialised from the bundle default while the picture drew 5000");
  void host;
});

test("the seed is a block_id, so a re-baked hood cannot silently reseed a published link", async () => {
  const { store } = await mountWithSearch(`seed=${FIXTURE_BLOCKS[2]!.block_id}`);
  assert.equal(store.get().seed, FIXTURE_BLOCKS[2]!.block_id);
});

test("a seed this hood does not carry falls back to the bundle's own AND leaves the URL", async () => {
  const { loc, store } = await mountWithSearch("seed=NOT_A_BLOCK");
  assert.equal(store.get().seed, FIXTURE_BUNDLE.seed);
  assert.equal(loc.written.at(-1), "", "the bad key is gone, not carried");
});
```

In `web/test/field-boot.test.ts`:

```ts
test("a URL width reaches the width SLIDER", async () => {
  const { widthSlider } = await mountWithSearch("width=12");
  assert.equal(widthSlider.value, "12");
});
```

In `web/test/perm-graph-boot.test.ts`:

```ts
test("a prefix past the last one is clamped, not drawn out of range", async () => {
  const { store, loc } = await mountWithSearch("prefix=99999");
  assert.equal(store.get().prefix, FIXTURE_BUNDLE.n_prefixes - 1);
  assert.ok(!loc.written.at(-1)!.includes("prefix=99999"));
});
```

In `web/test/frontier-boot.test.ts`:

```ts
test("an unknown ?method= draws EVERY curve, not an empty chart", async () => {
  const { cv, store } = await mountWithSearch("method=not_a_method");
  assert.equal(store.get().isolated, null);
  const strokes = lastFrame(cv).filter((c) => c.op === "stroke");
  assert.ok(strokes.length >= Object.keys(FIXTURE_BUNDLE.methods).length,
    "isolating a method that does not exist filtered out all of them");
});

test("a prototype key is not a method name", async () => {
  const { store } = await mountWithSearch("method=toString");
  assert.equal(store.get().isolated, null);
});
```

**Do not add a `mountWithSearch` helper.** Every boot test already has its own `mount(...)` — with
three different signatures (`mount(width, drawFailure)` in region-grow, `mount(host, drawFailure,
...)` in field, `mount(host, width, ...)` in perm-graph, `mount(host, payload)` in frontier,
`mount(width, drawFailure)` in screen-map) — and each already loads the **real committed bundle**
via `JSON.parse(readFileSync(...))`, not a synthetic fixture. Extend the existing helper instead:

1. Add `search: string = ""` as a new **trailing** optional parameter.
2. Replace the `localState` argument with a store bound to this widget's codec, and return the
   bound `StateSource` and the fake location alongside whatever the helper already returns:

```ts
// in the existing mount(...) helper, in place of the bare `localState` argument
const loc = fakeLocation(search);
const urls = urlStore(loc, writeNow);
let bound: StateSource<RegionGrowState> | null = null;
regionGrow(host as never, (initial) => {
  bound = urls.bind(REGION_GROW_URL, initial);
  return bound;
});
// ... the helper's existing "let the fetch chain settle" await and fireResize stay exactly as they
// are ...
assert.ok(bound !== null, "the widget never asked for a state store");
```

Defaulting `search` to `""` means every EXISTING test in the file now runs through the production
store rather than `localState` — deliberately. An empty query yields the widget's own initial state
and writes nothing, so they must all still pass; if one does not, that is a real finding about the
store, not a reason to branch the helper back to `localState`.

Substitute each file's own names: the region-grow fixture is `bundle`; screen-map's are `ct` and
`nb`; field, frontier and perm-graph each call theirs `bundle`. Never add a second fetch stub
alongside the one a file already installs.

- [ ] **Step 2: Run them to make sure they fail**

Run: `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: 7 failures — three region-grow, one field, one perm-graph, two frontier.

- [ ] **Step 3: `region-grow.ts` — `seed` becomes a block_id**

Change the interface (and its docstring's first sentence, which currently says `seed` indexes
`blocks`):

```ts
/** `seed` is a **block_id**, `budget` a building_count on the slider's own scale. `seed` is an
 * identity rather than a position deliberately: it is citable in a URL (piece E), and an array
 * index there would point at a DIFFERENT block after any re-bake that reorders `hood.json` -- no
 * error, right type, right shape, wrong value. Keeping both here rather than as loose widget
 * variables is what makes a click-then-slide sequence replay through the SAME `growth()` call every
 * render, instead of the picture and the model drifting apart across two pieces of mutable state. */
export interface RegionGrowState { seed: string; budget: number }
```

In `boot`, replace the `seed0` block with:

```ts
// block_id -> index, built once. `growth()` and `draw()` both take a POSITION (and `hood.json`'s
// `reference` fixtures pin accretion by index), so the conversion lives at the two boundaries
// rather than in the model.
const indexOf = new Map(blocks.map((blk, i) => [blk.block_id, i]));
// The bundle is a BOUNDARY: it arrives over the network, and a page can outlive the artifact it
// was generated beside. A `seed` that no longer names one of `blocks` would otherwise reach
// `growth()` as a negative index and fail far from here, with no message a reader could act on.
if (!indexOf.has(b.seed)) {
  throw new Error(`hood.json's seed "${b.seed}" is not among its own ${blocks.length} blocks`);
}
const state: StateSource<RegionGrowState> = makeState<RegionGrowState>({
  seed: b.seed, budget: b.budget.default,
});
// A URL (piece E) may name a block this hood does not carry. That is a reader's typo, not a broken
// artifact, so reset rather than throw -- and because the reset makes the field equal its initial,
// the store stops emitting `?seed=` and the URL self-corrects (design §2.3).
if (!indexOf.has(state.get().seed)) state.set({ seed: b.seed });
```

Change `REGION_GROW_URL.seed` to `stringParam("seed")`. Change `slider.value` to
`String(state.get().budget)`. In `render`, resolve the seed:

```ts
const s = state.get();
const seed = indexOf.get(s.seed)!;   // reset above guarantees membership before the first render
const g = growth(blocks, seed, s.budget);
const frontier = frontierOf(blocks, g.order);
draw(ctx, blocks, e, { view, region: g.order, frontier, seed }, size);
```

In the `pointerdown` handler, write the id: `state.set({ seed: blocks[hit]!.block_id });`

- [ ] **Step 4: `displacement-field.ts` — the width slider, and the segment invariant**

Change `slider.value = String(width0);` to:

```ts
// From STATE, not from `width0`: the bundle's default is what `makeState` was seeded with, but a
// URL (piece E) may have overridden it before this line, and a slider showing 7 while the corridor
// is drawn at 12 is the exact desync this widget's own state exists to prevent.
slider.value = String(state.get().roads[0]!.width_m);
```

**Then extend `boot`'s bundle validation.** This step also DISCHARGES A DEBT: Task 1's
`roadsParam` docstring points forward to this check, deliberately without claiming it already
exists (it does not, as of Task 1). If this step is dropped or renamed, that docstring becomes
a comment naming a guard that is nowhere — the same defect Task 1's review just fixed, one file
over. Re-read `web/src/url/param.ts`'s `roadsParam` docstring after landing this, and make its
forward reference true.

**Then extend `boot`'s bundle validation.** It currently checks the road *count* (exactly two) and
not the *points per road*, while `roadsParam` (Task 1) indexes `coords[0]`/`coords[1]` directly and
spells a road as `x1,y1,x2,y2`. Task 1's review found the gap: its own docstring claimed the codec
checked this, and it does not. Enforce it once, at the boundary the bundle actually crosses —
immediately after the existing `b.roads.length !== 2` throw, in the same idiom:

```ts
  // Each road is a two-point SEGMENT, for the same reason the count is exactly two: the drag
  // handles, the corridor geometry, and piece E's URL spelling of a road (`x1,y1,x2,y2`, whose
  // encoder indexes `coords[0]`/`coords[1]` directly) all assume it. Enforced HERE, once, where the
  // bundle crosses into the widget -- a second length check inside the codec would be a guard that
  // cannot fire, which this project treats as a defect of its own.
  for (const [i, r] of b.roads.entries()) {
    if (r.coords.length !== 2) {
      throw new Error(
        `field.json's road ${i + 1} has ${r.coords.length} points; this widget needs exactly two`);
    }
  }
```

and add the guard's test to `web/test/field-boot.test.ts`, mounting over a modified payload
(`frontier-boot.test.ts`'s `mount(host, payload = bundle)` is the precedent; if `field-boot`'s own
`mount` has no payload override, add one in that same shape rather than a second helper):

```ts
test("a road that is not a two-point segment fails LOUDLY, on the page", async () => {
  const host = mountPoint();
  const bad = {
    ...bundle,
    roads: [{ ...bundle.roads[0]!, coords: [[0, 0], [1, 1], [2, 2]] }, bundle.roads[1]!],
  };
  await mount(host, null, bad);
  // The message reaches the reader where the caption was -- not the console behind an intact PNG.
  assert.match(captionText(host), /road 1 has 3 points/);
});
```

- [ ] **Step 5: `perm-graph.ts` — clamp the prefix**

Immediately after `const state: StateSource<PermGraphState> = makeState(initialState(host));`:

```ts
// `?prefix=` is a bare non-negative integer; how many prefixes THIS block has is a property of the
// fetched bundle, which no codec can know (design §2.3). Clamp rather than throw -- an
// out-of-range number must land on the last prefix, not on an error card -- and because a clamped
// value equal to the initial stops being emitted, the URL self-corrects.
const maxPrefix = b.n_prefixes - 1;
if (state.get().prefix > maxPrefix) state.set({ prefix: maxPrefix });
```

- [ ] **Step 6: `frontier.ts` — reset an unknown isolated method**

Immediately after the `makeState<FrontierState>({...})` call:

```ts
// `?method=` names a curve in THIS bundle. An unknown one is not inert: the draw loop's
// `if (s.isolated !== null && s.isolated !== key) continue` would skip EVERY curve and render an
// empty chart. Reset to null, which also drops the key from the URL (design §2.3).
//
// `Object.hasOwn`, never `in`: `"toString" in b.methods` is true for every object, so an
// `in`-based membership test would accept a prototype key as a method name.
const isolated = state.get().isolated;
if (isolated !== null && !Object.hasOwn(b.methods, isolated)) state.set({ isolated: null });
```

- [ ] **Step 7: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: type-check clean, `# fail 0`.

- [ ] **Step 8: Fault injection**

1. Revert `slider.value` in `region-grow.ts` to `String(b.budget.default)` ⇒ the budget test reddens.
2. Revert the width slider to `String(width0)` ⇒ the field test reddens.
2b. Delete the `coords.length !== 2` loop ⇒ the segment-invariant test reddens. Confirm the failure
   is the ASSERTION and not an uncaught `TypeError` from somewhere downstream — if it is the latter,
   the guard is not the thing being tested and you must say so rather than accept the red.
3. Delete the prefix clamp ⇒ the perm-graph test reddens.
4. Swap `Object.hasOwn(b.methods, isolated)` for `isolated in b.methods` ⇒ the `toString` test
   reddens (and only that one).
5. Change `REGION_GROW_URL.seed` back to `intParam("seed")` ⇒ `npm run check` must fail.

- [ ] **Step 9: Commit**

```bash
git add web/src/widgets web/test
git commit -m "fix(web): honour URL-supplied state in the controls and against the bundle"
```

---

### Task 5: `ScreenState.floor` becomes `number | null`

**Files:**
- Modify: `web/src/widgets/screen-map.ts`
- Test: `web/test/screen-map-boot.test.ts`

**Interfaces:**
- Consumes: `SCREEN_MAP_URL` (Task 3).
- Produces: `ScreenState { city; metric; floor: number | null }`;
  `SCREEN_MAP_URL.floor` becomes `nullableNumberParam("floor", 6)`;
  new module-private `defaultFloorFor(bundle, metric, sc): number`.

**Why a model change and not a write-back:** `syncFloor` writes the slider *before* `makeState`, and
the floor's default is a function of the metric. A plain write-back would leave `?metric=density`
alone clamping `depth_density_proxy`'s 0.0128 into `density`'s unrelated range — selecting all
16,451 blocks from a URL that asked for nothing unusual. `null` meaning "this metric's own default"
is already exactly what `syncFloor`'s `preferred` parameter means, so all four URL combinations come
out right with no "did the URL set this?" question. Design §1.6 carries the table.

- [ ] **Step 1: Write the failing tests**

In `web/test/screen-map-boot.test.ts`, extending that file's existing `mount(width, drawFailure)`
helper with a trailing `search` parameter exactly as Task 4 describes. Its fixtures `ct` and `nb`
are the REAL committed bundles, so `ct.floors` genuinely carries both `depth_density_proxy` and
`density_compactness` — no fixture needs extending. Use the file's existing `setMetric`, `setCity`,
`setFloor` and `readoutText` helpers rather than poking DOM nodes directly.

```ts
test("?metric= alone takes THAT metric's own default floor, not the previous metric's number", async () => {
  const { store, readout } = await mountWithSearch("metric=density_compactness");
  const shipped = FIXTURE_CT.floors.find((f) => f.metric === "density_compactness")!;
  assert.equal(store.get().floor, null, "null means 'this metric's default', and stays null");
  assert.match(readout.textContent, new RegExp(String(shipped.n)),
    "the pool is this metric's calibrated pool, not every block in the city");
});

test("?metric=&floor= together honour the explicit floor", async () => {
  const { store } = await mountWithSearch("metric=density_compactness&floor=0.0004");
  assert.equal(store.get().metric, "density_compactness");
  assert.equal(store.get().floor, 0.0004);
});

test("switching metric drops ?floor= from the URL", async () => {
  const { loc, metricSelect } = await mountWithSearch("floor=0.02");
  assert.ok(loc.written.at(-1)!.includes("floor="), "precondition: an explicit floor is in the URL");
  metricSelect.value = "density_compactness";
  metricSelect.fire("change");
  assert.ok(!loc.written.at(-1)!.includes("floor="),
    "a metric switch resets to the new metric's calibration, so no floor belongs in the URL");
});

test("switching city PINS the floor, carrying the absolute number across corpora", async () => {
  const { loc, cityToggle, store } = await mount(700, null, "");
  cityToggle.checked = true;
  cityToggle.fire("change");
  assert.equal(typeof store.get().floor, "number", "resolved at the switch, never left null");
  assert.ok(loc.written.at(-1)!.includes("city=nairobi"));
  assert.ok(loc.written.at(-1)!.includes("floor="));
});

test("an out-of-range ?floor= is clamped AND stops being emitted at that value", async () => {
  const { loc, store } = await mountWithSearch("floor=999");
  assert.ok(store.get().floor !== 999);
  assert.ok(!loc.written.at(-1)!.includes("floor=999"));
});
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: 5 failures in `screen-map-boot`.

- [ ] **Step 3: Change the interface and the codec**

```ts
/** `city` picks which already-fetched bundle is active; `metric` and `floor` are indices into THAT
 * bundle's own scoring.
 *
 * `floor` is `number | null`, and `null` means **this metric's own default** -- exactly what
 * `syncFloor`'s `preferred` parameter has always meant. The alternative, a resolved number, cannot
 * be right: the four metrics' scores span unrelated scales, so carrying one metric's number into
 * another (which a URL supplying only `?metric=` would force) clamps to that metric's minimum and
 * selects every block in the city. Keeping the "unset" case spellable also keeps `?floor=` out of
 * the URL for a reader who only switched metric (design §1.6). */
export interface ScreenState {
  city: "capetown" | "nairobi";
  metric: MetricName;
  floor: number | null;
}
```

and in `SCREEN_MAP_URL`, `floor: nullableNumberParam("floor", 6)`.

- [ ] **Step 4: Factor `defaultFloorFor` out of `syncFloor`**

Above `syncFloor`:

```ts
/** This metric's own default floor: its shipped calibration where `bundle.floors` carries one, else
 * `floorAtShippedPoolSize`'s non-invented fallback. Factored out so the SLIDER's resolution
 * (`syncFloor`) and the PICTURE's resolution (`render`) cannot diverge -- `ScreenState.floor` is
 * `null` for "this metric's default" and both paths resolve it through here.
 *
 * Takes the already-computed `sc` rather than calling `scores` again: `render` has it cached in
 * `rankedFor`'s companion, and re-scoring 16,451 blocks per frame to answer a question already
 * answered would be waste with no correctness gain. */
function defaultFloorFor(bundle: CityBundle, metric: MetricName, sc: Float64Array): number {
  return bundle.floors.find((f) => f.metric === metric)?.value
      ?? floorAtShippedPoolSize(bundle, metric, sc);
}
```

and inside `syncFloor`, replace
`const target = preferred ?? shipped ?? floorAtShippedPoolSize(bundle, metric, sc);`
(plus its now-unused `shipped` binding) with
`const target = preferred ?? defaultFloorFor(bundle, metric, sc);`.

- [ ] **Step 5: Rewire boot and the three handlers**

Replace the `initialFloor`/`makeState`/`metricSelect.value`/`cityToggle.checked` block with:

```ts
const state: StateSource<ScreenState> = makeState<ScreenState>({
  city: "capetown", metric: "depth_density_proxy", floor: null,
});
// The URL (piece E) may have set any of the three before this line, so the slider's bounds and
// value are resolved against the STATE's own city and metric -- and `state.floor` goes straight
// through as `preferred`, since `null` already means "this metric's default" (design §1.6).
const s0 = state.get();
metricSelect.value = s0.metric;
cityToggle.checked = s0.city === "nairobi";
const resolved = syncFloor(floorSlider, bundles[s0.city], s0.metric, s0.floor);
// Clamping is bundle-dependent (design §2.3): an out-of-range `?floor=` lands on the nearest usable
// value and stops being emitted, rather than silently selecting nothing or everything.
if (s0.floor !== null && resolved !== s0.floor) state.set({ floor: resolved });
```

Metric handler:

```ts
metricSelect.addEventListener("change", () => {
  const metric = asMetric(metricSelect.value);
  syncFloor(floorSlider, bundles[state.get().city], metric, null);
  // `null`, not the number syncFloor just resolved: the same picture, and it keeps `?floor=` out of
  // the URL for a reader who only switched metric.
  state.set({ metric, floor: null });
});
```

City handler:

```ts
cityToggle.addEventListener("change", () => {
  const city: ScreenState["city"] = cityToggle.checked ? "nairobi" : "capetown";
  const s = state.get();
  const floor = syncFloor(floorSlider, bundles[city], s.metric, s.floor);
  // RESOLVED, never null. `syncFloor`'s own docstring argues that an ABSOLUTE floor must carry
  // across corpora rather than being redefined per city -- pinning the number here is what carries
  // it, and saying so in the URL is honest, because the reader chose to carry it.
  state.set({ city, floor });
});
```

In `render`, resolve before selecting:

```ts
const floor = st.floor ?? defaultFloorFor(bundle, st.metric, s);
const sel = selectAt(bundle, order, s, floor);
```

- [ ] **Step 6: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: type-check clean, `# fail 0`.

- [ ] **Step 7: Fault injection**

1. Make the metric handler `state.set({ metric, floor: syncFloor(...) })` (a resolved number) ⇒ the
   "drops ?floor=" test reddens.
2. Make the city handler `state.set({ city, floor: null })` ⇒ the "pins the floor" test reddens.
3. In `boot`, pass `null` instead of `s0.floor` as `preferred` ⇒ the `metric=&floor=` test reddens.
4. In `render`, use `st.floor ?? 0` ⇒ the `?metric=` alone test reddens.

- [ ] **Step 8: Commit**

```bash
git add web/src/widgets/screen-map.ts web/test/screen-map-boot.test.ts
git commit -m "fix(web): ScreenState.floor is number|null -- 'this metric's own default' is a state"
```

---

### Task 6: Bake the spine into the city bundles

**Files:**
- Modify: `scripts/gen_screen_map.py`
- Regenerate (committed artifacts): `examples/screen-map/{capetown.json,nairobi.json,screen_map.png,README.md}`, `web/src/screen_map.d.ts`
- Test: `tests/test_screen_map_bundle.py`

**Interfaces:**
- Produces, in each city bundle and in the regenerated `web/src/screen_map.d.ts`:
  ```ts
  export interface CityFollow { block_id: string; index: number; x: number; y: number }
  // on CityEncoding:  follow_color: string;
  // on CityBundle:    follow?: CityFollow;
  ```
- Consumed by Task 7 (`render/city.ts`, `screen-map.ts`).

**Data note:** the kblock parquet cache is present at `~/.cache/reblock` (verified), so the baker
runs locally without downloading anything.

- [ ] **Step 1: Write the failing test**

In `tests/test_screen_map_bundle.py`:

```python
FOLLOW_SOURCE = Path("examples/perm-graph/bundle.json")


def test_the_followed_block_is_the_one_every_later_stage_uses(capetown: dict[str, Any]) -> None:
    """The site's spine, derived rather than typed. perm-graph, displacement-field and
    method-comparison all pin one block and region-grow seeds from it; this asserts the city map
    marks the SAME one, so a re-bake of any of them cannot leave the marker pointing elsewhere."""
    want = json.loads(FOLLOW_SOURCE.read_text(encoding="utf-8"))["block_id"]
    follow = capetown["follow"]
    assert follow["block_id"] == want
    assert capetown["block_id"][follow["index"]] == want


def test_the_follow_marker_sits_inside_its_own_block(capetown: dict[str, Any]) -> None:
    """A centroid outside its polygon would draw the ring in a neighbour's block -- silently, and
    at 0.61 CSS px per block, invisibly wrong rather than obviously wrong."""
    follow = capetown["follow"]
    ring = capetown["rings"][follow["index"]][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    assert min(xs) <= follow["x"] <= max(xs)
    assert min(ys) <= follow["y"] <= max(ys)


def test_nairobi_has_no_follow_key_at_all(nairobi: dict[str, Any]) -> None:
    """The followed block is in Cape Town. ABSENT, not null -- a null field is one that looks
    answerable and is not, exactly as `informal` is handled."""
    assert "follow" not in nairobi


def test_both_cities_carry_the_follow_colour(capetown: dict[str, Any],
                                             nairobi: dict[str, Any]) -> None:
    """The ENCODING is shared even where the marker is not: the widget reads its colour from
    whichever bundle is active, and a city switch must not leave it undefined."""
    for b in (capetown, nairobi):
        assert b["encoding"]["follow_color"].startswith("#")
```

`test_dts_declares_exactly_the_keys_both_bundles_carry` needs no edit — it already unions both
bundles' keys and compares against the `.d.ts`, so it will fail until the template is regenerated.

- [ ] **Step 2: Run it to make sure it fails**

Run: `pixi run python -m pytest tests/test_screen_map_bundle.py -q 2>&1 | tail -15`
Expected: 5 failures (four new, plus the `.d.ts` key test once the bundle changes).

- [ ] **Step 3: Bake `follow` and `follow_color`**

In `scripts/gen_screen_map.py`:

Add `_ROAD_COLOR` to the existing `from reblock.render import ...` line. Add near the other paths:

```python
# The site's spine: the one block every later stage is about. READ, never typed -- perm-graph,
# displacement-field and method-comparison all pin it, and region-grow seeds from it, so taking it
# from an artifact is what keeps the city map's marker and those four figures from drifting apart.
FOLLOW_SOURCE = Path("examples/perm-graph/bundle.json")
FOLLOW_CITY = "capetown"
```

Add the TypedDict beside `CityFloor`, and the two new fields:

```python
class CityFollow(TypedDict):
    block_id: str
    index: int
    x: float
    y: float
```

`CityEncoding` gains `follow_color: str`; `CityBundle` gains `follow: NotRequired[CityFollow]` (the
same shape `informal` uses). The `Encoding` dataclass gains `follow_color: str`, `ENCODING` gains
`follow_color=_ROAD_COLOR`, and `ENCODING_DICT` gains the matching entry. Extend `Encoding`'s
docstring with:

```
`follow_color` reuses `_ROAD_COLOR`, the site's one blue. It is the only palette constant that is
distinct at a glance from all three colours already here -- `base_color` #dddddd, `selected_color`
#c0392b and `informal_color` #d98c00 -- which is the whole requirement for a marker that has to be
findable among 16,451 blocks.
```

In `build_bundle`, after the `rings` comprehension:

```python
    follow: CityFollow | None = None
    if city == FOLLOW_CITY:
        want = json.loads(FOLLOW_SOURCE.read_text(encoding="utf-8"))["block_id"]
        if want not in block_ids:
            raise ValueError(
                f"{FOLLOW_SOURCE} pins block {want!r}, which is not among {city}'s "
                f"{len(block_ids)} blocks above MIN_COUNT={MIN_COUNT} -- the site's spine and this "
                f"bundle disagree about which block the walkthrough follows")
        idx = block_ids.index(want)
        # representative_point(), not centroid: guaranteed INSIDE the polygon even where the block
        # is concave or holed, so the marker can never be drawn over a neighbour.
        pt = simplified[idx].representative_point()
        follow = CityFollow(block_id=want, index=idx,
                            x=cm(float(pt.x) - ox), y=cm(float(pt.y) - oy))
```

and after the `informal` block:

```python
    if follow is not None:
        bundle["follow"] = follow
```

- [ ] **Step 4: Draw the same ring in the PNG**

Give `_render_screen_map` a `follow_xy: tuple[float, float] | None` parameter and, before the
`set_aspect` call:

```python
    if follow_xy is not None:
        # The same ring the canvas widget draws, in the same colour, on the same block -- one
        # `encoding` feeding the matplotlib fallback and the widget alike, so a JS-off or print
        # reader sees the block the prose says the page follows.
        #
        # Size: the widget's ring is 6 CSS px of radius on a ~700 px canvas. This figure is 10 in
        # wide, so the proportional diameter is 2 * 6/700 * 10 * 72 = 12.3 pt.
        ax.plot(*follow_xy, marker="o", markersize=12.3, markerfacecolor="none",
                markeredgecolor=ENCODING.follow_color, markeredgewidth=2.0, zorder=4)
```

The caller passes the followed block's **unprojected-into-the-figure** coordinates — i.e. the raw
`representative_point()` of that block in `gdf`'s CRS, not the origin-relative bundle value, since
`gdf.plot` draws in world coordinates. Compute it in `main()` from the same `simplified` list the
bundle used, so the two cannot disagree.

- [ ] **Step 5: Update the `.d.ts` template and regenerate everything**

Add `CityFollow`, `follow_color` and `follow?` to `DTS_TEMPLATE`, keeping the existing docstring
style (each field that needs a reason gets one). Then:

Run: `pixi run python -m scripts.gen_screen_map`
Expected: rewrites both bundles, `screen_map.png`, `README.md` and `web/src/screen_map.d.ts`.

- [ ] **Step 6: Run the tests, the lint and the type-check**

Run: `pixi run python -m pytest tests/test_screen_map_bundle.py -q && pixi run lint && cd web && npm run check`
Expected: all green.

- [ ] **Step 7: Fault injection**

1. Point `FOLLOW_SOURCE` at `examples/displacement-field/field.json` — the SAME block, so the guard
   must stay green (this proves the assertion is about the block, not the file). Restore.
2. Hard-code `want = "ZAF.9.3.1_1_44882"` ⇒ the baker must **raise**, not silently omit. Restore.
3. Replace `representative_point()` with a point outside the block (e.g. `Point(ox, oy)`) ⇒ the
   inside-its-own-block test must redden. Restore and re-bake.

- [ ] **Step 8: Commit**

```bash
git add scripts/gen_screen_map.py examples/screen-map web/src/screen_map.d.ts tests/test_screen_map_bundle.py
git commit -m "feat(screen-map): bake the followed block and its colour into the city bundles"
```

---

### Task 7: Draw the follow ring

**Files:**
- Modify: `web/src/render/city.ts`, `web/src/widgets/screen-map.ts`
- Test: `web/test/screen-map-boot.test.ts`

**Interfaces:**
- Consumes: `CityFollow`, `CityEncoding.follow_color`, `CityBundle.follow` (Task 6).
- Produces: nothing later tasks consume.

**Why a ring and not an outline:** D3 measured the median Cape Town block at **0.61 CSS px²** on the
shipped canvas, 69.6% under 1 px². Outlining the followed block would be invisible. The marker is
sized in **screen pixels**, on the frame layer, so a floor or metric change never re-touches it.

- [ ] **Step 1: Write the failing test**

```ts
These use `screen-map-boot.test.ts`'s `mount(...)` as extended in Task 5 — `mount(700, null, "")`
for Cape Town, `mount(700, null, "city=nairobi")` for Nairobi. That file already has `cityBbox` and
`VIEW_CT`; `viewOf(bundle)` below is `fitBbox(cityBbox(bundle), SIZE, SIZE, E.pad)`, which for Cape
Town is the existing `VIEW_CT` constant — use it rather than recomputing.

**`arc` is a `PathOp`, not a `Call`** — verified in `harness.ts`: `RecordingContext` records only
`clearRect`/`stroke`/`fill`/`drawImage` as `Call`s, and path construction (`moveTo`/`lineTo`/`arc`/
`closePath`) accumulates into that call's `path`. So the ring is a **`stroke` whose path is exactly
one `arc`**, which is a sharper assertion than a bare op count: it distinguishes the ring from the
`e.block_lw * 2` block outlines that share the frame. Do NOT widen the harness to make this easier.

```ts
/** The follow ring: the one stroke whose whole path is a single arc. Every other stroke on this
 * frame is a block outline -- a polyline of moveTo/lineTo/closePath -- so this shape test names the
 * ring by what it IS, rather than by counting ops and hoping. */
function followRings(cv: FakeElement): Call[] {
  return lastFrame(cv).filter(
    (c) => c.op === "stroke" && c.path.length === 1 && c.path[0]!.op === "arc");
}

test("the followed block is ringed, at a fixed screen radius, on the frame layer", async () => {
  const { cv, bundle } = await mount(700, null, "");
  const rings = followRings(cv);
  assert.equal(rings.length, 1, "exactly one ring, on the frame -- not one per block");
  const [x, y, r] = rings[0]!.path[0]!.args;
  const expected = toScreen(viewOf(bundle), bundle.follow!.x, bundle.follow!.y);
  assert.ok(Math.abs(x - expected[0]) < 0.5 && Math.abs(y - expected[1]) < 0.5);
  assert.equal(r, FOLLOW_RADIUS_PX, "a WORLD radius would shrink to nothing at city zoom");
  // Recorded AT the call, which is the whole point of the style snapshot: "the widget assigned
  // follow_color at some moment" says nothing about what was stroked with it.
  assert.equal(rings[0]!.strokeStyle, bundle.encoding.follow_color);
  assert.equal(rings[0]!.lineWidth, 2);
});

test("a bundle with no follow draws no ring", async () => {
  const { cv } = await mount(700, null, "city=nairobi");
  assert.equal(followRings(cv).length, 0);
});

test("the ring survives a floor change, which never repaints the base layer", async () => {
  const { cv, floorSlider } = await mount(700, null, "");
  floorSlider.value = "0.02";
  floorSlider.fire("input");
  fireAnimationFrame();
  assert.equal(followRings(cv).length, 1);
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: 2 failures (the third passes vacuously until the ring exists — note that in the report).

- [ ] **Step 3: Draw it in `render/city.ts`**

```ts
/** The followed block's ring, in CSS pixels. Fixed on SCREEN, never in world units: the median Cape
 * Town block is 0.61 CSS px² at this canvas size (measured, D3), so a world-scaled marker -- or an
 * outline of the block itself -- would be smaller than a pixel and simply absent. */
export const FOLLOW_RADIUS_PX = 6;
const FOLLOW_LW_PX = 2;
```

At the end of `paintFrame`, after the selected-prefix stroke loop:

```ts
      // Painted on the FRAME, not the base layer: the base is repainted only on a resize or a city
      // switch, and this must survive a floor or metric change, which re-blits it unchanged.
      const follow = bundle.follow;
      if (follow !== undefined) {
        const [fx, fy] = toScreen(view, follow.x, follow.y);
        ctx.beginPath();
        ctx.arc(fx, fy, FOLLOW_RADIUS_PX, 0, Math.PI * 2);
        ctx.strokeStyle = e.follow_color;
        ctx.lineWidth = FOLLOW_LW_PX;
        ctx.stroke();
      }
```

Import `toScreen` from `../view/transform.js` if the module does not already.

- [ ] **Step 4: Say so in the widget's readout**

`describeSelection` gains one clause, only where `bundle.follow` exists, naming the block the rest of
the site follows — the canvas carries no accessible text, so a screen-reader user must be told the
ring is there. Read the block id from `bundle.follow.block_id`, never typed.

- [ ] **Step 5: Run the tests and the type-check**

Run: `cd web && npm run check && npm test 2>&1 | grep -E "(pass|fail) [0-9]+"`
Expected: type-check clean, `# fail 0`.

- [ ] **Step 6: Fault injection**

1. Move the ring into `paintBase` ⇒ the floor-change test must redden.
2. Scale the radius by the view (`FOLLOW_RADIUS_PX * view.scaleX`) ⇒ the radius assertion reddens.
3. Draw the ring unconditionally with `follow!` ⇒ the Nairobi test reddens (or throws).

- [ ] **Step 7: Commit**

```bash
git add web/src/render/city.ts web/src/widgets/screen-map.ts web/test/screen-map-boot.test.ts
git commit -m "feat(web): ring the followed block on the city map's frame layer"
```

---

### Task 8: The Explore page

**Files:**
- Create: `docs/_partials/explore.md`
- Modify: `scripts/gen_site_pages.py`, `docs/stylesheets/sbu.css`, `mkdocs.yml`, `.gitignore`
- Test: `tests/test_gen_site_pages.py`

**Interfaces:**
- Consumes: everything above; the five existing figure producers.
- Produces: `docs/explore.md` (generated, gitignored), served at `<base>/explore/`.

- [ ] **Step 1: Write the failing test**

`tests/test_gen_site_pages.py` already has both helpers this needs — use them, do not add parallels:

* `render_page(name)` renders a partial AND applies `_write_page`'s depth/url_depth rewrite, reading
  those two numbers off `main()`'s own call **so a hardcoded copy cannot drift**. Its regex is
  currently anchored to `_write_page(methodology_dir / "<name>.md", …)`, and Explore is written to
  `DOCS / "explore.md"`. **Generalise the regex to accept either receiver** — keep the
  read-it-off-`main()` property, which is the whole point of the helper:

  ```python
      m = re.search(rf'_write_page\((?:DOCS|methodology_dir|results_dir) / "{name}\.md",\s*'
                    rf'_render_partial\("{name}"\),\s*depth=(\d+), url_depth=(\d+)', src)
  ```

* a `<name>_body` fixture (see `screening_body`) returns `_render_partial(name)` with `DOCS`/`ASSETS`
  monkeypatched into `tmp_path`, because `_render_partial` runs **every** producer in `MARKERS` and
  producers copy assets. Add `explore_body` in exactly that shape.

Then add:

```python
def test_explore_carries_all_five_mount_points(explore_body: str) -> None:
    """The page the whole piece exists for: one mount point per stage, in pipeline order, each
    substituted from a marker rather than typed."""
    order = ["screen-map", "region-grow", "perm-graph", "displacement-field", "frontier"]
    assert re.findall(r'data-widget="([a-z-]+)"', explore_body) == order


def test_explore_rewrites_the_screen_maps_two_bundle_urls() -> None:
    """explore.md serves at <base>/explore/ -- url_depth 1 -- so each bundle path needs exactly one
    `../`. ScreenMap carries TWO bundle attributes where every other widget carries one, and the
    general `data-bundle="([^"]+)"` regex matches NEITHER of them (that literal substring never
    occurs inside `data-bundle-capetown="`). This is the same trap the Screening page's own twin of
    this test guards, on the page that now carries the most mount points on the site."""
    page = render_page("explore")
    assert 'data-bundle-capetown="../assets/screen-map/capetown.json"' in page
    assert 'data-bundle-nairobi="../assets/screen-map/nairobi.json"' in page
    for attr in ("data-bundle-capetown", "data-bundle-nairobi"):
        # INSIDE the loop. Hoisted out, this runs once against whichever `attr` the loop left
        # behind -- checking one of the two, which is the exact half-blind shape D3's own guard
        # had and the reason this test exists.
        assert f'{attr}="assets/' not in page, attr
    for url in re.findall(r'data-bundle="([^"]+)"', page):
        assert url.startswith("../assets/"), url


def test_explore_is_generated_and_gitignored() -> None:
    """A generated page that is committed drifts from its artifacts the moment anything re-bakes --
    the reason docs/index.md and docs/reproduce.md are ignored too."""
    ignored = [line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "docs/explore.md" in ignored


def test_the_widget_bundle_guard_can_see_the_explore_page() -> None:
    """`_assert_widget_bundle_present` scans a HAND-WRITTEN page list. Explore carries the most
    mount points of any page on the site; leaving it out would make the guard blind exactly where a
    missing docs/js/widgets.js does the most damage -- five dead widgets behind five intact PNGs."""
    src = (ROOT / "scripts" / "gen_site_pages.py").read_text(encoding="utf-8")
    block = re.search(r"generated_pages = \(([^)]*)\)", src, flags=re.S)
    assert block is not None, "main()'s generated_pages assignment moved; update this derivation"
    assert '"explore.md"' in block.group(1)


def test_explore_is_in_the_nav() -> None:
    """An orphan page -- one outside nav -- logs at INFO under `mkdocs build --strict` and the
    build still exits 0, so nothing else here would catch its absence."""
    assert re.search(r"^\s+- Explore: explore\.md$",
                     (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), flags=re.M)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `pixi run python -m pytest tests/test_gen_site_pages.py -q 2>&1 | tail -15`
Expected: 5 failures.

- [ ] **Step 3: Factor the Permeability widget panel out**

In `scripts/gen_site_pages.py`, split `_perm_graph_figures` so the caption text and the mount-point
attributes are built by one helper used by both:

```python
def _perm_graph_panel(layer: str, state: str, *, mounted: bool) -> str:
    """One panel of the egress-graph grid, with its caption read off perm_graph.json.

    Extracted so the Permeability section (all four panels) and Explore (the one a reader can drag)
    cannot quote the same artifact two different ways. `mounted` decides whether this panel carries
    the widget's mount-point attributes -- only current/after ever does, because it is the panel the
    fallback PNG and the caption already describe."""
    meta = json.loads((PERMGRAPH / "perm_graph.json").read_text(encoding="utf-8"))
    method = friendly_method_name(meta["method"])
    p_after, road_m, n_upgraded = (meta["permeability_after"] * 100.0, meta["road_m"],
                                   meta["n_upgraded"])
    url = _copy_asset(PERMGRAPH / f"graph_{layer}_{state}.png", "perm-graph")
    if url is None:
        return ""
    roads_clause = ("No new roads." if state == "before" else
                    f"Under {method}'s roads at the matched-permeability standard: "
                    f"{road_m:,.0f} m of road, reaching {p_after:.1f}% permeability.")
    caption = (f"Grey edge width is {layer} on the footpath mesh; the {n_upgraded} road-raised "
               f"edges (blue) draw at a fixed width instead. {roads_clause}")
    attrs = ""
    if mounted:
        bundle_url = _copy_asset(PERMGRAPH / "bundle.json", "perm-graph")
        if bundle_url:
            bundle_meta = json.loads((PERMGRAPH / "bundle.json").read_text(encoding="utf-8"))
            attrs = (f'data-widget="perm-graph" data-bundle="{bundle_url}" '
                     f'data-layer="current" data-prefix="{bundle_meta["lens_b_index"]}"')
    return _figure(url, f"egress graph, {layer}, {state} roads", caption, attrs=attrs)


def _perm_graph_widget_figure() -> str:
    """JUST the interactive panel -- current, after -- for the Explore page.

    The Permeability section wants all four because the point THERE is the before/after and
    conductance/current comparison. Explore wants the one a reader can drag; repeating the other
    three would add three large images to a page that already fetches every bundle on the site, for
    no additional teaching. Both pages go through `_perm_graph_panel`, so neither can drift.

    The block-level lead sentence the four-panel grid carries stays with THAT grid: on Explore the
    same facts are in the surrounding prose and the stage heading, and repeating them under one
    figure would be the caption-duplication finding F5 again."""
    return _perm_graph_panel("current", "after", mounted=True)
```

`_perm_graph_figures` then becomes its own loop over `_perm_graph_panel(layer, state,
mounted=(layer == "current" and state == "after"))`, keeping its existing intro sentence and
`.sbu-figure-grid` wrapper and its full docstring — the reasoning there is load-bearing and must not
be lost in the extraction.

- [ ] **Step 4: Register the two new markers**

Add to `MARKERS`:

```python
    "PERMGRAPHWIDGET": _perm_graph_widget_figure,
    "FRONTIER": _frontier_figure,
```

`_frontier_figure` already exists and is called directly by `gen_benchmark_section()`; registering it
changes nothing there and makes it available to a partial. The existing both-directions marker test
then covers both automatically, so a marker with no producer — or a producer no partial uses — fails.

- [ ] **Step 5: Write `docs/_partials/explore.md`**

Open with the maintainer comment every partial carries (do-not-edit pointer, the five marker names,
the "no typed numbers" rule). Then:

```markdown
# Explore

<nav class="sbu-stage-rail" aria-label="Pipeline stages">
  <ol>
    <li><a href="#screening">Screening</a></li>
    <li><a href="#growth">Growth</a></li>
    <li><a href="#permeability">Permeability</a></li>
    <li><a href="#displacement">Displacement</a></li>
    <li><a href="#methods">Methods</a></li>
  </ol>
</nav>

Five stages, one block. Everything below follows the same Cape Town block from the city-wide screen
that first flagged it to the method frontier that scores the roads through it. Every control writes
to the address bar, so a view you find here is a link you can paste into a review.

## Screening {#screening}

A metro has far too many blocks to reblock every one of them, so the pipeline opens with one cheap
number per block and a floor on it. Drag the floor and watch the selected pool grow and shrink;
switch the metric to see how much the choice of number matters. Switching to Nairobi lands the same
*absolute* floor on a corpus it was never calibrated against — which is the whole reason the floor is
a score and not a percentile. The ringed block is the one every stage below follows.

<!-- SCREENMAP -->

## Growth {#growth}

A block is rarely the right unit of work: the roads that would serve it run through its neighbours,
and a road that stops at a boundary serves nobody on the other side. Growth takes a seed block and
accretes neighbours greedily until a building budget is spent. Drag the budget to watch the region
grow, and click any block to reseed from it. The accretion order is production's own — the browser
replays the same sequence the pipeline records, rather than re-deriving the rule.

<!-- REGIONGROW -->

## Permeability {#permeability}

Permeability is the number the methods are scored on: how easily every parcel drains to the street.
Drag the road prefix to add roads in the order the method builds them, and watch current concentrate
into each new corridor as the potential field flattens behind it. Grey edges are the footpath mesh
the metric solves over; blue ones are the edges a road raised.

<!-- PERMGRAPHWIDGET -->

## Displacement {#displacement}

A road that reaches everyone by demolishing everyone is not a solution, so every road is charged for
what it costs. Displacement asks how far the road's corridor reaches into each building's own disk.
Drag either road's endpoints, widen the corridor, and switch the second road on: two overlapping
corridors are charged once, not twice, which is why the cost is a property of the road *set* rather
than a sum over roads.

<!-- DISPFIELD -->

## Methods {#methods}

Every method is a different answer to the same trade — permeability bought with displacement — and
each curve is one method's road build-out from nothing to its full network. Set a displacement or a
permeability target and the guides show which methods reach it and what each one pays to get there.
Click a method's name to isolate its curve.

<!-- FRONTIER -->
```

Note what is **not** here: no numerals anywhere in the prose. Every figure below carries its own
artifact-read caption, and a number typed into this file is exactly the drift class the site's truth
pass closed.

The prose in each stage says what the stage decides and what the reader should try with the control
in front of them — two to four sentences each. **No numerals**: every figure the page shows already
carries its own artifact-read caption, and a number typed here is exactly the drift class the site's
truth pass closed.

- [ ] **Step 6: Write the page out**

In `main()`, immediately after the `reproduce.md` call (same `depth=0, url_depth=1` reasoning, and
likewise written directly into `docs/` rather than into a cleared subdirectory, so ruling F4's
`rmtree` hazard does not apply):

```python
    # docs/explore.md serves at <base>/explore/ -- source depth 0, url_depth 1, exactly like
    # reproduce.md above. It carries FIVE mount points, including ScreenMap's two bundle
    # attributes, which is why _write_page rewrites each attribute name separately.
    _write_page(DOCS / "explore.md", _render_partial("explore"), depth=0, url_depth=1,
                title="Explore")
```

Add `DOCS / "explore.md"` to the `generated_pages` list, add `docs/explore.md` to `.gitignore` beside
`docs/reproduce.md`, and add `- Explore: explore.md` to `mkdocs.yml`'s nav between `background.md`
and the Methodology section. **Change nothing else in `mkdocs.yml`.**

- [ ] **Step 7: Style the rail**

In `docs/stylesheets/sbu.css`, beside the existing `.sbu-figure-grid` rules:

```css
/* ------------------------------------------------------------------- the Explore stage rail */

/* The stage strip at the top of Explore. Deliberately STATIC: mkdocs.yml already enables
   `toc.follow`, so the right-hand table of contents already scroll-follows the reader through these
   same five headings. A second IntersectionObserver here would duplicate behaviour the theme ships
   and add a per-page observer to maintain.

   The numbers come from `counter()` on the <ol>, never typed into the markup: an <ol> whose items
   carry hand-written "1." text renumbers wrongly the moment a stage is added or reordered, and a
   screen reader would announce the number twice.

   `flex-wrap` with `min-width: 0` nowhere needed here (the items are short), but the wrap itself
   matters: five items do not fit one line at a 414px viewport, and a non-wrapping row would scroll
   the whole page sideways -- the bug this sheet has already hit once (see the .sbu-hero rule). */
.md-typeset .sbu-stage-rail ol {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 1.4rem;
  margin: 0 0 1.6rem;
  padding: 0.55rem 0;
  list-style: none;
  counter-reset: sbu-stage;
  border-top: 1px solid var(--sbu-rule);
  border-bottom: 1px solid var(--sbu-rule);
}

.md-typeset .sbu-stage-rail li {
  margin: 0;
  counter-increment: sbu-stage;
}

.md-typeset .sbu-stage-rail li::before {
  content: counter(sbu-stage);
  display: inline-block;
  min-width: 1.3em;
  color: var(--sbu-ink-muted);
  font-variant-numeric: tabular-nums;
}

.md-typeset .sbu-stage-rail a {
  font-weight: 600;
}
```

- [ ] **Step 8: Run everything**

```bash
pixi run python scripts/gen_site_pages.py
pixi run python -m pytest tests/test_gen_site_pages.py -q
pixi run lint && pixi run test
```

Then render the site for real, which is runnable locally (the backlog records the path; a browser is
still absent, so drag feel and glyph metrics stay unverified):

```bash
~/.cache/rattler/cache/cached-envs-v0/4937c48afb8986c1/bin/mkdocs build --strict \
  --site-dir "$(mktemp -d)"
```

Expected: exit 0, and `explore/index.html` in that site dir carries five `data-widget` attributes
with every bundle path resolving one directory up.

- [ ] **Step 9: Fault injection**

1. Remove `DOCS / "explore.md"` from `generated_pages` ⇒ the guard-visibility test reddens.
2. Delete the `data-bundle-nairobi` line from `_write_page` ⇒ the two-attribute test reddens (this
   is the guard D3's regex was blind to; confirm it fires).
3. Reorder two markers in the partial ⇒ the pipeline-order test reddens.

- [ ] **Step 10: Commit**

```bash
git add docs/_partials/explore.md scripts/gen_site_pages.py docs/stylesheets/sbu.css mkdocs.yml .gitignore tests/test_gen_site_pages.py
git commit -m "feat(site): the Explore page -- one block down all five stages"
```
