/** The URL as the page's state: one store per page, shared by every widget on it.
 *
 * `replaceState` only, never `pushState`. Back-to-undo-a-slider is not a behaviour anyone expects,
 * and skipping history entries is what lets four widgets keep their one-way "control writes state"
 * wiring instead of each growing a state-to-control write-back for `popstate` (design §2.2).
 */
import type { StateSource } from "../state.js";
import type { Param, UrlCodec } from "./param.js";

/** The `location`/`history` seam. Injected rather than reached for, so the store is testable
 * without stubbing two globals, and so exactly one place in the codebase knows the browser API. */
export interface UrlLocation {
  /** With or without a leading "?" -- `browserLocation`'s returns one (`window.location.search`'s
   * own contract; "" when there is no query at all); `fakeLocation` in tests does not. Both are
   * fine here because the only consumer, `new URLSearchParams(...)`, accepts either form; a future
   * consumer that is not `URLSearchParams` would need to normalize first. */
  search(): string;
  /** `search` is the query WITHOUT its leading "?"; "" means "no query at all". */
  replace(search: string): void;
}

export function browserLocation(): UrlLocation {
  return {
    search: () => window.location.search,
    replace: (search) => {
      // The fragment is the reader's position on the page, not this store's state, and every
      // heading on the site is an anchor (mkdocs.yml's `toc: permalink: true`). A bare `?search`
      // or bare pathname resolves against the CURRENT url and DELETES the fragment -- silently,
      // since `replaceState` never navigates, so there is no scroll jump to notice it by.
      const { pathname, hash } = window.location;
      history.replaceState(null, "", (search === "" ? pathname : `?${search}`) + hash);
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
 * intermediate states nobody was ever looking at. It is also what keeps calls under Safari's
 * `replaceState` rate limit -- roughly 100 calls per 30 s, which a write per `pointermove` inside a
 * single drag would exceed on its own without it. */
export function debounce(ms: number, timers: Timers): Scheduler {
  let pending: number | null = null;
  return (write) => {
    if (pending !== null) timers.clear(pending);
    pending = timers.set(() => { pending = null; write(); }, ms);
  };
}

/** One widget's place in the emitted query, taken before that widget has any state to put in it.
 *
 * The split exists because *when a widget binds* and *where its keys belong in the URL* are two
 * different orderings. Every widget calls `makeState` from inside its own `fetch().then(boot)`, so
 * binds arrive in bundle-arrival order -- on Explore, screen-map is FIRST in the DOM and awaits
 * both city tiers (7.33 MB, its own comment's figure) while frontier is LAST and has 0.02 MB, so
 * that order comes out roughly reversed and need not repeat between loads. `reserve()` is called
 * from `register`'s mount closure instead (mount.ts), synchronously, as `mountAll` walks the
 * DOM. */
export interface UrlSlot {
  bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T>;
}

export interface UrlStore {
  reserve(): UrlSlot;
}

/** Percent-encoding that leaves "," alone. RFC 3986 lists "," as a sub-delim that is legal in a
 * query, and `roadsParam` puts three of them in every segment: `URLSearchParams.toString()` would
 * spell `road1=132.5%2C3.8%2C40.2%2C113.9`, which is correct and unreadable, in a URL whose entire
 * purpose is to be pasted into a review. */
function enc(s: string): string {
  return encodeURIComponent(s).replace(/%2C/g, ",");
}

export function urlStore(loc: UrlLocation, schedule: Scheduler): UrlStore {
  // Parsed ONCE, at construction: one parse per PAGE rather than one per widget, and it keeps
  // `bind` independent of write ordering -- which would matter only if two codecs on the page ever
  // claimed the same key, a case `mountAll` throws on before the second widget's own `bind` ever
  // runs (mount.ts). It does NOT protect a later widget from reading pruned defaults: `write()`
  // re-emits every unclaimed key verbatim, so an earlier widget's own write can never prune what a
  // later widget reads.
  const arrived: [string, string][] = [...new URLSearchParams(loc.search())];
  const claimed = new Set<string>();
  // One entry per `reserve()` call, in the order the slots were taken. An entry is the empty
  // emitter `reserve` installs, until that slot's own `bind` replaces it -- so on a page this
  // array's LENGTH is settled by `mountAll`'s walk and only its contents arrive late.
  const contributors: (() => Record<string, string>)[] = [];

  const write = (): void => {
    const parts: string[] = [];
    // Unclaimed first, in the order they arrived -- someone else's `utm_source` or `ref` is not
    // this store's to reorder or discard.
    for (const [k, v] of arrived) {
      if (!claimed.has(k)) parts.push(`${enc(k)}=${enc(v)}`);
    }
    // Then everything the widgets on this page contribute, slot by slot: `reserve()` takes a slot
    // as `mountAll` walks the DOM and `bind` fills it whenever that widget's bundle lands, so this
    // loop runs in MOUNT order however late a bind is, and each slot emits its own keys in codec-
    // declaration order. A slot nobody has bound yet emits nothing, and its widget's arrived keys
    // are carried by the unclaimed pass above meanwhile -- so a page mid-load can write a
    // different order than the same page settled, and it is the settled one that is reproducible.
    for (const emit of contributors) {
      for (const [k, v] of Object.entries(emit())) parts.push(`${enc(k)}=${enc(v)}`);
    }
    loc.replace(parts.join("&"));
  };

  return {
    reserve(): UrlSlot {
      // The slot is taken HERE, on a call `mountAll` makes synchronously as it walks the DOM, and
      // stands empty until the `bind` below replaces this emitter -- a network round trip later,
      // and possibly after several other widgets have already bound and written.
      let emit: () => Record<string, string> = () => ({});
      contributors.push(() => emit());
      return {
        bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T> {
          // `Object.keys` over a DECLARED mapped type: a loop over a schema, which is the
          // allowed form of dynamic access -- not a string lookup into a closed set. TypeScript
          // cannot express the per-key existential, so the four casts below (`codec as object`,
          // `as (keyof T)[]`, and `as Param<T[keyof T]>` at each of the two call sites that read
          // a field off `codec`) are the standard cost of iterating a mapped type; nothing
          // outside `bind` deals in `keyof T` strings.
          const fields = Object.keys(codec as object) as (keyof T)[];
          // `Map` is last-wins on a duplicate key; `URLSearchParams.get` is first-wins. A
          // hand-edited URL with a repeated key therefore decodes differently here than the
          // platform convention a reader might expect from `.get` -- benign, since the very next
          // write collapses it back to the one value `current` holds.
          const present = new Map(arrived);
          // Shallow. `current`'s nested values (e.g. `roads`) stay THE SAME OBJECTS as
          // `initial`'s until a `.set()` replaces them wholesale, so `same()`'s diff baseline is
          // only trustworthy if nothing ever mutates `initial` or anything reachable from it in
          // place. Safe today (`displacement-field.ts` always rebuilds via `.map()`); a widget
          // that instead mutated `state.get().roads[0].width_m` directly would move the baseline
          // together with the value, and the key would silently stop being emitted.
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
            // Every single-key `Param` in param.ts asserts `present[key]!` non-null rather than
            // checking -- sound only because `decode` below is never called unless at least one of
            // this param's own keys is present. Skip, don't call, on an empty subset.
            if (Object.keys(got).length === 0) continue;
            const decoded = p.decode(got, initial[field]);
            // No default. An unusable value falls back to the widget's own initial AND is
            // dropped, so the URL corrects itself in front of the reader, not carrying a lie.
            if (decoded === null) { dropped = true; continue; }
            current[field] = decoded;
          }

          emit = () => {
            const out: Record<string, string> = {};
            for (const field of fields) {
              const p = codec[field] as Param<T[keyof T]>;
              if (p.same(current[field], initial[field])) continue;
              Object.assign(out, p.encode(current[field], initial[field]));
            }
            return out;
          };

          // Self-correction does not wait for the reader to touch a control: if anything was
          // dropped, the corrected URL is written now.
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
    },
  };
}
