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
