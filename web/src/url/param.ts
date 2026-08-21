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
