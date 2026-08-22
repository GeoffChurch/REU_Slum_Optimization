# Explore chain + URL-as-state (site redesign, piece E) — design

**Parent design:** `docs/superpowers/specs/2026-08-13-site-redesign-design.md` (§1 nav, §7 piece E,
the *URL-as-state* and *Embedded and chained are one component* subsections).

**Predecessors:** C (`2026-08-15-web-bundle-and-widget-substrate-design.md`),
D1 (`2026-08-16-frontier-widget-and-substrate-hardening-design.md`),
D2 (`2026-08-19-displacement-field-widget-design.md`),
D3 (`2026-08-20-region-grow-and-screen-map-design.md`).

Piece E is the last piece before F. It ships three things: a keyed, page-shared, URL-synced
`StateFactory`; the mount-contract change that makes the keying type-checked rather than
string-keyed; and `docs/explore.md`, a walkthrough of **one block descending all five stages**.

---

## §1 Measurements

Everything below was measured on this branch's checkout, not assumed.

### §1.1 The chain is factually coherent today, with no new bakes

Every stage's shipped artifact pins the **same block**:

| Stage | Artifact | Block |
| --- | --- | --- |
| Screening | `examples/screen-map/capetown.json` | 16,451 blocks; `ZAF.9.3.1_1_40972` is index **9511** |
| Growth | `examples/region-grow/hood.json` | `seed = "ZAF.9.3.1_1_40972"` (213-block hood) |
| Permeability | `examples/perm-graph/bundle.json` | `block_id = "ZAF.9.3.1_1_40972"` |
| Displacement | `examples/displacement-field/field.json` | `block_id = "ZAF.9.3.1_1_40972"` |
| Methods | `examples/method-comparison/frontier.json` | `block_id = "ZAF.9.3.1_1_40972"` |

`web/src/hood.d.ts:38-39` already documents this ("The pinned seed -- the same block PermGraph,
Frontier and DisplacementField use"). So the Explore page is a real walkthrough of one block, not
five figures stacked, and it needs **no** per-block bundle work. The full-prefix-table long pole
(backlog: "the bundle needs a *full* prefix table … per block per method") stays deferred, and E
does **not** unblock arbitrary-block exploration.

### §1.2 Payload: Explore costs +0.14 MB gz over a page that already exists

Gzip -9, measured:

| Bundle | raw | gz |
| --- | --- | --- |
| `screen-map/capetown.json` | 5.88 MB | **1.94 MB** |
| `screen-map/nairobi.json` | 1.11 MB | **0.37 MB** |
| `region-grow/hood.json` | 0.09 MB | 0.02 MB |
| `perm-graph/bundle.json` | 0.26 MB | 0.10 MB |
| `displacement-field/field.json` | 0.04 MB | 0.01 MB |
| `method-comparison/frontier.json` | 0.02 MB | 0.01 MB |
| **all six together** | | **2.45 MB** |

The Screening page already fetches the first three (**2.33 MB gz**) today. Explore therefore costs
**+0.14 MB** over an existing page. **No lazy-loading / IntersectionObserver machinery is in
scope** — it would be a mechanism built for a problem the measurement says does not exist.

### §1.3 All thirteen state fields are distinct — today

| Widget | state interface | fields |
| --- | --- | --- |
| `PermGraph` | `PermGraphState` | `prefix`, `layer`, `halos` |
| `Frontier` | `FrontierState` | `targetDisplacement`, `targetPermeability`, `isolated` |
| `DisplacementField` | `FieldState` | `roads`, `second` |
| `RegionGrow` | `RegionGrowState` | `seed`, `budget` |
| `ScreenMap` | `ScreenState` | `city`, `metric`, `floor` |

Thirteen fields, no collision. That is **luck, not design** — nothing enforces it, and the Screening
page already carries two widgets. §4 keeps URL keys flat and readable (which is the point of a
citable URL) and closes the hazard with a mount-time throw rather than by relying on the accident.

### §1.4 A city block is sub-pixel: the spine marker cannot be an outline

D3 measured the median Cape Town block at **0.61 CSS px²** on the shipped canvas, with 69.6% under
1 px². Outlining block 9511 would be invisible. The marker must be a **fixed-screen-size ring at the
block's centroid** (§6), sized in pixels rather than in world units.

### §1.5 Widget control initialisation — where a cited URL would visibly desync

Every widget except two builds its controls from `state.get()` at boot, so a URL-supplied initial
value already reaches the control for free:

| Widget | control initial | reads |
| --- | --- | --- |
| `PermGraph` | `perm-graph.ts:87,101,112` | `state.get()` ✅ |
| `ScreenMap` | `screen-map.ts:231,232` (`metric`, `city`) | `state.get()` ✅ |
| `ScreenMap` | `screen-map.ts:125` via `syncFloor` (`floor`) | **the bundle, before `makeState`** ❌ |
| `Frontier` | `frontier.ts:335,354` | `state.get()` ✅ (and writes back at `453-454`) |
| `RegionGrow` | `region-grow.ts:101` | **`b.budget.default`** ❌ |
| `DisplacementField` | `displacement-field.ts:129` | **`b.width.default_m`** ❌ |

The three ❌ rows are the whole of the "E touches no widget" claim being wrong (the backlog entry is
mine and is corrected by this document). Without them `?budget=5000` draws the region at 5,000 while
the slider reads 3,000.

**Two of the three are one-line fixes. `ScreenMap`'s floor is not**, and §1.6 says why.

**`width` is already in state.** `displacement-field.ts:130-138` writes the slider's value onto
`width_m` of **every** road (live or not, deliberately — "two coincident roads of one width are
algebraically one road"). So the corridor width is `roads[*].width_m` and needs no state-shape
change; only the slider's initial is wrong.

### §1.6 `ScreenState.floor` needs a model fix, not a write-back

`syncFloor` (`screen-map.ts:106-127`) computes the floor's live **bounds** for a (bundle, metric)
pair, resolves a value, writes all four slider attributes, and returns the value. `boot` calls it
*before* `makeState`, so a URL-supplied `floor` never reaches the slider — that is the desync above.
But re-applying the value afterwards is not sufficient, because **the floor's default is a function
of the metric**:

| URL | today's `initial` | what a write-back would do |
| --- | --- | --- |
| — | `depth_density_proxy`'s 0.0128 | correct |
| `?floor=X` | 0.0128, overridden to X | correct |
| `?metric=density` | 0.0128, metric now `density` | **0.0128 clamped into `density`'s range** |
| `?metric=density&floor=X` | X | correct |

The four metrics have unrelated score scales, so row 3 clamps to `density`'s minimum and selects all
16,451 blocks — a silently wrong picture from a URL that asked for nothing unusual. Resolving it by
asking "did the URL set `floor`?" would put a `fromUrl(key)` member on `StateSource<T>` for one
widget's benefit.

**The fix is the honest model: `floor: number | null`, where `null` means "this metric's own
default".** That is already exactly what `syncFloor`'s `preferred` parameter means, so all four rows
come out right with no new interface member, and an untouched floor emits no URL key at all.

Two consequences, both improvements:

* The metric `<select>` handler sets `floor: null` instead of a computed number, so switching metric
  resets to the new metric's calibration **and drops `?floor=` from the URL**.
* The city toggle must **resolve** the floor at the moment of the switch (`state.set({ city, floor:
  resolved })`), preserving the deliberate behaviour `syncFloor`'s docstring argues for — an
  absolute floor carries across corpora rather than being redefined. Pinning the number there is
  also honest: the reader chose to carry it, so the URL should say so.

`render` resolves `st.floor ?? defaultFloorFor(bundle, st.metric)`, where `defaultFloorFor` is the
`shipped ?? floorAtShippedPoolSize` half factored out of `syncFloor` so both resolve identically.

---

## §2 The store — `web/src/url/`

### §2.1 Shape

One `UrlStore` per page, constructed in `mountAll`. It owns the parsed query string, the set of
bound widgets, and one debounced write. Each widget gets a `StateSource<T>` through
`store.bind(codec, initial)` that is indistinguishable from `localState`'s.

```ts
// web/src/url/param.ts
export interface Param<V> {
  /** The query-string keys this param owns. DECLARED, not derived from the field name: renaming a
   * TypeScript field must not break a URL somebody published. Usually one; `roads` owns three. */
  readonly keys: readonly string[];

  /** Only the keys whose value differs from `initial`. The store never calls this when
   * `same(v, initial)` holds, so `{}` back from here means a param whose `same` disagrees with its
   * own `encode` -- a defect in the param, not a state the store has to represent. */
  encode(v: V, initial: V): Record<string, string>;

  /** `present` carries only THIS param's keys that the URL actually had -- possibly a subset, so a
   * hand-edited `?width=9` alone is decodable against `initial`. `null` means the URL said
   * something this widget cannot use: the store falls back to `initial` and DROPS every key of
   * this param from the URL, so the reader watches their typo disappear instead of being handed a
   * broken figure or a silent lie. */
  decode(present: Readonly<Record<string, string>>, initial: V): V | null;

  /** Value equality, declared per param because `roads` is an array and `===` on it would rewrite
   * the URL on every render. */
  same(a: V, b: V): boolean;
}

export type UrlCodec<T> = { readonly [K in keyof T]-?: Param<T[K]> };
```

`UrlCodec<T>` is a **mapped type over the state interface**. Adding a field to `ScreenState` without
adding a codec entry is a compile error; renaming one is a compile error at the codec. That is the
checker doing the auditing (owner directive, 2026-08-11) rather than a `Record<string, unknown>`
table that type-checks while silently dropping a field.

### §2.2 Rules

1. **Only values differing from `initial` are written.** An untouched page keeps whatever query
   string it arrived with; a citation carries only what the reviewer changed. Today's links keep
   working unchanged.
2. **Unknown params are preserved verbatim** — `utm_*`, a tracking tag, someone else's decoration.
   The store rewrites only keys some codec on the page claims.
3. **Invalid values self-correct.** `decode` → `null` ⇒ the widget uses its `initial` and the
   offending keys are dropped on the next write. A bind that dropped anything schedules a write
   immediately, so the URL corrects itself even if the reader never touches a control.
4. **`replaceState` only, 300 ms trailing debounce.** A drag writes once, when it settles. No
   history entries ⇒ no `popstate` ⇒ **no state→control write-back is needed in the four widgets
   that lack it**. It also stays under Safari's `replaceState` rate limit (≈100 calls / 30 s), which
   a per-`pointermove` write would exceed inside one drag.
5. **Deterministic emission order:** every unclaimed param first, in its original order, then every
   claimed param that differs from its initial, in mount order and then codec-declaration order.
   Stable across writes and directly testable.

### §2.3 Static validation vs bundle validation — the division, stated once

A codec is constructed at module load; the bundle arrives later over the network. So the two halves
of "is this URL value usable?" live in two places, and the rule is uniform across all five widgets:

* **The codec validates what is knowable statically** — type, enum membership, sign, arity,
  coordinate count. Failure ⇒ `decode` returns `null` ⇒ initial + the keys are dropped (§2.2 rule 3).
* **The widget validates what depends on its fetched bundle** — a `seed` naming no block, an
  `isolated` naming no method, a `prefix` past the last one — at boot, by **resetting that field to
  its initial**. Because the field then equals the initial, the store stops emitting its key and the
  URL self-corrects through the ordinary write path. No second mechanism.

A widget-side reset must never *throw*: `region-grow.ts:75-78` throws when the **bundle's** own seed
is unknown, and that stays — a broken artifact is a real failure. A reader's typo in a query string
is not, and must land on the default view rather than on an error card.

Concretely this adds **five** boot-time resets (§7): `RegionGrow.seed` unknown ⇒ the bundle's own
seed; `Frontier.isolated` naming no method in `b.methods` ⇒ `null` (today `frontier.ts:425` would
filter out **every** curve and draw an empty chart); `PermGraph.prefix` past `n_prefixes` ⇒ clamped;
and — **added during Task 4, after this section first said three** — `RegionGrow.budget` and
`DisplacementField`'s road width clamped to their own bundle bounds.

The last two were not foreseen here and were ruled in during execution, because the widgets' own new
comments asserted the property the gap violated: `?width=99999` reaches state, a real
`<input type="range">` clamps its displayed value to the element's `max`, and the corridor is drawn
at the unclamped metres — the slider and the picture disagreeing, which is exactly the desync
`§1.5`'s whole argument is about. Narrowing the comment to fit the gap would have documented a bug
as a design.

**One residual is known and deliberately not fixed.** Both sliders also carry a `step` the browser
snaps onto (`budget.step` 50, `width.step_m` 0.5) and no codec knows it, so `?budget=5001` shows
5000 on the control while the picture draws 5001. Snapping in the codec would silently rewrite a
reader's typed value, which is worse than a sub-step disagreement; the widgets' comments therefore
describe the **bound** only and claim nothing about the grid.

**Precedence:** a URL param beats the mount point's `data-*` attribute. `PermGraph` is the only
widget whose initial comes from `data-*` (`initialState(host)`), and the store overlays the query on
top of whatever initial it is handed, so `?prefix=3` wins over `data-prefix="14"`. The `data-*` value
remains what the URL is measured against for the default-omission rule.

### §2.4 Injected seams, resolved upstream

Per the owner's injection directive, nothing downstream reaches for a global:

```ts
// web/src/url/store.ts
export interface UrlLocation { search(): string; replace(search: string): void }
export function browserLocation(): UrlLocation;          // location.search + history.replaceState

export type Scheduler = (write: () => void) => void;
export interface Timers { set(fn: () => void, ms: number): number; clear(id: number): void }
export function debounce(ms: number, timers: Timers): Scheduler;
export const systemTimers: Timers;

export interface UrlStore { bind<T>(codec: UrlCodec<T>, initial: T): StateSource<T> }
export function urlStore(loc: UrlLocation, schedule: Scheduler): UrlStore;
```

`mountAll` builds `urlStore(browserLocation(), debounce(300, systemTimers))` once. Tests construct a
`UrlStore` over a fake `UrlLocation` and a synchronous `Scheduler`; `debounce` is tested separately
against a fake `Timers`. Neither `location`, `history` nor `setTimeout` is stubbed on `globalThis`.

**`StateSource.subscribe` still has no unsubscribe, and that stays safe for exactly one reason** —
widgets live as long as their page, because `navigation.instant` is off (§9). A `UrlStore` bound to
a widget that outlived its page would need one.

---

## §3 The mount contract

### §3.1 Why the current factory cannot be keyed

`StateFactory = <T>(initial: T) => StateSource<T>` is universally quantified: the **widget** picks
`T` at its own call site, so `register` has no `T` to pair a codec against, and any keying would
have to be a runtime string table. Narrow it so the type flows the other way:

```ts
// web/src/state.ts
export type StateFactory<T> = (initial: T) => StateSource<T>;

// web/src/mount.ts
export type Widget<T> = (host: HTMLElement, makeState: StateFactory<T>) => void;

interface Registration {
  readonly keys: readonly string[];
  mount(host: HTMLElement, store: UrlStore): void;
}

export function register<T>(name: string, w: Widget<T>, codec: UrlCodec<T>): void;
```

`register` captures `T` in a closure — `mount: (host, store) => w(host, (initial) =>
store.bind(codec, initial))` — and stores a non-generic `Registration`. `REGISTRY` stays a plain
`Map<string, Registration>`; the (widget, codec) pairing is checked at each of the five `register`
call sites.

`Registration.keys` is `Object.values(codec).flatMap(p => p.keys)` — a loop over a **declared**
mapped type, which is the allowed form of dynamic access, not a string lookup into a closed set.

### §3.2 The collision throw

`mountAll` keeps a `Map<urlKey, widgetName>` across the mount points it visits and throws when two
claim the same key. It sits **inside** the existing per-mount-point `try`, so the failure renders on
the page like every other mount failure rather than being console-only behind an intact PNG — the
same reasoning that moved the unknown-name lookup inside that `try` in D1.

Two mount points of the *same* widget on one page collide too, and should: they would otherwise
cross-talk silently through one set of query keys.

### §3.3 Cost

Five `Widget` → `Widget<XState>` annotations, five `StateFactory` → `StateFactory<XState>` in each
`boot`, five state interfaces exported. No behaviour changes, and `localState` still satisfies the
narrowed factory, so `web/test/*-boot.test.ts` need no edits.

---

## §4 The URL grammar

| Widget | field | key(s) | value |
| --- | --- | --- | --- |
| `ScreenMap` | `city` | `city` | `capetown` \| `nairobi` |
| | `metric` | `metric` | one of the four `MetricName`s |
| | `floor` | `floor` | number, 6 significant figures; **absent ⇒ the metric's own default** (§1.6) |
| `RegionGrow` | `seed` | `seed` | **block_id** (see below) |
| | `budget` | `budget` | integer |
| `PermGraph` | `prefix` | `prefix` | non-negative integer |
| | `layer` | `layer` | `conductance` \| `current` |
| | `halos` | `halos` | `0` \| `1` |
| `DisplacementField` | `roads` | `road1`, `road2`, `width` | `x1,y1,x2,y2` at 0.1 m; width at 0.1 m |
| | `second` | `road2on` | `0` \| `1` |
| `Frontier` | `targetDisplacement` | `disp` | number |
| | `targetPermeability` | `perm` | number |
| | `isolated` | `method` | method slug, or the key absent for `null` |

Example, and the whole point of the piece:

```
/explore/?floor=0.0102&budget=6000&prefix=14&width=12&disp=0.05
```

### §4.1 `seed` is a block_id, not an array index

`RegionGrowState.seed` is currently a **position** in `hood.json`'s 213 blocks. An array index in a
published URL points at a *different block* after any re-bake that reorders the hood — no error,
right type, right shape, wrong value: this project's own static-checkability incident pattern one
level up from the language.

`hood.json`'s own `seed` field is already a `block_id`, and `region-grow.ts:75-78` already resolves
it with `findIndex` and a boundary error. This design moves that conversion one hop later: state
holds the id, the two places that need a position convert at the boundary (`blockAt` returns an
index → take its `block_id`; `render`/`growth` take the id → `findIndex`). `model/accretion.ts`'s
index-based `growth()` API and `hood.json`'s index-based `reference` fixtures are **unchanged**.

### §4.2 `roads` is one field over three keys

`Road` is `{ coords: [number, number][]; width_m: number }` and the bundle ships exactly two roads
of exactly two points each — an invariant `boot` already validates and `liveIndices` already relies
on with literal `0`/`1`. The param asserts the same on decode and returns `null` otherwise.

Splitting the width into its own key is deliberate: it is the knob a reader is most likely to
hand-edit, and `encode(v, initial)`'s per-key diff means a width-only change emits `?width=12`
rather than the full geometry.

Coordinates are origin-relative metres at **0.1 m**, one decimal. The bundle carries 2 dp; 10 cm is
below anything visible in a drag, keeps the URL short, and is idempotent after one round trip.

---

## §5 The Explore page

### §5.1 Files and wiring

* `docs/_partials/explore.md` — handwritten prose with marker holes (tracked).
* `docs/explore.md` — generated, **added to `.gitignore`** beside `docs/index.md` and
  `docs/reproduce.md`.
* `scripts/gen_site_pages.py` — `_write_page(DOCS / "explore.md", _render_partial("explore"),
  depth=0, url_depth=1, title="Explore")`, written **directly into `docs/`** like `reproduce.md`, so
  it is not subject to the `rmtree`/`mkdir` ordering hazard (ruling F4) that `results/` and
  `methodology/` carry.
* `mkdocs.yml` — `- Explore: explore.md`, between Background and Methodology, matching the parent
  design's §1 nav.
* `scripts/gen_site_pages.py`'s `generated_pages` list (the input to
  `_assert_widget_bundle_present`) **must gain `docs/explore.md`**. It currently enumerates
  `index.md`, `reproduce.md`, `methodology/**` and `results/**` by hand; leaving Explore out would
  make the guard blind to the page carrying the **most** mount points — a build with no
  `docs/js/widgets.js` would ship five dead widgets behind five intact PNGs, which is this branch's
  signature defect.

### §5.2 Structure

Five `##` sections in pipeline order, each with an explicit `attr_list` id so the anchors do not
depend on heading wording: `{#screening}`, `{#growth}`, `{#permeability}`, `{#displacement}`,
`{#methods}`.

Above them, a **static numbered strip** — an `<ol>` of five anchors in
`<nav class="sbu-stage-rail" aria-label="Pipeline stages">`, styled in `docs/stylesheets/sbu.css`.
No JavaScript and no scrollspy: Material's `toc.follow` is already enabled and already scroll-follows
the right-hand TOC, which is the behaviour the parent design's "stage rail" was asking for. A second
`IntersectionObserver` per page to duplicate it would be mechanism without a gap to fill.

Each section's framing prose says what the stage asks and what the reader should try; **every
number on the page comes from a producer**, never typed — the same rule the rest of the site is
held to.

### §5.3 Figures are the existing producers

Explore reuses the *same* producers the methodology pages use, so no second caption quotes the same
artifact numbers and there is nothing to drift. Two generator changes:

1. **`_perm_graph_widget_figure()`** — the Permeability grid is four panels at ~1.7 MB each and only
   the current/after panel carries the mount point. Factor that panel out of `_perm_graph_figures`
   so both call one caption builder, and expose it as its own marker (`PERMGRAPHWIDGET`).
2. **`FRONTIER` joins `MARKERS`.** `_frontier_figure` exists but is called directly from
   `gen_benchmark_section()`. Registering it makes it available to `explore.md`, and
   `tests/test_gen_site_pages.py`'s existing both-directions marker test then covers it for free.

Markers used by `explore.md`: `SCREENMAP`, `REGIONGROW`, `PERMGRAPHWIDGET`, `DISPFIELD`, `FRONTIER`.

---

## §6 The spine marker

### §6.1 What is baked

`scripts/gen_screen_map.py` reads the followed block's id from `examples/perm-graph/bundle.json`
(`block_id`) — derived, never typed — and bakes into each city bundle:

```ts
export interface CityFollow {
  /** The block every later stage of the site is about. Read from perm-graph/bundle.json. */
  block_id: string;
  /** Index into the column arrays -- so the widget needs no search. */
  index: number;
  /** Ring centroid, in the same origin-relative encoded metres as `rings`. Baked, not derived in
   * JS, so the canvas marker and the PNG marker cannot land in two different places. */
  x: number;
  y: number;
}
```

`follow?: CityFollow` — **absent for Nairobi**, exactly as `informal` is absent rather than a null
column ("a null column is a field that looks answerable and is not"). The baker raises if the id is
not among a city's blocks *for the city that should have it*, rather than silently omitting.

`CityEncoding` gains `follow_color`, sourced from `reblock.render._ROAD_COLOR` (`#1E90FF`): the
site's one blue, and the only palette constant distinct from `base_color` `#dddddd`,
`selected_color` `#c0392b` and `informal_color` `#d98c00` at a glance.

### §6.2 How it is drawn

**A ring at fixed screen size, not a block outline** (§1.4): a circle of radius 6 CSS px stroked at
2 px in `follow_color`, centred on `follow.x/​y`, painted on the **frame** layer (above the base
blit), so a floor or metric change never re-touches it.

`gen_screen_map.py`'s `_render_screen_map` draws the same ring, in the same colour, at a matching
size in the PNG. That is this branch's standing rule — one `encoding`, feeding the matplotlib
fallback and the canvas widget alike — and it is why the marker appears on the **Screening** page as
well as Explore: one artifact, one truth, and the through-line ("this is the block the rest of the
site follows") is worth having on both.

The marker is a **fixed fact about the site's spine, not a view parameter**, so it is bundle state
and takes no URL key.

---

## §7 Widget touches — the complete list

1. `web/src/widgets/region-grow.ts` — slider initial from `state.get().budget` (§1.5).
2. `web/src/widgets/displacement-field.ts` — width slider initial from
   `state.get().roads[0]!.width_m` (§1.5).
3. `web/src/widgets/region-grow.ts` — `RegionGrowState.seed` becomes a `block_id` (§4.1).
4. All five widgets — `Widget<XState>` / `StateFactory<XState>` annotations, state interfaces
   exported, one exported `UrlCodec` each (§3.3).
5. `web/src/widgets/screen-map.ts` + `web/src/render/city.ts` — draw the follow ring (§6.2).
6. `web/src/mount.ts` — `register`'s third argument, the store, the collision throw (§3).
7. Three boot-time resets for bundle-dependent URL values (§2.3): `region-grow.ts` (unknown
   `seed` ⇒ the bundle's own), `frontier.ts` (unknown `isolated` ⇒ `null`), `perm-graph.ts`
   (`prefix` past the last ⇒ clamped).
8. `web/src/widgets/screen-map.ts` — `ScreenState.floor` becomes `number | null`, `defaultFloorFor`
   is factored out of `syncFloor`, the metric handler sets `null`, the city handler resolves, and
   `render` resolves (§1.6). This is the one non-mechanical widget change in the piece.

---

## §8 Testing

Node's built-in runner, `web/test/`, through the existing `web/test/harness.ts`. No new fake DOM.

* **`Param` round-trips**, per primitive: `decode(encode(v, init), init) === v`, and the
  default-omission rule (`same(v, init)` ⇒ the store emits no key).
* **Per-widget codec**: encode∘decode identity on a non-default state for each of the five, and a
  compile-level check that the codec covers every field (the mapped type does this; the test pins
  the key list so a *silent key rename* is also caught).
* **Garbage input**: `?metric=bogus&floor=NaN&seed=nope&road1=1,2&halos=maybe` ⇒ every widget at its
  initial, every bad key dropped from the emitted query, and an unclaimed `utm_source` preserved.
* **Emission order** (§2.2 rule 5), asserted on a query with unclaimed params interleaved.
* **Key collision**: a fake page with two mount points claiming one key ⇒ throws, and the message
  renders through `showWidgetError` rather than only to the console.
* **`debounce`** against a fake `Timers`: N `set` calls inside the window ⇒ exactly one write, with
  the *last* value.
* **`ScreenMap` specifically** — the widget D3's path guard was structurally blind to: assert
  `docs/explore.md` carries `data-bundle-capetown` and `data-bundle-nairobi` rewritten for
  `url_depth=1`, in `tests/test_gen_site_pages.py`.
* **`ScreenMap`'s floor default** (§1.6), the row a write-back would have got wrong:
  `?metric=density` alone ⇒ `density`'s **own** calibrated floor, not 0.0128 clamped, and the pool
  is not all 16,451 blocks. Plus `?metric=density&floor=X` ⇒ X, and a metric switch after a manual
  floor drag ⇒ `?floor=` gone from the emitted query.
* **Bundle-dependent resets** (§2.3), one per widget: `?seed=NOPE` ⇒ the hood's own seed and the
  key gone; `?method=nope` ⇒ every curve still drawn, not an empty chart; `?prefix=99999` ⇒ clamped
  to the last prefix. Each asserts the *emitted query*, not only the state.
* **Bake guard**: `tests/test_screen_map_bundle.py` gains an assertion that `follow.block_id` equals
  `perm-graph/bundle.json`'s `block_id`, that `follow.index` indexes that id in `block_id`, and that
  Nairobi has no `follow` key at all.

**Every guard is proved by fault injection** — break the thing it guards, observe RED, restore. An
injection that will not redden is *reported*, not tuned. This branch has now produced three tests
that passed while guarding nothing; the acceptance criterion is the injection, not the green tick.

---

## §9 Out of scope, and one trap

* **`navigation.instant` stays OFF.** URL-as-state is precisely the feature that makes it tempting.
  Two documented mechanisms fail the moment it is on: `mount.ts` mounts from `DOMContentLoaded`,
  which never fires again once navigations are fetch+DOM-replacement, so every widget after the
  first page is silently dead behind an intact PNG; and `dom/resize.ts`'s `observeSize` disposer has
  no caller *because* instant is off, so every navigation would leak one `ResizeObserver` per widget
  firing against detached elements. Both files carry the full reasoning. **Do not touch
  `mkdocs.yml`'s `features:` list.**
* **No `popstate`, no history entries.** Back-to-undo-a-slider is not a behaviour anyone expects,
  and skipping it is what keeps four widgets free of state→control write-back (§2.2 rule 4).
* **No arbitrary-block exploration.** That needs the full per-block prefix tables (the backlog's
  long pole) and is not unblocked by this piece.
* **No lazy bundle loading** (§1.2).
* **No Pyodide** — piece F.

---

## §10 File structure

**New**

| File | Responsibility |
| --- | --- |
| `web/src/url/param.ts` | `Param<V>`, `UrlCodec<T>`, and the primitives (`intParam`, `numberParam`, `boolParam`, `enumParam`, `nullableSlugParam`, `roadsParam`) |
| `web/src/url/store.ts` | `UrlLocation`, `browserLocation`, `Timers`, `systemTimers`, `debounce`, `UrlStore`, `urlStore` |
| `web/test/url-param.test.ts` | round-trips, garbage input, default omission |
| `web/test/url-store.test.ts` | bind/emit order, preservation, self-correction, debounce |
| `docs/_partials/explore.md` | the walkthrough prose (tracked) |

**Modified**

| File | Change |
| --- | --- |
| `web/src/state.ts` | `StateFactory<T>` narrowed |
| `web/src/mount.ts` | `Widget<T>`, `register(name, w, codec)`, `Registration`, store construction, collision throw |
| `web/src/widgets/*.ts` (×5) | annotations, exported state interfaces, exported codecs, the two slider initials, `seed` as block_id, the follow ring |
| `web/src/render/city.ts` | the follow ring on the frame layer |
| `web/src/screen_map.d.ts` | regenerated: `CityFollow`, `follow?`, `follow_color` |
| `web/test/mount.test.ts` | `register`'s third argument; the collision test |
| `scripts/gen_screen_map.py` | bake `follow`, `follow_color`; draw the ring in the PNG |
| `scripts/gen_site_pages.py` | `_perm_graph_widget_figure`, `FRONTIER` + `PERMGRAPHWIDGET` markers, the Explore page |
| `docs/stylesheets/sbu.css` | `.sbu-stage-rail` |
| `mkdocs.yml` | nav entry only |
| `.gitignore` | `docs/explore.md` |
| `tests/test_gen_site_pages.py`, `tests/test_screen_map_bundle.py` | §8 |

---

## §11 Global constraints (binding on every task)

* `scripts/gen_site_pages.py` stays **stdlib-only** and must **never** import `reblock`.
* `docs/js/` and `docs/assets/` are gitignored — never stage them.
* Generated bundles and their `.d.ts` are **generated and committed, never hand-edited**.
* No `# type: ignore`, no mypy excludes, no unreachable guards as fixes.
* Never reach into a closed, known-at-authoring-time set with a runtime string, position or count;
  dynamic access over a genuinely open set has **no default**.
* No legacy-compatibility shims — migrate the data and delete the old path.
* Every generated number is read from an artifact, never typed into prose.
