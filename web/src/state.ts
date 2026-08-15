/** The state a widget reads, injected at mount.
 *
 * In prose pages this is a local store seeded from the mount point's data-* attributes; piece E
 * swaps in a URL-synced shared store for the Explore page. The widget never learns which it has --
 * that is the whole reason to inject it rather than grow an `if (embedded)` branch per widget, with
 * the branch count climbing alongside the widget count.
 */
export interface WidgetState {
  prefix: number;
  layer: "conductance" | "current";
  halos: boolean;
}

export interface StateSource {
  get(): WidgetState;
  set(patch: Partial<WidgetState>): void;
  subscribe(fn: (s: WidgetState) => void): void;
}

export function localState(initial: WidgetState): StateSource {
  let current = { ...initial };
  const listeners: ((s: WidgetState) => void)[] = [];
  return {
    get: () => current,
    set(patch) {
      current = { ...current, ...patch };
      for (const fn of listeners) fn(current);
    },
    subscribe(fn) { listeners.push(fn); },
  };
}
