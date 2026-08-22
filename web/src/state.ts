/** The state a widget reads, injected at mount.
 *
 * Generic because widget #2 broke the single-shape assumption: piece C shipped one concrete
 * WidgetState carrying PermGraph's fields, and Frontier needs entirely different ones. Widening one
 * interface per widget would grow it with the widget count and show every widget fields it does not
 * use.
 *
 * Widgets receive a FACTORY rather than constructing their own store, and that is the whole point:
 * piece E can pass a URL-synced factory and no widget learns which one it got.
 */
export interface StateSource<T> {
  get(): T;
  set(patch: Partial<T>): void;
  subscribe(fn: (s: T) => void): void;
}

/** Parameterised by the state it makes, NOT universally quantified over it.
 *
 * The old `<T>(initial: T) => StateSource<T>` let the WIDGET pick `T` at its own call site, which
 * meant nothing could pair a widget with a description of its state -- any URL keying would have
 * had to be a runtime string table. With `T` on the type instead, `register(name, widget, codec)`
 * type-checks the pairing at each of the five call sites, and `UrlCodec<T>`'s mapped type makes a
 * missing or renamed field a compile error. `localState` is still generic, so it satisfies this at
 * every instantiation and every existing boot test keeps working unchanged. */
export type StateFactory<T> = (initial: T) => StateSource<T>;

export function localState<T>(initial: T): StateSource<T> {
  let current = { ...initial };
  const listeners: ((s: T) => void)[] = [];
  return {
    get: () => current,
    set(patch) {
      current = { ...current, ...patch };
      for (const fn of listeners) fn(current);
    },
    subscribe(fn) { listeners.push(fn); },
  };
}
