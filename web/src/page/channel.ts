/** A value one widget on a page publishes and others on the same page follow.
 *
 * The mount contract gives every widget its OWN state, bound to its own URL keys, and refuses two
 * widgets claiming one key (mount.ts) -- right for independent figures, wrong for a page where one
 * control decides what several figures show. The Demo page is that page: its block switch picks
 * the block, and the screen-geometry diagram, the screen map's ring and close-up, and the
 * build-out all follow it. So the switch alone owns the `block` URL key and publishes the block
 * here; the followers read it at boot and subscribe.
 *
 * `get()` is `null` until a writer has set it: on a page with no switch, a follower falls back to
 * its own default (its bundle's first block). The switch mounts first in the page's order and
 * sets the channel synchronously, before any follower's bundle fetch has resolved, so on the Demo
 * page a follower never boots on a block the switch did not choose.
 *
 * Like every widget module, this imports nothing from mount.ts. */
export interface Channel<T> {
  get(): T | null;
  set(value: T): void;
  subscribe(fn: (value: T) => void): void;
}

export function createChannel<T>(): Channel<T> {
  let current: T | null = null;
  const listeners: ((value: T) => void)[] = [];
  return {
    get: () => current,
    set(value) {
      if (value === current) return;
      current = value;
      for (const fn of listeners) fn(value);
    },
    subscribe(fn) { listeners.push(fn); },
  };
}

/** The Demo page's block: its full id (`ZAF.9.3.1_1_5810`). */
export const DEMO_BLOCK: Channel<string> = createChannel<string>();
