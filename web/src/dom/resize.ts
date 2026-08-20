/** Re-run `onSize` whenever `el`'s content box changes to a positive width.
 *
 * Replaces `window.addEventListener("resize", ...)` in both widgets. A window listener misses the
 * case that actually breaks a figure: a CONTAINER narrowing with the window untouched -- Material's
 * nav drawer at some breakpoints, a <details> opening, a tab panel switching, print. There,
 * Frontier's absolute-pixel SVG overflows and PermGraph's canvas stretches a stale backing store.
 *
 * Deliberately NOT a viewBox, which was the recorded plan: a viewBox scales text with the box, so
 * Frontier's 11 px axis labels would land at ~5 px on a 320 px screen. Re-laying out at the
 * measured width keeps type at its designed size and re-nices the ticks for the narrower span.
 *
 * A zero width means "not laid out yet" (a hidden container, a collapsed tab), so it is SKIPPED
 * rather than drawn or thrown on. Skipping is only safe because both widgets now remove their
 * fallback <img> after a successful draw: nothing is drawn, so the static figure is still there.
 *
 * `ResizeObserver` needs no shim: it is declared in the checked `lib: ["ES2022", "DOM"]` surface,
 * and it has been unprefixed in every browser this site supports since 2020.
 *
 * THE RETURNED DISPOSER HAS NO CALLER TODAY, and that is a consequence of a decision the project has
 * already recorded rather than an unused hook kept in hope. A widget lives exactly as long as its
 * page because Material's `navigation.instant` is deliberately OFF -- see
 * docs/superpowers/specs/2026-08-13-site-redesign-design.md ("Do not enable Material's
 * `navigation.instant`. It swaps pages without reload and breaks naive widget initialisation") and
 * mount.ts's own DOMContentLoaded comment, which turns on the same fact. With it off there is
 * nothing to dispose: the page unload takes the observer with it. Turn it on and page swaps stop
 * reloading, `mountAll` has to move to Material's `document$` subscription, and observers from the
 * page you navigated AWAY from keep firing against detached elements -- accumulating one per widget
 * per navigation. This return value is the hook that stops that, and it is the reason the two lines
 * exist now rather than being reconstructed under time pressure later.
 */
export function observeSize(el: HTMLElement,
                            onSize: (size: { width: number; height: number }) => void): () => void {
  const obs = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect;
      if (width > 0) onSize({ width, height });
    }
  });
  obs.observe(el);
  return () => obs.disconnect();
}
