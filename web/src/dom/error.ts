/** The one place a widget failure becomes something a reader can see.
 *
 * There were three near-identical copies of this -- `mount.ts`'s `showMountError` and a `showError`
 * in each widget -- and they had already diverged on the sentence that matters most: only the
 * mount.ts copy told the reader "the static image above still applies", which is the whole reason
 * the fallback `<img>` stays in the generated markup and is removed only after a successful draw.
 * A reader landing on either of the other two paths was told a figure had failed and nothing about
 * the picture still sitting above it. Divergence already realised is the cost of duplication already
 * paid, so the copies are gone.
 *
 * It lives in its own module, NOT in mount.ts, and that is load-bearing: a widget importing
 * `mount.js` at runtime is the circular import that once made the whole bundle throw during module
 * evaluation while the page still looked fine (see mount.ts's registration comment). This module
 * imports nothing.
 */
export function showWidgetError(host: HTMLElement, label: string, err: unknown): void {
  const cause = err instanceof Error ? err.message : String(err);
  const message = `${label} could not load interactively (${cause}). `
    + `The static image above still applies.`;
  // The figcaption if there is one -- the failure then reads where the caption a reader is already
  // looking at was -- otherwise a paragraph appended to the mount point itself.
  const caption = host.querySelector("figcaption");
  if (caption) {
    caption.textContent = message;
  } else {
    const p = document.createElement("p");
    p.textContent = message;
    host.append(p);
  }
}
