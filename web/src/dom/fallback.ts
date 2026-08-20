/** Remove the static fallback image, and the link mkdocs-glightbox wrapped it in.
 *
 * `mkdocs-glightbox` wraps every figure image in `<a class="glightbox" href="...png">`. Removing
 * only the <img> leaves an empty anchor: invisible, but focusable, and announced by a screen reader
 * as a link with no text -- which cuts against the accessibility rationale for drawing SVG in the
 * first place. PermGraph ships that on the live site today.
 *
 * The anchor goes only when the image was its only element child: an anchor with other content is
 * somebody else's, and removing it would be a different bug.
 */
export function removeFallbackImage(host: HTMLElement): void {
  const img = host.querySelector("img");
  if (!img) return;
  const anchor = img.parentElement;
  img.remove();
  if (anchor && anchor.tagName === "A" && anchor.children.length === 0
      && (anchor.textContent ?? "").trim() === "") {
    anchor.remove();
  }
}
