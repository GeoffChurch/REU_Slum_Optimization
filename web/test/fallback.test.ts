import { strict as assert } from "node:assert";
import { test } from "node:test";
import { removeFallbackImage } from "../src/dom/fallback.js";

/** `removeFallbackImage` is shared substrate -- both shipped widgets call it, and a third will --
 * so its ONE conditional is worth specifying rather than assuming. That condition ("remove the
 * anchor only when the image was its only content") cannot go false on this site's own markup,
 * because mkdocs-glightbox wraps exactly the one <img> and nothing else; left untested it would be
 * an unfalsifiable guard, which is the shape this branch keeps finding. Here it is instead a stated
 * behaviour with both branches exercised: an anchor that is somebody ELSE's does not get removed by
 * a helper whose job is to remove a fallback image.
 *
 * Minimal fake DOM, same spirit as the other suites: only the four members the module touches
 * (`querySelector`, `parentElement`, `remove`, `children`/`textContent`/`tagName`).
 */
class FakeElement {
  readonly tagName: string;
  children: FakeElement[] = [];
  parent: FakeElement | null = null;
  textContent = "";
  removed = false;

  /** UPPERCASE, as a real element reports it -- the module tests `tagName === "A"`. */
  constructor(tagName: string) { this.tagName = tagName.toUpperCase(); }
  append(...nodes: FakeElement[]): FakeElement {
    for (const n of nodes) { n.parent = this; this.children.push(n); }
    return this;
  }
  get parentElement(): FakeElement | null { return this.parent; }
  remove(): void {
    this.removed = true;
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }
  querySelector(selector: string): FakeElement | null {
    return this.descendants().find((c) => c.tagName === selector.toUpperCase()) ?? null;
  }
  descendants(): FakeElement[] { return this.children.flatMap((c) => [c, ...c.descendants()]); }
  all(tagName: string): FakeElement[] {
    return this.descendants().filter((c) => c.tagName === tagName.toUpperCase());
  }
}

function host(...children: FakeElement[]): FakeElement {
  return new FakeElement("figure").append(...children);
}
const el = (tag: string): FakeElement => new FakeElement(tag);

test("the glightbox anchor goes when the fallback image was all it held", () => {
  // The shape the site actually serves: mkdocs-glightbox's `wrap_img_with_anchor_selectolax` puts
  // the <img> inside an <a class="glightbox"> as its only child, at build time.
  const figure = host(el("a").append(el("img")), el("figcaption"));
  removeFallbackImage(figure as unknown as HTMLElement);
  assert.equal(figure.all("img").length, 0);
  assert.equal(figure.all("a").length, 0,
    "an anchor emptied of its image is invisible, still focusable, and announced as a link with no text");
  assert.equal(figure.all("figcaption").length, 1, "the caption was collateral damage");
});

test("an anchor holding anything else survives -- it is not this helper's anchor to remove", () => {
  // The FALSE branch, which the site's own markup cannot reach. It is not a defensive reflex: this
  // is shared substrate, so "do not remove an element you did not put there" is a real constraint on
  // the helper, and an untested constraint is a guess. Both ways an anchor can hold more are
  // covered, because the condition tests both: another ELEMENT, and bare TEXT.
  for (const [name, extra] of [["an element", el("span")], ["text", null]] as const) {
    const anchor = el("a");
    anchor.append(el("img"));
    if (extra) anchor.append(extra); else anchor.textContent = "Full-size figure";
    const figure = host(anchor);
    removeFallbackImage(figure as unknown as HTMLElement);
    assert.equal(figure.all("img").length, 0, `${name}: the image should still go`);
    assert.equal(anchor.removed, false,
      `${name}: an anchor with other content was removed by a helper that only owns the image`);
    assert.equal(figure.all("a").length, 1, `${name}: the anchor left the document`);
  }
});

test("an image that is not inside an anchor takes nothing with it", () => {
  // The `skip_lightbox` figures on this site: `_figure(skip_lightbox=True)` marks an image as
  // decoration, glightbox's `skip_classes` leaves it unwrapped, and its parent is the <figure>
  // itself. Removing THAT would delete the whole mount point the widget just drew into.
  const figure = host(el("img"), el("figcaption"));
  removeFallbackImage(figure as unknown as HTMLElement);
  assert.equal(figure.all("img").length, 0);
  assert.equal(figure.removed, false, "the mount point itself was removed as if it were an anchor");
  assert.equal(figure.all("figcaption").length, 1);
});

test("a mount point with no fallback image at all is left alone", () => {
  // Reachable: a widget whose figure was authored without an <img>, and every call after the first
  // (both widgets guard the call with a first-draw flag, but the helper must not depend on that).
  const figure = host(el("figcaption"));
  removeFallbackImage(figure as unknown as HTMLElement);
  assert.equal(figure.all("figcaption").length, 1);
  assert.equal(figure.removed, false);
});
