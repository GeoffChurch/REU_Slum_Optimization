import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import jsQR from "jsqr";
import { createQr, qrText } from "../src/qr/qr.js";
import { urlStore, notifying } from "../src/url/store.js";
import { BUILD_OUT_URL } from "../src/widgets/build-out.js";
import { fakeLocation, FakeElement, installStubs, writeNow } from "./harness.js";

installStubs();

const CANONICAL = "https://geoffchurch.github.io/REU_Slum_Optimization/demo/";

/** The rendered code read back as a scanner would: the image's SVG modules rasterized at 4 px a
 * module, white where there is no module, and decoded by jsQR. */
function scan(img: FakeElement): string {
  const src = img.getAttribute("src")!;
  const svg = decodeURIComponent(src.slice(src.indexOf(",") + 1));
  const n = Number(/viewBox="0 0 (\d+) \d+"/.exec(svg)![1]);
  const S = 4, W = n * S;
  const px = new Uint8ClampedArray(W * W * 4).fill(255);
  for (const [, x, y] of svg.matchAll(/M(\d+) (\d+)h1v1h-1z/g)) {
    for (let dy = 0; dy < S; dy++) {
      for (let dx = 0; dx < S; dx++) {
        const i = ((Number(y) * S + dy) * W + Number(x) * S + dx) * 4;
        px[i] = px[i + 1] = px[i + 2] = 0;
      }
    }
  }
  const hit = jsQR(px, W, W);
  assert.ok(hit !== null, "the rendered code does not scan");
  return hit.data;
}

test("the QR encodes the canonical URL plus the current state, after a slider change", () => {
  const loc = fakeLocation("");
  let hash = "#comparison";
  const slot = new FakeElement("div");
  const qr = createQr(slot as never, { canonical: CANONICAL, search: () => loc.search(),
                                       hash: () => hash });
  // The page's store, told to redraw the QR after every write, as mount.ts builds it.
  const store = urlStore(notifying(loc, qr.update), writeNow);
  const state = store.reserve().bind(BUILD_OUT_URL, { budget: 5 });
  state.set({ budget: 9 });
  assert.equal(qr.text(), `${CANONICAL}?buildout=9#comparison`);
  const [img, big] = slot.findAll("img");
  assert.equal(scan(img!), qr.text());
  assert.equal(scan(big!), qr.text());
  assert.equal(img!.getAttribute("alt"), `QR code for ${qr.text()}`);
  // And it keeps following: a second change, and the fragment.
  hash = "";
  state.set({ budget: 2 });
  assert.equal(scan(img!), `${CANONICAL}?buildout=2`);
});

test("with no query the QR is the canonical URL alone, never the host it was opened on", () => {
  assert.equal(qrText({ canonical: CANONICAL, search: () => "", hash: () => "" }), CANONICAL);
  assert.equal(qrText({ canonical: CANONICAL, search: () => "?a=1", hash: () => "" }),
               `${CANONICAL}?a=1`);
});

test("the enlarge control is a keyboard-reachable button", () => {
  const slot = new FakeElement("div");
  createQr(slot as never, { canonical: CANONICAL, search: () => "", hash: () => "" });
  const [open] = slot.findAll("button");
  assert.ok(open !== undefined && open.type === "button");
  assert.equal(open.getAttribute("aria-label"), "Enlarge the QR code");
  assert.equal(open.findAll("img").length, 1, "the code is inside the button");
  assert.equal(slot.findAll("dialog").length, 1);
});

test("the QR shows only where the navigation sidebar stays on screen, never in print", () => {
  // Every rule naming .sbu-qr itself: hidden by default, shown only inside the one screen media
  // query at Material's sidebar breakpoint -- so a phone (below it) and print (not screen) hide it.
  const css = readFileSync("../docs/stylesheets/sbu.css", "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
  const shown: string[] = [];
  let media = "";
  let depth = 0;
  for (const m of css.matchAll(/([^{}]*)\{|\}/g)) {
    if (m[0] === "}") { depth--; if (depth === 0) media = ""; continue; }
    const sel = m[1]!.trim();
    if (sel.startsWith("@media")) { media = sel; depth++; continue; }
    depth++;
    if (/(^|[\s,])\.sbu-qr(\s*$|,)/.test(sel)) {
      const body = css.slice(m.index! + m[0].length, css.indexOf("}", m.index!));
      const display = /display:\s*([a-z-]+)/.exec(body)?.[1];
      if (display !== undefined && display !== "none") shown.push(media);
    }
    depth--;
  }
  assert.deepEqual(shown, ["@media screen and (min-width: 76.25em)"]);
  assert.match(css, /\.sbu-qr\s*\{\s*display:\s*none;/);
});
