/** The page's QR code: "Open this view on your phone", at the bottom of Material's navigation
 * sidebar (owner, 2026-10-29). It encodes the page's canonical URL -- `site_url` plus the page's
 * path, which overrides/main.html puts on the slot -- with the CURRENT query and fragment, so a
 * scan opens the exact view: every slider, floor and toggle the URL state carries. Canonical, not
 * `location.href`: from a local build or the claude.ai preview the QR still opens the live site.
 *
 * It follows the URL state by being told: mount.ts hands the URL store a location that calls
 * `update` after every write (`notifying`, url/store.ts), and those writes are already debounced
 * (`URL_DEBOUNCE_MS`), so a slider drag redraws the code once it settles, not per pixel.
 *
 * Drawn as an SVG of dark modules on a white quiet zone of four modules -- in both colour
 * schemes, since scanners want dark on light -- at medium error correction. Clicking it (a button,
 * so the keyboard reaches it) opens a large copy in a `<dialog>`, for a projector or a poster;
 * Esc or a click closes it. Shown only where the sidebar stays on screen (CSS: 76.25em and up,
 * never in print). */
import qrcode from "qrcode-generator";

/** Where the QR reads the page's address from: the canonical URL (fixed for the page), and the
 * live query and fragment. A seam so a test can drive it without a browser. */
export interface QrSource {
  canonical: string;
  search(): string;
  hash(): string;
}

/** The text the QR encodes: the canonical URL with the current query and fragment. */
export function qrText(src: QrSource): string {
  const q = src.search().replace(/^\?/, "");
  return src.canonical + (q === "" ? "" : `?${q}`) + src.hash();
}

const QUIET = 4;                     // modules of white margin, the QR standard's minimum

/** The code's modules for `text`, row-major, true for dark: the smallest version that holds it at
 * medium error correction, in byte mode (a URL's own characters). */
export function qrModules(text: string): boolean[][] {
  const qr = qrcode(0, "M");
  qr.addData(text, "Byte");
  qr.make();
  const n = qr.getModuleCount();
  return Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => qr.isDark(r, c)));
}

/** The modules as an SVG data URL: a white square with the quiet zone, a dark unit square per dark
 * module, crisp edges at any size. */
export function qrSvg(modules: readonly (readonly boolean[])[]): string {
  const n = modules.length + 2 * QUIET;
  let d = "";
  modules.forEach((row, r) => row.forEach((dark, c) => {
    if (dark) d += `M${c + QUIET} ${r + QUIET}h1v1h-1z`;
  }));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${n} ${n}" `
    + `shape-rendering="crispEdges"><rect width="${n}" height="${n}" fill="#ffffff"/>`
    + `<path d="${d}" fill="#000000"/></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

export interface Qr {
  /** Redraw for the address as it is now. */
  update(): void;
  /** What the code currently encodes. */
  text(): string;
}

/** Fills `slot` (overrides/main.html's, already moved where it shows): the label, the code as an
 * image inside an enlarge button, and the dialog the button opens; draws the current address. */
export function createQr(slot: HTMLElement, src: QrSource): Qr {
  const label = document.createElement("p");
  label.className = "sbu-qr__label";
  label.textContent = "Open this view on your phone";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "sbu-qr__open";
  button.setAttribute("aria-label", "Enlarge the QR code");
  const img = document.createElement("img");
  img.className = "sbu-qr__code";
  button.append(img);
  const dialog = document.createElement("dialog");
  dialog.className = "sbu-qr__dialog";
  const big = document.createElement("img");
  big.className = "sbu-qr__big";
  const close = document.createElement("button");
  close.type = "button";
  close.className = "sbu-qr__close";
  close.textContent = "Close";
  dialog.append(big, close);
  slot.append(label, button, dialog);
  // `showModal` gives Esc-to-close and a focus trap natively; a click anywhere closes it too.
  button.addEventListener("click", () => dialog.showModal());
  dialog.addEventListener("click", () => dialog.close());

  let current = "";
  const update = (): void => {
    const text = qrText(src);
    if (text === current) return;
    current = text;
    const url = qrSvg(qrModules(text));
    for (const el of [img, big]) {
      el.setAttribute("src", url);
      el.setAttribute("alt", `QR code for ${text}`);
    }
  };
  update();
  slot.hidden = false;
  return { update, text: () => current };
}
