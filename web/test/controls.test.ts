import { strict as assert } from "node:assert";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import {
  checkboxInput, controlButton, rangeInput, selectInput,
} from "../src/dom/controls.js";
import { installStubs } from "./harness.js";

installStubs();

/** Each widget's source, comments stripped, so a mention in prose does not count. */
const widgets = readdirSync("src/widgets").filter((f) => f.endsWith(".ts")).map((f) => [
  f, readFileSync(`src/widgets/${f}`, "utf8").replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\n]*/g, ""),
] as const);

test("no widget makes its own input, select or button: all come from dom/controls.ts", () => {
  // The screen map's floor slider was a bare `createElement("input")` and drew in the browser's
  // blue; making controls only through the shared helpers is what keeps them in the site's style.
  assert.ok(widgets.length >= 7, "the widget sources were not found");
  for (const [file, code] of widgets) {
    const own = code.match(/createElement\(\s*["'](input|select|button)["']\s*\)/g);
    assert.equal(own, null, `${file} creates its own ${own?.join(", ")}`);
  }
});

test("every widget with a slider, select, checkbox or button uses the shared ones", () => {
  const uses = (code: string, fn: string): boolean => new RegExp(`\\b${fn}\\(`).test(code);
  for (const [file, code] of widgets) {
    if (/\.type = "range"|type: "range"/.test(code)) assert.ok(uses(code, "rangeInput"), file);
  }
  const expect: Record<string, string[]> = {
    "screen-map.ts": ["rangeInput", "selectInput", "checkboxInput", "readoutLine"],
    "region-grow.ts": ["rangeInput", "readoutLine"],
    "displacement-field.ts": ["rangeInput", "checkboxInput", "readoutLine"],
    "frontier.ts": ["rangeInput", "controlButton", "readoutLine"],
    "draw-road.ts": ["controlButton", "readoutLine"],
    "peel-zoom.ts": ["controlButton"],
    "build-out.ts": ["rangeInput"],
  };
  for (const [file, fns] of Object.entries(expect)) {
    const code = widgets.find(([f]) => f === file)?.[1];
    assert.ok(code !== undefined, file);
    for (const fn of fns) assert.ok(uses(code, fn), `${file} does not use ${fn}`);
  }
});

test("the shared controls carry the shared classes and stay native elements", () => {
  const r = rangeInput();
  assert.equal(r.type, "range");
  assert.equal(r.className, "sbu-range");
  assert.equal(checkboxInput().className, "sbu-check");
  assert.equal(checkboxInput().type, "checkbox");
  assert.equal(selectInput().className, "sbu-select");
  const b = controlButton("Go", "sbu-peel__in");
  assert.equal(b.type, "button");
  assert.equal(b.className, "sbu-button sbu-peel__in");
  // And the stylesheet styles each of them, focus included.
  const css = readFileSync("../docs/stylesheets/sbu.css", "utf8");
  for (const cls of ["sbu-range", "sbu-check", "sbu-select", "sbu-button"]) {
    assert.match(css, new RegExp(`\\.${cls}:focus-visible`), cls);
  }
  assert.match(css, /\.sbu-range,\s*\.sbu-check \{\s*accent-color: var\(--sbu-red-fill\);/);
});
