/** Every widget's controls, made here so they share one look (docs/stylesheets/sbu.css's "shared
 * controls" section) rather than the browser's defaults or a style per widget: a brand-red range,
 * a bordered select and button, a red-ticked checkbox, a labelled row for each and a readout
 * line, in both colour schemes, each with a visible keyboard focus. The elements stay the native
 * ones -- a real `<input type="range">`, `<select>`, `<input type="checkbox">`, `<button>` -- so
 * the keyboard and assistive technology reach them as they always did.
 *
 * Widgets create controls ONLY through these (web/test/controls.test.ts checks no widget calls
 * `document.createElement` for an input, select or button itself): that is what keeps a new
 * control from arriving in the browser's blue, as the screen map's floor slider once did. */

/** The row of controls under a widget's picture. */
export function controlRow(): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "sbu-controls";
  return el;
}

/** A control's label: its words, then (or before) the control. */
export function controlLabel(...parts: (string | Node)[]): HTMLLabelElement {
  const el = document.createElement("label");
  el.className = "sbu-control";
  el.append(...parts);
  return el;
}

export function rangeInput(): HTMLInputElement {
  const el = document.createElement("input");
  el.type = "range";
  el.className = "sbu-range";
  return el;
}

export function checkboxInput(): HTMLInputElement {
  const el = document.createElement("input");
  el.type = "checkbox";
  el.className = "sbu-check";
  return el;
}

export function selectInput(): HTMLSelectElement {
  const el = document.createElement("select");
  el.className = "sbu-select";
  return el;
}

/** A plain button; `extra` adds the widget's own classes (layout, a colour) after the shared one. */
export function controlButton(text: string, extra = ""): HTMLButtonElement {
  const el = document.createElement("button");
  el.type = "button";
  el.className = extra === "" ? "sbu-button" : `sbu-button ${extra}`;
  el.textContent = text;
  return el;
}

/** The line under the controls that says what they now show. */
export function readoutLine(): HTMLParagraphElement {
  const el = document.createElement("p");
  el.className = "sbu-readout";
  return el;
}
