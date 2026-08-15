import type { Bundle } from "../bundle.js";
import { draw, sizeCanvas } from "../render/canvas.js";
import { register, type Widget } from "../mount.js";
import { fitBbox, nearest, panned, toWorld, zoomed, type View } from "../view/transform.js";
import type { StateSource } from "../state.js";

const permGraph: Widget = (host, state) => {
  const src = host.dataset.bundle!;
  void fetch(src).then((r) => r.json()).then((b: Bundle) => boot(host, state, b));
};

function boot(host: HTMLElement, state: StateSource, b: Bundle): void {
  // The fallback PNG shows clearance's Lens-B prefix; boot anywhere else and the page swaps in a
  // picture the caption below it does not describe.
  state.set({ prefix: b.lens_b_index });

  const fallback = host.querySelector("img");
  const cv = document.createElement("canvas");
  cv.style.width = "100%";
  cv.style.aspectRatio = "1 / 1";
  const controls = document.createElement("div");
  const slider = document.createElement("input");
  slider.type = "range";                    // native: keyboard- and screen-reader-reachable
  slider.min = "0";
  slider.max = String(b.n_prefixes - 1);
  slider.value = String(b.lens_b_index);
  slider.setAttribute("aria-label", "road prefix");
  const readout = document.createElement("p");
  controls.append(slider, readout);
  host.append(cv, controls);
  if (fallback) fallback.remove();

  const xs = b.nodes.cx, ys = b.nodes.cy;
  const bbox = { minX: Math.min(...xs), minY: Math.min(...ys),
                 maxX: Math.max(...xs), maxY: Math.max(...ys) };
  let size = sizeCanvas(cv);
  let view: View = fitBbox(bbox, size.width, size.height);
  const ctx = cv.getContext("2d")!;

  const render = (): void => {
    const s = state.get();
    draw(ctx, b, { view, ...s }, size);
    // Every number the picture shows is also present as text.
    readout.textContent =
      `${b.prefix.road_m[s.prefix]!.toFixed(0)} m of road · ` +
      `${(b.prefix.permeability[s.prefix]! * 100).toFixed(1)}% permeability`;
  };
  state.subscribe(render);

  slider.addEventListener("input", () => state.set({ prefix: Number(slider.value) }));

  let dragging: [number, number] | null = null;
  cv.addEventListener("pointerdown", (ev) => { dragging = [ev.offsetX, ev.offsetY]; });
  cv.addEventListener("pointerup", () => { dragging = null; });
  cv.addEventListener("pointermove", (ev) => {
    if (dragging) {
      view = panned(view, ev.offsetX - dragging[0], ev.offsetY - dragging[1]);
      dragging = [ev.offsetX, ev.offsetY];
      render();
      return;
    }
    const [wx, wy] = toWorld(view, ev.offsetX, ev.offsetY);
    const i = nearest(xs, ys, wx, wy);
    cv.title = `φ = ${b.prefix.potential[state.get().prefix]![i]!.toPrecision(4)}`;
  });
  cv.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    view = zoomed(view, ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.offsetX, ev.offsetY);
    render();
  }, { passive: false });
  window.addEventListener("resize", () => {
    size = sizeCanvas(cv);
    view = fitBbox(bbox, size.width, size.height);
    render();
  });

  render();
}

register("perm-graph", permGraph);
