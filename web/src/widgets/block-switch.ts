// Type-only: erased at compile time, so this module has NO runtime import of mount.js (the cycle
// mount.ts's registration comment describes). This file must never import `register`.
import type { Widget } from "../mount.js";
import { DEMO_BLOCK, type Channel } from "../page/channel.js";
import { stringParam, type UrlCodec } from "../url/param.js";

/** The Demo page's block switch: one button per block, written by the page's generator
 * (`data-block`, the short id the URL carries; `data-block-id`, the full one), and the block it
 * picks published on DEMO_BLOCK for the page's figures to follow. It also shows the page's
 * per-block static passages (`[data-demo-block]`, the generator's) for the chosen block and hides
 * the others. The switch is hidden in the markup and unhidden here, so a reader without
 * JavaScript sees the default block's passages and no buttons that do nothing. */
const LABEL = "BlockSwitch";

export interface BlockSwitchState {
  /** The short id (`5810`): the URL's spelling. Which ids exist is the page's markup, not a set
   * this module could know, so it is a string checked against the buttons at boot. */
  block: string;
}

export const BLOCK_SWITCH_URL: UrlCodec<BlockSwitchState> = {
  block: stringParam("block"),
};

/** Show the passages for `short` and hide every other block's. */
export function showBlock(root: ParentNode, short: string): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>("[data-demo-block]"))) {
    el.hidden = el.dataset.demoBlock !== short;
  }
}

/** The widget, publishing on `channel` (DEMO_BLOCK on the page; a fresh one in tests). */
export function blockSwitchOn(channel: Channel<string>): Widget<BlockSwitchState> {
  return (host, makeState) => {
    const buttons = Array.from(host.querySelectorAll<HTMLButtonElement>("button[data-block]"));
    const first = buttons[0];
    if (first === undefined) throw new Error(`${LABEL}: no button[data-block] in the switch`);
    const idOf = new Map<string, string>();
    for (const b of buttons) {
      const short = b.dataset.block!, full = b.dataset.blockId;
      if (full === undefined || full === "") {
        throw new Error(`${LABEL}: ${short} has no data-block-id`);
      }
      idOf.set(short, full);
    }
    const state = makeState({ block: first.dataset.block! });
    // A `?block=` naming no block on this page: back to the first, and the URL rewritten.
    if (!idOf.has(state.get().block)) state.set({ block: first.dataset.block! });
    const render = (): void => {
      const short = state.get().block;
      for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.block === short));
      showBlock(document, short);
      channel.set(idOf.get(short)!);
    };
    for (const b of buttons) {
      b.addEventListener("click", () => state.set({ block: b.dataset.block! }));
    }
    state.subscribe(render);
    render();
    host.hidden = false;
  };
}

export const blockSwitch: Widget<BlockSwitchState> = blockSwitchOn(DEMO_BLOCK);
