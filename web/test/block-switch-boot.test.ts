import { strict as assert } from "node:assert";
import { test } from "node:test";
import { createChannel } from "../src/page/channel.js";
import type { StateSource } from "../src/state.js";
import { urlStore } from "../src/url/store.js";
import { fakeLocation, FakeElement, installStubs, writeNow } from "./harness.js";
import {
  BLOCK_SWITCH_URL, blockSwitchOn, type BlockSwitchState,
} from "../src/widgets/block-switch.js";

installStubs();

const BLOCKS = [["5810", "ZAF.9.3.1_1_5810"], ["41785", "ZAF.9.3.1_1_41785"]] as const;

/** A page as the generator writes it: the switch (hidden, one button per block) and, for each
 * block, a passage tagged with its short id, all but the first hidden. */
function page(): { root: FakeElement; host: FakeElement; passages: FakeElement[] } {
  const root = new FakeElement("main");
  const host = new FakeElement("nav");
  host.hidden = true;
  for (const [short, id] of BLOCKS) {
    const b = new FakeElement("button");
    b.dataset.block = short;
    b.dataset.blockId = id;
    host.append(b);
  }
  root.append(host);
  const passages = BLOCKS.map(([short], k) => {
    const div = new FakeElement("div");
    div.dataset.demoBlock = short;
    div.hidden = k > 0;
    root.append(div);
    return div;
  });
  (globalThis as unknown as { document: Record<string, unknown> }).document.querySelectorAll =
    (sel: string): FakeElement[] => root.querySelectorAll(sel);
  return { root, host, passages };
}

function mount(search: string): { host: FakeElement; passages: FakeElement[];
                                  store: StateSource<BlockSwitchState>; channel: ReturnType<
                                    typeof createChannel<string>>;
                                  loc: ReturnType<typeof fakeLocation> } {
  const { host, passages } = page();
  const channel = createChannel<string>();
  const loc = fakeLocation(search);
  const slot = urlStore(loc, writeNow).reserve();
  let store: StateSource<BlockSwitchState> | null = null;
  blockSwitchOn(channel)(host as never, (initial) => {
    store = slot.bind(BLOCK_SWITCH_URL, initial);
    return store;
  });
  assert.ok(store !== null);
  return { host, passages, store, channel, loc };
}

test("it publishes the first block and shows the switch, synchronously at mount", () => {
  const { host, passages, channel } = mount("");
  assert.equal(channel.get(), BLOCKS[0][1], "followers booting later must find the block set");
  assert.equal(host.hidden, false);
  assert.deepEqual(passages.map((p) => p.hidden), [false, true]);
});

test("a URL block is shown, published, and its button pressed", () => {
  const { host, passages, channel } = mount("?block=41785");
  assert.equal(channel.get(), BLOCKS[1][1]);
  assert.deepEqual(passages.map((p) => p.hidden), [true, false]);
  assert.deepEqual(host.findAll("button").map((b) => b.getAttribute("aria-pressed")),
                   ["false", "true"]);
});

test("a click switches block, passages and channel, and writes the URL", () => {
  const { host, passages, channel, loc } = mount("");
  const seen: string[] = [];
  channel.subscribe((v) => seen.push(v));
  host.findAll("button")[1]!.dispatch("click");
  assert.deepEqual(seen, [BLOCKS[1][1]]);
  assert.deepEqual(passages.map((p) => p.hidden), [true, false]);
  assert.equal(loc.written.at(-1), "block=41785");
});

test("a URL naming no block on the page falls back to the first and is rewritten", () => {
  const { channel, loc } = mount("?block=99999");
  assert.equal(channel.get(), BLOCKS[0][1]);
  assert.ok(!loc.search().includes("99999"), loc.search());
});
