// Run: node --test widget/test/parse.test.js   (tests the widget's pure stream parser; no browser needed)
const test = require("node:test");
const assert = require("node:assert");
const { parseSSEBuffer } = require("../p02-widget.js");

test("complete events are parsed and nothing is left over", () => {
  const r = parseSSEBuffer('event: token\ndata: {"text":"Hi"}\n\nevent: done\ndata: {"text":"Hi."}\n\n');
  assert.deepStrictEqual(r.events, [
    { event: "token", data: { text: "Hi" } },
    { event: "done", data: { text: "Hi." } },
  ]);
  assert.strictEqual(r.rest, "");
});

test("an unfinished event is kept for the next chunk", () => {
  const first = parseSSEBuffer('event: token\ndata: {"text":"A"}\n\nevent: tok');
  assert.strictEqual(first.events.length, 1);
  const second = parseSSEBuffer(first.rest + 'en\ndata: {"text":"B"}\n\n');
  assert.deepStrictEqual(second.events, [{ event: "token", data: { text: "B" } }]);
});

test("a broken or incomplete event is skipped, not thrown", () => {
  const r = parseSSEBuffer('event: token\ndata: {not json}\n\nevent: token\n\ndata: {"a":1}\n\n');
  assert.deepStrictEqual(r.events, []);
});

test("text with markup stays text (the widget only ever uses textContent)", () => {
  const r = parseSSEBuffer('event: token\ndata: {"text":"<img src=x onerror=alert(1)>"}\n\n');
  assert.strictEqual(r.events[0].data.text, "<img src=x onerror=alert(1)>");
});
