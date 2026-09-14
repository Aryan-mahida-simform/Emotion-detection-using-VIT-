import assert from "node:assert/strict";
import test from "node:test";

import { addEntry, loadHistory, saveHistory } from "../js/history.js";

function withFakeStorage(run) {
  const store = new Map();
  const previous = globalThis.localStorage;
  globalThis.localStorage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: (key) => store.delete(key),
  };
  try {
    return run(store);
  } finally {
    if (previous === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = previous;
  }
}

const entry = (id, overrides = {}) => ({
  id,
  label: "happy",
  confidence: 0.9,
  timestamp: new Date().toISOString(),
  thumbnail: "data:image/jpeg;base64,AAAA",
  blob: new Blob(["x"]),
  ...overrides,
});

test("addEntry puts the newest record first", () => {
  const first = entry("a");
  const second = entry("b");
  const list = addEntry(addEntry([], first, 5), second, 5);
  assert.deepEqual(
    list.map((item) => item.id),
    ["b", "a"],
  );
});

test("addEntry drops records past the limit", () => {
  let list = [];
  for (const id of ["a", "b", "c", "d"]) list = addEntry(list, entry(id), 2);
  assert.deepEqual(
    list.map((item) => item.id),
    ["d", "c"],
  );
});

test("addEntry replaces a record with the same id instead of duplicating it", () => {
  const list = addEntry(addEntry([], entry("a"), 5), entry("a", { label: "sad" }), 5);
  assert.equal(list.length, 1);
  assert.equal(list[0].label, "sad");
});

test("saved records never carry the image blob into storage", () => {
  withFakeStorage(() => {
    saveHistory([entry("a")], 5);
    const stored = JSON.parse(globalThis.localStorage.getItem("vit-emotion-frontend.history.v1"));
    assert.equal(stored.length, 1);
    assert.equal("blob" in stored[0], false);
    assert.equal(stored[0].thumbnail, "data:image/jpeg;base64,AAAA");
  });
});

test("loadHistory returns records without a blob and survives junk storage", () => {
  withFakeStorage(() => {
    saveHistory([entry("a")], 5);
    const loaded = loadHistory(5);
    assert.equal(loaded.length, 1);
    assert.equal(loaded[0].id, "a");
    assert.equal(loaded[0].blob, null);
  });

  withFakeStorage((store) => {
    store.set("vit-emotion-frontend.history.v1", "{not json");
    assert.deepEqual(loadHistory(5), []);
  });

  withFakeStorage((store) => {
    store.set("vit-emotion-frontend.history.v1", JSON.stringify([{ id: 1 }, null, "nope"]));
    assert.deepEqual(loadHistory(5), []);
  });
});

test("loadHistory respects the limit", () => {
  withFakeStorage(() => {
    saveHistory([entry("a"), entry("b"), entry("c")], 3);
    assert.equal(loadHistory(2).length, 2);
  });
});
