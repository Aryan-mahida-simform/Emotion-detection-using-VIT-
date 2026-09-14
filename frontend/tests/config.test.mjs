import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_SETTINGS, normalizeSettings, readQuerySettings } from "../js/config.js";

test("falls back to the defaults for junk settings", () => {
  assert.deepEqual(normalizeSettings({}), DEFAULT_SETTINGS);
  assert.deepEqual(normalizeSettings({ endpoint: "   ", requestTimeoutMs: 12, historyLimit: 0 }), DEFAULT_SETTINGS);
});

test("clamps the warning threshold into the 0-1 range", () => {
  assert.equal(normalizeSettings({ confidenceWarningThreshold: 4 }).confidenceWarningThreshold, 1);
  assert.equal(normalizeSettings({ confidenceWarningThreshold: -2 }).confidenceWarningThreshold, 0);
  assert.equal(normalizeSettings({ confidenceWarningThreshold: "0.55" }).confidenceWarningThreshold, 0.55);
});

test("reads endpoint and threshold overrides from the query string", () => {
  const patch = readQuerySettings("?endpoint=http://localhost:9000/predict&threshold=70&timeout=5000");
  assert.equal(patch.endpoint, "http://localhost:9000/predict");
  assert.equal(patch.confidenceWarningThreshold, 0.7);
  assert.equal(patch.requestTimeoutMs, 5000);
});

test("ignores query values it cannot understand", () => {
  assert.deepEqual(readQuerySettings("?threshold=abc&timeout=-5"), {});
  assert.deepEqual(readQuerySettings(""), {});
});
