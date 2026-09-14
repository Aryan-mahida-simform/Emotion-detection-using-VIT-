import assert from "node:assert/strict";
import test from "node:test";

import { colorForLabel, emojiForLabel, humanizeLabel, normalizeLabel } from "../js/emotions.js";
import { formatBytes, formatMilliseconds, formatPercent, formatTimestamp } from "../js/format.js";

test("formats a probability ratio as a percentage", () => {
  assert.equal(formatPercent(0.9345), "93.5%");
  assert.equal(formatPercent(1), "100.0%");
  assert.equal(formatPercent(1.4), "100.0%");
  assert.equal(formatPercent(-0.2), "0.0%");
  assert.equal(formatPercent(undefined), "—");
});

test("formats a duration in milliseconds or seconds", () => {
  assert.equal(formatMilliseconds(41.7), "42 ms");
  assert.equal(formatMilliseconds(1500), "1.50 s");
  assert.equal(formatMilliseconds(null), "—");
});

test("formats byte sizes", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(4096), "4 KB");
  assert.equal(formatBytes(3 * 1024 * 1024), "3.0 MB");
  assert.equal(formatBytes(-1), "—");
});

test("formats timestamps relatively and absolutely", () => {
  const now = Date.parse("2024-05-01T12:00:00Z");
  assert.equal(formatTimestamp("2024-05-01T11:59:40Z", now), "just now");
  assert.equal(formatTimestamp("2024-05-01T11:30:00Z", now), "30 min ago");
  assert.equal(formatTimestamp("2024-05-01T09:00:00Z", now), "3 h ago");
  assert.equal(formatTimestamp("not a date", now), "—");
});

test("humanises model class labels", () => {
  assert.equal(normalizeLabel("  Happy "), "happy");
  assert.equal(normalizeLabel("disgusted-face"), "disgusted_face");
  assert.equal(humanizeLabel("happy"), "Happy");
  assert.equal(humanizeLabel("disgusted_face"), "Disgusted Face");
  assert.equal(humanizeLabel(""), "Unknown");
});

test("maps the backend's label aliases onto canonical emotions", () => {
  assert.equal(normalizeLabel("happiness"), "happy");
  assert.equal(normalizeLabel("joy"), "happy");
  assert.equal(normalizeLabel("anger"), "angry");
  assert.equal(normalizeLabel("rage"), "angry");
  assert.equal(normalizeLabel("fearful"), "fear");
  assert.equal(normalizeLabel("sadness"), "sad");
  assert.equal(normalizeLabel("calm"), "neutral");
  assert.equal(humanizeLabel("joy"), "Happy");
});

test("gives known emotions a fixed colour and unknown ones a stable colour", () => {
  assert.equal(colorForLabel("HAPPY"), "#ffc53d");
  assert.equal(colorForLabel("fondness"), colorForLabel("fondness"));
  assert.equal(emojiForLabel("angry"), "😠");
  assert.equal(emojiForLabel("fondness"), "🙂");
});
