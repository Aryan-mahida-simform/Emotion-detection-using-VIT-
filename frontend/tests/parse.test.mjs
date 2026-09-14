import assert from "node:assert/strict";
import test from "node:test";

import { ModelOutputError, normalizeBox, normalizeDistribution, parsePrediction } from "../js/parse.js";

const close = (actual, expected, epsilon = 1e-6) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} is not within ${epsilon} of ${expected}`);

const APP_RESPONSE = {
  label: "happy",
  confidence: 0.93,
  scores: [
    { label: "happy", probability: 0.93 },
    { label: "neutral", probability: 0.06 },
    { label: "angry", probability: 0.01 },
  ],
  probabilities: { angry: 0.01, happy: 0.93, neutral: 0.06 },
  affect: { valence: 0.71, arousal: 0.58 },
  model: {
    backend: "vit",
    model_id: "vit-emotion-base",
    device: "cuda",
    neural: true,
    labels: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
  },
  latency_ms: 41.7,
};

const BACKEND_RESPONSE = {
  label: "sad",
  emoji: "😢",
  confidence: 0.64,
  label_id: 5,
  probabilities: { sad: 0.64, neutral: 0.3, angry: 0.06 },
  scores: [
    { label: "sad", emoji: "😢", probability: 0.64 },
    { label: "neutral", emoji: "😐", probability: 0.3 },
    { label: "angry", emoji: "😠", probability: 0.06 },
  ],
  image: { width: 48, height: 48, mode: "RGB" },
  model_name: "vit-emotion-base",
  device: "cpu",
  elapsed_ms: 88.25,
  face_detected: true,
  face_crop: { x: 12, y: 10, width: 24, height: 26 },
  model_input_size: { width: 224, height: 224 },
};

test("reads the app/ contract response", () => {
  const prediction = parsePrediction(APP_RESPONSE);

  assert.equal(prediction.top.label, "happy");
  close(prediction.top.score, 0.93);
  assert.equal(prediction.distributionReported, true);
  assert.equal(prediction.scores.length, 3);
  assert.equal(prediction.latencyMs, 41.7);
  assert.deepEqual(prediction.affect, { valence: 0.71, arousal: 0.58 });
  assert.deepEqual(prediction.model, {
    name: "vit-emotion-base",
    backend: "vit",
    device: "cuda",
    neural: true,
    labels: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
  });
});

test("reads the backend/ contract response", () => {
  const prediction = parsePrediction(BACKEND_RESPONSE);

  assert.equal(prediction.top.label, "sad");
  close(prediction.top.score, 0.64);
  assert.equal(prediction.top.emoji, "😢");
  assert.equal(prediction.latencyMs, 88.25);
  assert.equal(prediction.model.name, "vit-emotion-base");
  assert.equal(prediction.model.device, "cpu");
  assert.deepEqual(prediction.image, { width: 48, height: 48, mode: "RGB" });
  assert.deepEqual(prediction.modelInput, { width: 224, height: 224 });
  assert.equal(prediction.faceDetected, true);
});

test("turns the backend/ face crop into a drawable box", () => {
  const prediction = parsePrediction(BACKEND_RESPONSE);

  assert.equal(prediction.faces.length, 1);
  assert.deepEqual(prediction.faces[0].box, { x: 12, y: 10, w: 24, h: 26, unit: "pixel" });
});

test("reports no affect when the payload omits it", () => {
  assert.equal(parsePrediction(BACKEND_RESPONSE).affect, null);
  assert.equal(parsePrediction({ scores: { happy: 1 } }).affect, null);
  assert.equal(parsePrediction({ scores: { happy: 1 }, affect: { note: "n/a" } }).affect, null);
});

test("keeps a normalised probability map and sorts it descending", () => {
  const scores = normalizeDistribution({ sad: 0.2, happy: 0.7, neutral: 0.1 });
  assert.deepEqual(
    scores.map((entry) => entry.label),
    ["happy", "sad", "neutral"],
  );
  close(scores[0].score, 0.7);
});

test("rescales a distribution whose values do not sum to one", () => {
  const scores = normalizeDistribution({ happy: 4, sad: 1 });
  close(scores[0].score, 0.8);
  close(scores[1].score, 0.2);
});

test("treats percentages above one as a 0-100 scale", () => {
  const scores = normalizeDistribution({ happy: 72.5, sad: 27.5 });
  close(scores[0].score, 0.725);
  close(scores[1].score, 0.275);
});

test("applies softmax to raw logits", () => {
  const scores = normalizeDistribution({ happy: -1.4, sad: 2.2, fear: 0.1 });
  const total = scores.reduce((sum, entry) => sum + entry.score, 0);
  close(total, 1);
  assert.equal(scores[0].label, "sad");
  assert.ok(scores.every((entry) => entry.score >= 0 && entry.score <= 1));
});

test("zips a bare score array with a label array", () => {
  const scores = normalizeDistribution([0.3, 0.5, 0.2], ["angry", "happy", "sad"]);
  assert.deepEqual(
    scores.map((entry) => entry.label),
    ["happy", "angry", "sad"],
  );
});

test("reads an array of label and probability objects", () => {
  const scores = normalizeDistribution([
    { label: "Angry", probability: 0.4 },
    { label: "Happy", probability: 0.6 },
  ]);
  assert.equal(scores[0].label, "Happy");
  close(scores[0].score, 0.6);
});

test("sums duplicate labels instead of dropping them", () => {
  const scores = normalizeDistribution({ happy: 0.3, Happy: 0.3, sad: 0.4 });
  assert.equal(scores.length, 2);
  close(scores.find((entry) => entry.label === "happy").score, 0.6);
});

test("groups alias labels under the canonical emotion", () => {
  const scores = normalizeDistribution({ happiness: 0.6, joy: 0.2, sadness: 0.2 });
  assert.equal(scores.length, 2);
  close(scores[0].score, 0.8);
  assert.ok(["happiness", "joy"].includes(scores[0].label));
});

test("returns an empty distribution for unusable input", () => {
  assert.deepEqual(normalizeDistribution(undefined), []);
  assert.deepEqual(normalizeDistribution("nope"), []);
  assert.deepEqual(normalizeDistribution([{ label: "happy" }]), []);
});

test("unwraps a single wrapper object", () => {
  const prediction = parsePrediction({ status: "ok", data: { scores: { fear: 1 } } });
  assert.equal(prediction.scores[0].label, "fear");
});

test("prefers a face distribution when the payload only reports faces", () => {
  const prediction = parsePrediction({
    faces: [
      { box: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 }, scores: { sad: 0.8, neutral: 0.2 } },
    ],
  });

  assert.equal(prediction.faces.length, 1);
  assert.equal(prediction.top.label, "sad");
  assert.deepEqual(prediction.faces[0].box, { x: 0.1, y: 0.2, w: 0.3, h: 0.4, unit: "fraction" });
});

test("accepts a top-1 only payload and flags it as partial", () => {
  const prediction = parsePrediction({ emotion: "surprise", confidence: 88 });
  assert.equal(prediction.top.label, "surprise");
  close(prediction.top.score, 0.88);
  assert.equal(prediction.distributionReported, false);
});

test("resolves a reported label that is missing from the distribution", () => {
  const prediction = parsePrediction({ label: "happy", scores: { sad: 0.7, neutral: 0.3 } });
  assert.equal(prediction.top.label, "sad");
  assert.equal(prediction.top.label, prediction.scores[0].label);
});

test("throws a ModelOutputError when there is nothing to show", () => {
  assert.throws(() => parsePrediction({ status: "ok" }), ModelOutputError);
  assert.throws(() => parsePrediction(null), ModelOutputError);
  assert.throws(() => parsePrediction({ scores: {} }), ModelOutputError);
});

test("normalises the box shapes a model may return", () => {
  assert.deepEqual(normalizeBox([10, 20, 30, 40]), { x: 10, y: 20, w: 30, h: 40, unit: "pixel" });
  assert.deepEqual(normalizeBox({ xmin: 0.1, ymin: 0.2, xmax: 0.5, ymax: 0.7 }), {
    x: 0.1,
    y: 0.2,
    w: 0.4,
    h: 0.5,
    unit: "fraction",
  });
  assert.equal(normalizeBox({ x: 0.1, y: 0.2 }), null);
  assert.equal(normalizeBox({ x: 0.4, y: 0.4, width: 0, height: 0.2 }), null);
  assert.equal(normalizeBox(null), null);
});
