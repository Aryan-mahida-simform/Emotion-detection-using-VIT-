import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import { checkHealth, InferenceError, resolveHealthUrl, resolvePredictUrl, runInference } from "../js/api.js";
import { ModelOutputError } from "../js/parse.js";

function startServer(handler) {
  const server = createServer((request, response) => {
    void handler(request, response);
  });
  return new Promise((done) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      done({
        url: `http://127.0.0.1:${port}`,
        close: () =>
          new Promise((closed) => {
            server.closeAllConnections?.();
            server.close(() => closed());
          }),
      });
    });
  });
}

test("resolvePredictUrl keeps a full inference url and appends the default path", () => {
  assert.equal(resolvePredictUrl(""), "/predict");
  assert.equal(resolvePredictUrl("/predict"), "/predict");
  assert.equal(resolvePredictUrl("http://localhost:9000/"), "http://localhost:9000/predict");
  assert.equal(resolvePredictUrl("http://localhost:9000/api/v1"), "http://localhost:9000/api/v1/predict");
  assert.equal(resolvePredictUrl("http://localhost:9000/infer"), "http://localhost:9000/infer");
  assert.equal(resolvePredictUrl("http://localhost:9000/classify/"), "http://localhost:9000/classify/");
});

test("resolveHealthUrl derives /health from the predict url when unset", () => {
  assert.equal(resolveHealthUrl("", "http://localhost:9000/predict"), "http://localhost:9000/health");
  assert.equal(resolveHealthUrl("/healthz", "http://localhost:9000/predict"), "/healthz");
});

test("posts the image in the field the backend declares and parses the model output", async (context) => {
  let receivedBody = "";
  const server = await startServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      receivedBody = Buffer.concat(chunks).toString("latin1");
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          label: "happy",
          confidence: 0.8,
          scores: [{ label: "happy", probability: 0.8 }],
          probabilities: { happy: 0.8, sad: 0.2 },
          affect: { valence: 0.5, arousal: 0.4 },
          model: { backend: "vit", model_id: "vit-emotion-test", device: "cpu", neural: true, labels: [] },
          latency_ms: 12.5,
        }),
      );
    });
  });
  context.after(server.close);

  const result = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob([new Uint8Array([1, 2, 3, 4])], { type: "image/png" }),
    filename: "face.png",
    timeoutMs: 2000,
  });

  assert.equal(result.status, 200);
  assert.equal(result.prediction.top.label, "happy");
  assert.equal(result.prediction.model.name, "vit-emotion-test");
  assert.match(receivedBody, /name="file"/);
  assert.match(receivedBody, /filename="face\.png"/);
});

test("uses the configured upload field name when one is given", async (context) => {
  let receivedBody = "";
  const server = await startServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      receivedBody = Buffer.concat(chunks).toString("latin1");
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ scores: { happy: 1 } }));
    });
  });
  context.after(server.close);

  await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    fieldName: "image",
    timeoutMs: 2000,
  });

  assert.match(receivedBody, /name="image"/);
  assert.doesNotMatch(receivedBody, /name="file"/);
});

test("reads the backend's error envelope from a failing response", async (context) => {
  const server = await startServer((request, response) => {
    request.resume();
    response.writeHead(503, { "content-type": "application/json" });
    response.end(
      JSON.stringify({
        error: { code: "model_unavailable", message: "no model weights are loaded", details: {} },
      }),
    );
  });
  context.after(server.close);

  const error = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 2000,
  }).catch((caught) => caught);

  assert.ok(error instanceof InferenceError);
  assert.equal(error.status, 503);
  assert.equal(error.code, "model_unavailable");
  assert.equal(error.message, "no model weights are loaded");
});

test("falls back to the status line when the error body is not the envelope", async (context) => {
  const server = await startServer((request, response) => {
    request.resume();
    response.writeHead(503, { "content-type": "text/plain" });
    response.end("model warming up");
  });
  context.after(server.close);

  const error = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 2000,
  }).catch((caught) => caught);

  assert.ok(error instanceof InferenceError);
  assert.equal(error.status, 503);
  assert.equal(error.code, null);
  assert.match(error.message, /503/);
  assert.match(error.detail, /warming up/);
});

test("rejects a success body that is not json", async (context) => {
  const server = await startServer((request, response) => {
    request.resume();
    response.writeHead(200, { "content-type": "text/html" });
    response.end("<html>not json</html>");
  });
  context.after(server.close);

  const error = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 2000,
  }).catch((caught) => caught);
  assert.ok(error instanceof InferenceError);
  assert.match(error.message, /not JSON/i);
  assert.equal(error.status, 200);
});

test("surfaces a payload without scores as a model output error", async (context) => {
  const server = await startServer((request, response) => {
    request.resume();
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ status: "ok" }));
  });
  context.after(server.close);

  const error = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 2000,
  }).catch((caught) => caught);
  assert.ok(error instanceof ModelOutputError);
});

test("honours cancellation through the abort signal", async (context) => {
  const server = await startServer(() => {});
  context.after(server.close);

  const controller = new AbortController();
  const promise = runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 5000,
    signal: controller.signal,
  }).catch((caught) => caught);

  setTimeout(() => controller.abort(), 20);
  const error = await promise;
  assert.ok(error instanceof InferenceError);
  assert.equal(error.cancelled, true);
});

test("gives up when the model does not answer in time", async (context) => {
  const server = await startServer(() => {});
  context.after(server.close);

  const error = await runInference({
    endpoint: `${server.url}/predict`,
    blob: new Blob(["x"]),
    filename: "face.png",
    timeoutMs: 60,
  }).catch((caught) => caught);

  assert.ok(error instanceof InferenceError);
  assert.equal(error.cancelled, false);
  assert.match(error.message, /did not answer within/);
});

test("checkHealth reports a reachable and an unreachable service", async (context) => {
  const server = await startServer((request, response) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(
      JSON.stringify({
        status: "ok",
        ready: true,
        version: "0.1.0",
        uptime_seconds: 4.2,
        backend: "vit",
        model_id: "vit-emotion-test",
        upload_modes: ["multipart"],
      }),
    );
  });
  context.after(server.close);

  const online = await checkHealth({ healthEndpoint: `${server.url}/health`, timeoutMs: 2000 });
  assert.equal(online.ok, true);
  assert.equal(online.ready, true);
  assert.match(online.model, /vit-emotion-test/);

  const offline = await checkHealth({ healthEndpoint: "http://127.0.0.1:1/health", timeoutMs: 500 });
  assert.equal(offline.ok, false);
});

test("treats a degraded health payload as not ready", async (context) => {
  const server = await startServer((request, response) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(
      JSON.stringify({
        status: "degraded",
        ready: false,
        version: "0.1.0",
        uptime_seconds: 1,
        backend: "unavailable",
        model_id: "vit",
        upload_modes: [],
      }),
    );
  });
  context.after(server.close);

  const health = await checkHealth({ healthEndpoint: `${server.url}/health`, timeoutMs: 2000 });
  assert.equal(health.ok, false);
  assert.match(health.detail, /not finished loading/);
});

test("reads the legacy health payload's readiness flag", async (context) => {
  const server = await startServer((request, response) => {
    response.writeHead(200, { "content-type": "application/json" });
    response.end(
      JSON.stringify({ status: "ok", model_available: true, weights_loaded: true, device: "cpu" }),
    );
  });
  context.after(server.close);

  const health = await checkHealth({ healthEndpoint: `${server.url}/health`, timeoutMs: 2000 });
  assert.equal(health.ok, true);
  assert.equal(health.ready, true);
});
