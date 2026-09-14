#!/usr/bin/env node
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, resolve, sep } from "node:path";

const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

const EMOJI = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😄",
  neutral: "😐",
  sad: "😢",
  surprise: "😲",
};

const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

const MAX_BODY_BYTES = 20 * 1024 * 1024;

const PUBLIC_PATHS = /^\/(index\.html|styles\.css|js\/[\w.-]+\.js)$/;

const args = parseArgs(process.argv.slice(2));
const root = resolve(args.root ?? join(import.meta.dirname, ".."));
const port = Number(args.port ?? 8080);
const delayMs = Number(args.delay ?? 0);
const failStatus = args.fail ? Number(args.fail) : null;

const server = createServer((request, response) => {
  handle(request, response).catch((error) => {
    sendJson(response, 500, {
      error: { code: "internal_error", message: "mock server failure", details: { detail: error?.message ?? null } },
    });
  });
});

server.listen(port, () => {
  const address = server.address();
  const shown = typeof address === "object" && address ? address.port : port;
  process.stdout.write(`frontend + mock model service on http://localhost:${shown}\n`);
  process.stdout.write(`  GET  /health, /model/info, /api/model-info\n`);
  process.stdout.write(`  POST /predict      -> app/ contract  (label, scores[], probabilities{}, affect, model{})\n`);
  process.stdout.write(`  POST /api/predict  -> backend/ contract (emoji, image{}, elapsed_ms, face_crop)\n`);
  process.stdout.write(`  multipart field "file" on both. delay=${delayMs}ms fail=${failStatus ?? "off"}\n`);
});

async function handle(request, response) {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);

  if (request.method === "OPTIONS") {
    response.writeHead(204, corsHeaders());
    response.end();
    return;
  }

  if (url.pathname === "/health") {
    sendJson(response, 200, healthPayload());
    return;
  }

  if (url.pathname === "/model/info" || url.pathname === "/api/model-info") {
    sendJson(response, 200, modelInfoPayload());
    return;
  }

  if (url.pathname === "/predict" || url.pathname === "/api/predict") {
    if (request.method !== "POST") {
      sendJson(response, 405, errorBody("method_not_allowed", "use POST"));
      return;
    }
    await handlePredict(request, response, url.pathname === "/api/predict");
    return;
  }

  if (request.method === "GET" || request.method === "HEAD") {
    await serveStatic(url.pathname, response, request.method === "HEAD");
    return;
  }

  sendJson(response, 405, errorBody("method_not_allowed", `method ${request.method} not supported`));
}

async function handlePredict(request, response, legacyContract) {
  const body = await readBody(request);
  const image = extractImage(body);

  if (!image.length) {
    sendJson(response, 422, errorBody("invalid_request", `the request carried no image in the "file" field`));
    return;
  }

  if (failStatus) {
    sendJson(
      response,
      failStatus,
      errorBody(failStatus === 503 ? "model_unavailable" : "internal_error", "simulated model failure"),
    );
    return;
  }

  const startedAt = performance.now();
  if (delayMs > 0) await new Promise((done) => setTimeout(done, delayMs));

  const probabilities = distributionFrom(image);
  const ranked = EMOTIONS.map((label) => ({ label, probability: probabilities[label] })).sort(
    (a, b) => b.probability - a.probability,
  );
  const top = ranked[0];
  const elapsed = Number((performance.now() - startedAt).toFixed(2));

  sendJson(response, 200, legacyContract ? legacyPayload(top, ranked, probabilities, image, elapsed) : appPayload(top, ranked, probabilities, elapsed));
}

function appPayload(top, ranked, probabilities, elapsed) {
  return {
    label: top.label,
    confidence: top.probability,
    scores: ranked.map(({ label, probability }) => ({ label, probability })),
    probabilities,
    affect: affectFrom(probabilities),
    model: {
      backend: "vit",
      model_id: "vit-emotion-mock",
      device: "cpu",
      neural: true,
      labels: EMOTIONS,
    },
    latency_ms: elapsed,
  };
}

function legacyPayload(top, ranked, probabilities, image, elapsed) {
  return {
    label: top.label,
    emoji: EMOJI[top.label],
    confidence: top.probability,
    label_id: EMOTIONS.indexOf(top.label),
    probabilities,
    scores: ranked.map(({ label, probability }) => ({ label, emoji: EMOJI[label], probability })),
    image: { width: 96, height: 96, mode: "RGB" },
    model_name: "vit-emotion-mock",
    device: "cpu",
    elapsed_ms: elapsed,
    face_detected: true,
    face_crop: faceCropFrom(image),
    model_input_size: { width: 224, height: 224 },
  };
}

function healthPayload() {
  const uptime = Math.round(process.uptime());
  return {
    status: "ok",
    ready: true,
    version: "0.1.0",
    uptime_seconds: uptime,
    backend: "vit",
    model_id: "vit-emotion-mock",
    upload_modes: ["multipart", "base64", "raw"],
    model_available: true,
    weights_loaded: true,
    device: "cpu",
  };
}

function modelInfoPayload() {
  return {
    backend: "vit",
    requested_backend: "vit",
    model_id: "vit-emotion-mock",
    device: "cpu",
    neural: true,
    fallback_reason: null,
    labels: EMOTIONS,
    canonical_labels: EMOTIONS,
    label_aliases: { happiness: "happy", anger: "angry" },
    image_size: 224,
    logit_temperature: 1.0,
    default_top_k: 7,
    max_batch_size: 8,
    max_image_bytes: 10 * 1024 * 1024,
    allowed_media_types: ["image/jpeg", "image/png", "image/webp"],
    upload_modes: ["multipart", "base64", "raw"],
    warmup_latency_ms: 12.4,
    available: true,
    model_name: "vit-emotion-mock",
    architecture: "ViT",
    patch_size: 16,
    num_labels: EMOTIONS.length,
    labels_source: "config",
    weights_loaded: true,
  };
}

function affectFrom(probabilities) {
  const profile = {
    angry: [-0.62, 0.8],
    disgust: [-0.7, 0.5],
    fear: [-0.7, 0.85],
    happy: [0.8, 0.6],
    neutral: [0.0, 0.2],
    sad: [-0.6, 0.25],
    surprise: [0.1, 0.9],
  };
  let total = 0;
  let valence = 0;
  let arousal = 0;
  for (const [label, probability] of Object.entries(probabilities)) {
    const point = profile[label];
    if (!point) continue;
    valence += point[0] * probability;
    arousal += point[1] * probability;
    total += probability;
  }
  if (total <= 0) return { valence: 0, arousal: 0 };
  return {
    valence: Number((valence / total).toFixed(4)),
    arousal: Number((arousal / total).toFixed(4)),
  };
}

async function serveStatic(pathname, response, headOnly) {
  const requested = pathname === "/" ? "/index.html" : pathname;
  const decoded = decodeURIComponent(requested);

  if (!PUBLIC_PATHS.test(decoded)) {
    sendText(response, 404, "not found");
    return;
  }

  const filePath = resolve(join(root, decoded));

  if (filePath !== root && !filePath.startsWith(`${root}${sep}`)) {
    sendText(response, 403, "forbidden");
    return;
  }

  let info;
  try {
    info = await stat(filePath);
  } catch {
    sendText(response, 404, "not found");
    return;
  }

  const target = info.isDirectory() ? join(filePath, "index.html") : filePath;
  const type = MIME_TYPES[extname(target).toLowerCase()] ?? "application/octet-stream";

  response.writeHead(200, {
    "content-type": type,
    "cache-control": "no-store",
    ...corsHeaders(),
  });

  if (headOnly) {
    response.end();
    return;
  }

  createReadStream(target)
    .on("error", () => response.destroy())
    .pipe(response);
}

function extractImage(body) {
  const marker = Buffer.from('name="file"');
  const start = body.indexOf(marker);
  if (start === -1) return Buffer.alloc(0);
  const headerEnd = body.indexOf("\r\n\r\n", start);
  if (headerEnd === -1) return Buffer.alloc(0);
  const next = body.indexOf("\r\n--", headerEnd);
  return body.subarray(headerEnd + 4, next === -1 ? body.length : next);
}

function distributionFrom(buffer) {
  const digest = createHash("sha256").update(buffer).digest();
  const weights = EMOTIONS.map((label, index) => {
    const strong = digest[index] / 255;
    const weak = digest[index + EMOTIONS.length] / 255;
    return Math.max(0.05, strong * 0.8 + weak * 0.2);
  });
  const total = weights.reduce((sum, value) => sum + value, 0);
  return Object.fromEntries(
    EMOTIONS.map((label, index) => [label, Number((weights[index] / total).toFixed(4))]),
  );
}

function faceCropFrom(buffer) {
  const digest = createHash("md5").update(buffer).digest();
  return {
    x: 12 + Math.round((digest[0] / 255) * 24),
    y: 8 + Math.round((digest[1] / 255) * 20),
    width: 40 + Math.round((digest[2] / 255) * 30),
    height: 40 + Math.round((digest[3] / 255) * 30),
  };
}

function readBody(request) {
  return new Promise((resolveBody, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("request body too large"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolveBody(Buffer.concat(chunks)));
    request.on("error", reject);
  });
}

function errorBody(code, message, details = {}) {
  return { error: { code, message, details } };
}

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type, accept",
  };
}

function sendJson(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    ...corsHeaders(),
  });
  response.end(body);
}

function sendText(response, status, text) {
  response.writeHead(status, { "content-type": "text/plain; charset=utf-8", ...corsHeaders() });
  response.end(text);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    const [rawKey, inlineValue] = token.slice(2).split("=");
    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (inlineValue !== undefined) {
      parsed[key] = inlineValue;
      continue;
    }
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith("--")) {
      parsed[key] = next;
      index += 1;
      continue;
    }
    parsed[key] = "true";
  }
  return parsed;
}
