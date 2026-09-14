import { parsePrediction } from "./parse.js";

export class InferenceError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "InferenceError";
    this.detail = options.detail ?? null;
    this.status = options.status ?? null;
    this.code = options.code ?? null;
    this.cancelled = options.cancelled === true;
  }
}

const INFERENCE_PATH = /\/(predict|infer|inference|analyse|analyze|classify|detect)\/?$/i;

export const DEFAULT_UPLOAD_FIELD = "file";

export function resolvePredictUrl(endpoint) {
  const trimmed = String(endpoint ?? "").trim();
  if (!trimmed) return "/predict";
  if (INFERENCE_PATH.test(trimmed)) return trimmed;
  return `${trimmed.replace(/\/+$/, "")}/predict`;
}

export function resolveHealthUrl(healthEndpoint, predictUrl = resolvePredictUrl()) {
  const trimmed = String(healthEndpoint ?? "").trim();
  if (trimmed) return trimmed;
  const base = String(predictUrl).replace(INFERENCE_PATH, "");
  return `${base.replace(/\/+$/, "")}/health`;
}

export function readErrorBody(payload) {
  if (!payload || typeof payload !== "object") return null;

  const nested = payload.error;
  const source = nested && typeof nested === "object" ? nested : payload;
  const code = typeof source.code === "string" && source.code ? source.code : null;
  const message = typeof source.message === "string" && source.message ? source.message : null;
  if (!code && !message) return null;

  const details = source.details && typeof source.details === "object" ? source.details : null;
  return { code, message, details };
}

export async function checkHealth({ healthEndpoint, predictUrl, timeoutMs = 4000 } = {}) {
  const url = resolveHealthUrl(healthEndpoint, predictUrl);
  const guard = linkAbortSignal(undefined, timeoutMs);
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { accept: "application/json" },
      signal: guard.signal,
    });
    if (!response.ok) return { ok: false, url, detail: `HTTP ${response.status} ${response.statusText}`.trim() };

    const payload = await parseJsonOrNull(response);
    const ready = readReadiness(payload);
    if (ready === false) {
      return { ok: false, url, detail: "the backend has not finished loading", model: describeModel(payload) };
    }
    return { ok: true, url, model: describeModel(payload), ready };
  } catch (error) {
    return { ok: false, url, detail: error?.message ?? "unreachable" };
  } finally {
    guard.dispose();
  }
}

export async function runInference({ endpoint, blob, filename, fieldName, timeoutMs, signal }) {
  const url = resolvePredictUrl(endpoint);
  const form = new FormData();
  form.append(fieldName || DEFAULT_UPLOAD_FIELD, blob, filename || "capture.png");

  const guard = linkAbortSignal(signal, timeoutMs);
  const startedAt = performance.now();

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      body: form,
      headers: { accept: "application/json" },
      signal: guard.signal,
    });
  } catch (error) {
    const message = guard.timedOut
      ? `The model service at ${url} did not answer within ${Math.round(guard.timeoutMs / 1000)} s.`
      : signal?.aborted
        ? "Analysis cancelled."
        : `Could not reach the model service at ${url}.`;
    throw new InferenceError(message, {
      cancelled: !guard.timedOut && signal?.aborted === true,
      detail: error?.message ?? null,
    });
  } finally {
    guard.dispose();
  }

  const text = await response.text();
  const roundTripMs = performance.now() - startedAt;

  let payload = null;
  let malformed = false;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    malformed = true;
  }

  if (!response.ok) {
    const body = malformed ? null : readErrorBody(payload);
    throw new InferenceError(
      body?.message ?? `Model service returned HTTP ${response.status} ${response.statusText}`.trim(),
      {
        detail: body ? formatDetails(body.details) ?? truncate(text) : truncate(text),
        status: response.status,
        code: body?.code ?? null,
      },
    );
  }

  if (malformed) {
    throw new InferenceError("The model service replied with a body that is not JSON.", {
      detail: truncate(text),
      status: response.status,
    });
  }

  const prediction = parsePrediction(payload);
  prediction.roundTripMs = roundTripMs;

  return { url, status: response.status, raw: payload, prediction };
}

function readReadiness(payload) {
  if (!payload || typeof payload !== "object") return null;
  if (typeof payload.ready === "boolean") return payload.ready;
  if (typeof payload.model_available === "boolean") return payload.model_available;
  return null;
}

function describeModel(payload) {
  if (!payload || typeof payload !== "object") return null;
  const parts = [
    stringOrNull(payload.model_id),
    stringOrNull(payload.model_name),
    stringOrNull(payload.backend),
    stringOrNull(payload.device),
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

function formatDetails(details) {
  if (!details || typeof details !== "object") return null;
  const entries = Object.entries(details);
  if (!entries.length) return null;
  try {
    return truncate(JSON.stringify(details, null, 2));
  } catch {
    return null;
  }
}

function stringOrNull(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function linkAbortSignal(externalSignal, timeoutMs) {
  const controller = new AbortController();
  const guard = { signal: controller.signal, timeoutMs, timedOut: false };

  const forward = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) forward();
    else externalSignal.addEventListener("abort", forward, { once: true });
  }

  const timer =
    Number.isFinite(timeoutMs) && timeoutMs > 0
      ? setTimeout(() => {
          guard.timedOut = true;
          controller.abort();
        }, timeoutMs)
      : null;

  guard.dispose = () => {
    if (timer !== null) clearTimeout(timer);
    externalSignal?.removeEventListener("abort", forward);
  };

  return guard;
}

async function parseJsonOrNull(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function truncate(text, limit = 400) {
  const value = String(text ?? "").trim();
  return value.length > limit ? `${value.slice(0, limit)}…` : value || null;
}
