import { DEFAULT_UPLOAD_FIELD } from "./api.js";

const STORAGE_KEY = "vit-emotion-frontend.settings.v1";

export const DEFAULT_SETTINGS = Object.freeze({
  endpoint: "/predict",
  healthEndpoint: "/health",
  uploadField: DEFAULT_UPLOAD_FIELD,
  confidenceWarningThreshold: 0.4,
  requestTimeoutMs: 30_000,
  historyLimit: 12,
});

export function normalizeSettings(candidate = {}) {
  const merged = { ...DEFAULT_SETTINGS, ...(candidate ?? {}) };
  const endpoint =
    typeof merged.endpoint === "string" && merged.endpoint.trim()
      ? merged.endpoint.trim()
      : DEFAULT_SETTINGS.endpoint;
  const healthEndpoint =
    typeof merged.healthEndpoint === "string" ? merged.healthEndpoint.trim() : DEFAULT_SETTINGS.healthEndpoint;
  const uploadField =
    typeof merged.uploadField === "string" && merged.uploadField.trim()
      ? merged.uploadField.trim()
      : DEFAULT_SETTINGS.uploadField;
  const threshold = Number(merged.confidenceWarningThreshold);
  const timeout = Number(merged.requestTimeoutMs);
  const historyLimit = Number(merged.historyLimit);

  return {
    endpoint,
    healthEndpoint,
    uploadField,
    confidenceWarningThreshold: Number.isFinite(threshold)
      ? Math.min(1, Math.max(0, threshold))
      : DEFAULT_SETTINGS.confidenceWarningThreshold,
    requestTimeoutMs: Number.isFinite(timeout) && timeout >= 1000 ? timeout : DEFAULT_SETTINGS.requestTimeoutMs,
    historyLimit:
      Number.isFinite(historyLimit) && historyLimit > 0
        ? Math.floor(historyLimit)
        : DEFAULT_SETTINGS.historyLimit,
  };
}

export function loadSettings() {
  try {
    const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
    return normalizeSettings(stored ? JSON.parse(stored) : {});
  } catch {
    return normalizeSettings({});
  }
}

export function saveSettings(patch = {}) {
  const settings = normalizeSettings({ ...loadSettings(), ...patch });
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    return settings;
  }
  return settings;
}

export function clearSettings() {
  try {
    globalThis.localStorage?.removeItem(STORAGE_KEY);
  } catch {
    return DEFAULT_SETTINGS;
  }
  return DEFAULT_SETTINGS;
}

export function readQuerySettings(search = globalThis.location?.search ?? "") {
  let params;
  try {
    params = new URLSearchParams(search);
  } catch {
    return {};
  }

  const patch = {};
  const endpoint = params.get("endpoint") ?? params.get("api");
  if (endpoint) patch.endpoint = endpoint;

  const health = params.get("health");
  if (health) patch.healthEndpoint = health;

  const field = params.get("field");
  if (field) patch.uploadField = field;

  const threshold = params.get("threshold");
  if (threshold !== null) {
    const value = Number(threshold);
    if (Number.isFinite(value)) {
      patch.confidenceWarningThreshold = value > 1 ? value / 100 : value;
    }
  }

  const timeout = params.get("timeout");
  if (timeout !== null) {
    const value = Number(timeout);
    if (Number.isFinite(value) && value > 0) patch.requestTimeoutMs = value;
  }

  return patch;
}
