const STORAGE_KEY = "vit-emotion-frontend.history.v1";

export function makeEntry({ prediction, image, url, blob }) {
  const top = prediction?.top ?? prediction?.scores?.[0] ?? null;
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    label: top?.label ?? "unknown",
    confidence: Number.isFinite(top?.score) ? top.score : null,
    model: prediction?.model ?? null,
    latencyMs: Number.isFinite(prediction?.latencyMs) ? prediction.latencyMs : null,
    endpoint: url ?? null,
    filename: image?.name ?? null,
    timestamp: new Date().toISOString(),
    thumbnail: null,
    blob: blob ?? null,
  };
}

export function addEntry(entries, entry, limit) {
  const next = [entry, ...entries.filter((existing) => existing.id !== entry.id)];
  return next.slice(0, Math.max(1, limit));
}

export function loadHistory(limit) {
  let stored;
  try {
    stored = JSON.parse(globalThis.localStorage?.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    stored = [];
  }
  if (!Array.isArray(stored)) return [];
  return stored.filter(isValidEntry).slice(0, Math.max(1, limit)).map((entry) => ({ ...entry, blob: null }));
}

export function saveHistory(entries, limit) {
  const trimmed = entries.slice(0, Math.max(1, limit)).map(({ blob, ...rest }) => rest);
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    return;
  } catch {
    const withoutThumbnails = trimmed.map(({ thumbnail, ...rest }) => ({ ...rest, thumbnail: null }));
    try {
      globalThis.localStorage?.setItem(STORAGE_KEY, JSON.stringify(withoutThumbnails));
    } catch {
      return;
    }
  }
}

export function clearHistory() {
  try {
    globalThis.localStorage?.removeItem(STORAGE_KEY);
  } catch {
    return;
  }
}

function isValidEntry(entry) {
  return (
    entry !== null &&
    typeof entry === "object" &&
    typeof entry.id === "string" &&
    typeof entry.label === "string" &&
    typeof entry.timestamp === "string"
  );
}
