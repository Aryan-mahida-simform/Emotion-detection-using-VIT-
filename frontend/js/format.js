export function formatPercent(ratio, digits = 1) {
  const value = toNumber(ratio);
  if (value === null) return "—";
  const clamped = Math.min(1, Math.max(0, value));
  return `${(clamped * 100).toFixed(digits)}%`;
}

export function formatMilliseconds(milliseconds) {
  const value = toNumber(milliseconds);
  if (value === null) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function formatBytes(bytes) {
  const value = toNumber(bytes);
  if (value === null || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function toNumber(input) {
  if (typeof input === "number") return Number.isFinite(input) ? input : null;
  if (typeof input === "string" && input.trim() !== "") {
    const parsed = Number(input);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function formatClock(input) {
  const date = toDate(input);
  if (!date) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatTimestamp(input, now = Date.now()) {
  const date = toDate(input);
  if (!date) return "—";
  const elapsed = now - date.getTime();
  if (elapsed < 45_000) return "just now";
  if (elapsed < 3_600_000) return `${Math.round(elapsed / 60_000)} min ago`;
  if (elapsed < 86_400_000) return `${Math.round(elapsed / 3_600_000)} h ago`;
  return date.toLocaleDateString([], { day: "2-digit", month: "short" });
}

export function toDate(input) {
  if (input instanceof Date) return Number.isNaN(input.getTime()) ? null : input;
  const date = new Date(input);
  return Number.isNaN(date.getTime()) ? null : date;
}
