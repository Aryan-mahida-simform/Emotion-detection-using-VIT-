import { canonicalRank, normalizeLabel } from "./emotions.js";

export class ModelOutputError extends Error {
  constructor(message, detail) {
    super(message);
    this.name = "ModelOutputError";
    this.detail = detail;
  }
}

const DISTRIBUTION_KEYS = [
  "scores",
  "probabilities",
  "probs",
  "emotions",
  "predictions",
  "logits",
  "classes",
  "output",
];

const WRAPPER_KEYS = ["data", "result", "prediction", "response", "payload"];

const BOX_KEYS = ["box", "bbox", "bounding_box", "bounds", "face_box", "rectangle"];

const LABEL_KEYS = ["label", "emotion", "name", "class", "class_name", "category", "expression"];

const SCORE_KEYS = ["probability", "score", "confidence", "prob", "value", "p"];

const EMOJI_KEYS = ["emoji", "icon", "glyph"];

const TOP_KEYS = [
  "label",
  "emotion",
  "predicted_emotion",
  "top_emotion",
  "predicted_label",
  "top_label",
  "class",
  "class_name",
];

const CONFIDENCE_KEYS = ["confidence", "probability", "score", "top_score"];

const LATENCY_KEYS = [
  "latency_ms",
  "latencyMs",
  "elapsed_ms",
  "elapsedMs",
  "inference_ms",
  "inferenceMs",
  "duration_ms",
  "time_ms",
];

const MODEL_NAME_KEYS = ["model_name", "modelName", "model_id", "modelId", "model", "architecture"];

const DEVICE_KEYS = ["device", "backend_device"];

const FACE_KEYS = ["faces", "detections", "instances"];

const CROP_KEYS = ["face_crop", "face_box", "crop"];

const NORMALIZE_TOLERANCE = 0.02;

export function parsePrediction(payload) {
  const body = unwrap(payload);
  if (body === null || body === undefined) {
    throw new ModelOutputError("The model service returned an empty response body.");
  }
  if (typeof body !== "object") {
    throw new ModelOutputError("The model service returned a value that is not a JSON object.");
  }

  const faces = parseFaces(body);
  const labels = pick(body, ["labels", "classes", "class_names", "classNames"]);
  const scores = normalizeDistribution(pickDistributionSource(body), Array.isArray(labels) ? labels : undefined);

  let primary = scores.length ? scores : (faces[0]?.scores ?? []);
  let distributionReported = primary.length > 0;

  if (!primary.length) {
    const label = toStringOrNull(pick(body, TOP_KEYS));
    if (label === null) {
      throw new ModelOutputError("The model response contained no emotion scores.", body);
    }
    primary = [{ label, score: toRatio(pick(body, CONFIDENCE_KEYS)) ?? 1 }];
    distributionReported = false;
  }

  const reportedTop = toStringOrNull(pick(body, TOP_KEYS));
  const top = alignTop(primary, reportedTop);

  return {
    faces,
    scores: primary,
    distributionReported,
    top,
    affect: parseAffect(body),
    model: parseModelDescriptor(body),
    count: toInteger(pick(body, ["num_labels", "label_count", "class_count"])),
    latencyMs: toFinite(pick(body, LATENCY_KEYS)),
    image: parseSize(pick(body, ["image", "source_image"]), ["width", "height", "mode"]),
    modelInput: parseSize(pick(body, ["model_input_size", "input_size"]), ["width", "height"]),
    faceDetected: toBoolean(pick(body, ["face_detected", "has_face"])) ?? (faces.length > 0 || null),
  };
}

export function normalizeDistribution(input, labels) {
  const pairs = collectPairs(input, labels);
  if (!pairs.length) return [];
  return mergeDuplicates(rescale(pairs)).sort(byScore);
}

export function normalizeBox(input) {
  if (!input) return null;
  const raw = Array.isArray(input)
    ? { x: input[0], y: input[1], w: input[2], h: input[3] }
    : {
        x: pick(input, ["x", "xmin", "left", "x_min", "x1"]),
        y: pick(input, ["y", "ymin", "top", "y_min", "y1"]),
        w: pick(input, ["w", "width"]),
        h: pick(input, ["h", "height"]),
        xMax: pick(input, ["xmax", "right", "x_max", "x2"]),
        yMax: pick(input, ["ymax", "bottom", "y_max", "y2"]),
      };

  const x = toFinite(raw.x);
  const y = toFinite(raw.y);
  const xMax = toFinite(raw.xMax);
  const yMax = toFinite(raw.yMax);
  const width = toFinite(raw.w) ?? (xMax !== null && x !== null ? xMax - x : null);
  const height = toFinite(raw.h) ?? (yMax !== null && y !== null ? yMax - y : null);

  if (x === null || y === null || width === null || height === null) return null;
  if (width <= 0 || height <= 0) return null;

  const unit = Math.max(x, y, width, height) > 1 ? "pixel" : "fraction";
  return { x: round(x), y: round(y), w: round(width), h: round(height), unit };
}

function round(value) {
  return Number(value.toFixed(6));
}

function unwrap(value) {
  let current = value;
  for (let depth = 0; depth < 4; depth += 1) {
    if (current === null || typeof current !== "object" || Array.isArray(current)) return current;
    const carriesResult =
      DISTRIBUTION_KEYS.some((key) => current[key] !== undefined) ||
      FACE_KEYS.some((key) => current[key] !== undefined) ||
      current.label !== undefined;
    if (carriesResult) return current;
    const wrappers = WRAPPER_KEYS.filter(
      (key) => current[key] !== null && typeof current[key] === "object",
    );
    if (wrappers.length !== 1) return current;
    current = current[wrappers[0]];
  }
  return current;
}

function pickDistributionSource(body) {
  for (const key of DISTRIBUTION_KEYS) {
    if (body[key] !== undefined && body[key] !== null) return body[key];
  }
  return undefined;
}

function parseFaces(body) {
  const raw = FACE_KEYS.map((key) => body[key]).find((value) => Array.isArray(value));
  const faces = raw
    ? raw
        .map((face, index) => buildFace(face, index))
        .filter((face) => face !== null && face.scores.length > 0)
    : [];

  if (faces.length) return faces;

  const crop = normalizeBox(pick(body, CROP_KEYS));
  if (!crop) return [];
  return [{ index: 0, box: crop, scores: [], top: null }];
}

function buildFace(face, index) {
  if (face === null || typeof face !== "object") return null;
  const faceLabels = Array.isArray(face.labels) ? face.labels : undefined;
  const scores = normalizeDistribution(pickDistributionSource(face) ?? face, faceLabels);
  return {
    index,
    box: normalizeBox(pick(face, BOX_KEYS)),
    scores,
    top: scores[0] ?? null,
  };
}

function alignTop(scores, reportedLabel) {
  const key = reportedLabel ? normalizeLabel(reportedLabel) : null;
  if (key) {
    const match = scores.find((entry) => normalizeLabel(entry.label) === key);
    if (match) return match;
  }
  return scores[0];
}

function parseAffect(body) {
  const affect = pick(body, ["affect", "valence_arousal"]);
  if (!affect || typeof affect !== "object") return null;
  const valence = toFinite(pick(affect, ["valence", "v"]));
  const arousal = toFinite(pick(affect, ["arousal", "a"]));
  if (valence === null && arousal === null) return null;
  return { valence, arousal };
}

function parseModelDescriptor(body) {
  const nested = pick(body, ["model", "model_info", "descriptor"]);
  const source = nested !== null && typeof nested === "object" ? nested : null;

  const name = toStringOrNull(pick(source ?? body, MODEL_NAME_KEYS));
  const backend = toStringOrNull(pick(source ?? body, ["backend", "requested_backend"]));
  const device = toStringOrNull(pick(source ?? body, DEVICE_KEYS));
  const labels = pick(source ?? body, ["labels", "canonical_labels"]);
  const neural = toBoolean(pick(source ?? body, ["neural", "is_neural"]));

  if (!name && !backend && !device && !Array.isArray(labels) && neural === null) return null;
  return {
    name,
    backend,
    device,
    neural,
    labels: Array.isArray(labels) ? labels.map((label) => String(label)) : null,
  };
}

function parseSize(input, keys) {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const entries = keys.map((key) => [key, input[key]]).filter(([, value]) => value !== undefined);
  if (!entries.length) return null;
  return Object.fromEntries(entries);
}

function collectPairs(input, labels) {
  if (input === undefined || input === null) return [];

  if (Array.isArray(input)) {
    return input.flatMap((entry, index) => {
      const fallbackLabel = Array.isArray(labels) ? labels[index] : undefined;

      if (typeof entry === "number" || typeof entry === "string") {
        const score = toFinite(entry);
        const label = fallbackLabel ?? `class_${index}`;
        return score === null ? [] : [{ label, score }];
      }

      if (Array.isArray(entry)) {
        const [first, second] = entry;
        const label = typeof first === "string" ? first : fallbackLabel;
        const score = toFinite(second);
        return label && score !== null ? [{ label, score }] : [];
      }

      if (entry && typeof entry === "object") {
        const label = pick(entry, LABEL_KEYS) ?? fallbackLabel;
        const score = entryScore(entry);
        if (label === undefined || score === null) return [];
        const emoji = toStringOrNull(pick(entry, EMOJI_KEYS));
        return [{ label: String(label), score, emoji }];
      }

      return [];
    });
  }

  if (typeof input === "object") {
    return Object.entries(input).flatMap(([label, value]) => {
      const score = value !== null && typeof value === "object" ? entryScore(value) : toFinite(value);
      return score === null ? [] : [{ label, score }];
    });
  }

  return [];
}

function entryScore(entry) {
  const direct = toFinite(pick(entry, SCORE_KEYS));
  if (direct !== null) return direct;
  const values = Object.values(entry).map(toFinite).filter((value) => value !== null);
  return values.length === 1 ? values[0] : null;
}

function rescale(pairs) {
  const values = pairs.map((pair) => pair.score);
  const smallest = Math.min(...values);
  const largest = Math.max(...values);

  if (smallest < 0 || largest > 100) {
    return softmax(values).map((score, index) => ({ ...pairs[index], score }));
  }

  const base = largest > 1 ? values.map((value) => value / 100) : values.slice();
  const total = base.reduce((sum, value) => sum + value, 0);

  if (total > 0 && Math.abs(total - 1) > NORMALIZE_TOLERANCE) {
    return base.map((value, index) => ({ ...pairs[index], score: value / total }));
  }

  return base.map((value, index) => ({ ...pairs[index], score: value }));
}

function softmax(values) {
  const largest = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - largest));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

function mergeDuplicates(pairs) {
  const merged = new Map();
  for (const pair of pairs) {
    const key = normalizeLabel(pair.label);
    const emoji = pair.emoji ?? null;
    if (merged.has(key)) {
      const existing = merged.get(key);
      existing.score += pair.score;
      existing.emoji = existing.emoji ?? emoji;
      continue;
    }
    merged.set(key, { label: pair.label, score: pair.score, emoji });
  }
  return [...merged.values()];
}

function byScore(a, b) {
  if (b.score !== a.score) return b.score - a.score;
  const rank = canonicalRank(a.label) - canonicalRank(b.label);
  if (rank !== 0) return rank;
  return String(a.label).localeCompare(String(b.label));
}

function pick(source, keys) {
  if (!source || typeof source !== "object") return undefined;
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) return source[key];
  }
  return undefined;
}

function toFinite(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function toInteger(value) {
  const number = toFinite(value);
  return number === null ? null : Math.trunc(number);
}

function toRatio(value) {
  const number = toFinite(value);
  if (number === null) return null;
  return number > 1 && number <= 100 ? number / 100 : number;
}

function toBoolean(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (value.toLowerCase() === "true") return true;
    if (value.toLowerCase() === "false") return false;
  }
  return null;
}

function toStringOrNull(value) {
  if (typeof value === "string" && value.trim() !== "") return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}
