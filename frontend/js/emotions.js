export const CANONICAL_EMOTIONS = Object.freeze([
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
]);

export const EMOTION_META = Object.freeze({
  angry: { label: "Angry", color: "#e5484d", emoji: "😠" },
  disgust: { label: "Disgust", color: "#b08817", emoji: "🤢" },
  fear: { label: "Fear", color: "#7c5cff", emoji: "😨" },
  happy: { label: "Happy", color: "#ffc53d", emoji: "😄" },
  neutral: { label: "Neutral", color: "#8b8f98", emoji: "😐" },
  sad: { label: "Sad", color: "#3b82f6", emoji: "😢" },
  surprise: { label: "Surprise", color: "#12b8a0", emoji: "😲" },
});

const LABEL_ALIASES = Object.freeze({
  anger: "angry",
  rage: "angry",
  disgusted: "disgust",
  contempt: "disgust",
  fearful: "fear",
  afraid: "fear",
  scared: "fear",
  happiness: "happy",
  joy: "happy",
  joyful: "happy",
  calm: "neutral",
  sadness: "sad",
  sorrow: "sad",
  surprised: "surprise",
});

export const CANONICAL_EMOJI = Object.freeze(
  Object.fromEntries(Object.entries(EMOTION_META).map(([key, meta]) => [key, meta.emoji])),
);

export function normalizeLabel(raw) {
  const slug = String(raw ?? "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  return LABEL_ALIASES[slug] ?? slug;
}

export function humanizeLabel(raw) {
  const key = normalizeLabel(raw);
  if (EMOTION_META[key]) return EMOTION_META[key].label;
  const words = key.split("_").filter(Boolean);
  if (!words.length) return "Unknown";
  return words.map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");
}

export function colorForLabel(raw) {
  const key = normalizeLabel(raw);
  if (EMOTION_META[key]) return EMOTION_META[key].color;
  let hash = 7;
  for (const character of key) hash = (hash * 31 + character.codePointAt(0)) % 360;
  return `hsl(${hash} 68% 58%)`;
}

export function emojiForLabel(raw, fallback = "🙂") {
  return EMOTION_META[normalizeLabel(raw)]?.emoji ?? fallback;
}

export function canonicalRank(raw) {
  const index = CANONICAL_EMOTIONS.indexOf(normalizeLabel(raw));
  return index === -1 ? CANONICAL_EMOTIONS.length : index;
}
