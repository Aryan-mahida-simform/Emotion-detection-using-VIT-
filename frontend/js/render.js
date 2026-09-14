import { colorForLabel, emojiForLabel, humanizeLabel } from "./emotions.js";
import {
  formatBytes,
  formatClock,
  formatMilliseconds,
  formatPercent,
  formatTimestamp,
} from "./format.js";

export const ELEMENT_IDS = [
  "dropzone",
  "fileInput",
  "analyseButton",
  "cameraButton",
  "cameraPanel",
  "cameraVideo",
  "cameraCanvas",
  "captureButton",
  "stopCameraButton",
  "clearButton",
  "settingsPanel",
  "endpointInput",
  "fieldInput",
  "thresholdInput",
  "saveSettingsButton",
  "resetSettingsButton",
  "settingsHint",
  "serviceStatus",
  "serviceStatusText",
  "result",
  "errorTitle",
  "errorCode",
  "errorDetail",
  "cancelButton",
  "retryButton",
  "topGlyph",
  "topLabel",
  "topConfidence",
  "lowConfidenceWarning",
  "facesList",
  "bars",
  "affectPanel",
  "affectValence",
  "affectArousal",
  "affectValenceBar",
  "affectArousalBar",
  "meta",
  "copyJsonButton",
  "exportJsonButton",
  "previewImage",
  "facesOverlay",
  "previewEmpty",
  "previewCaption",
  "historyList",
  "historyEmpty",
  "clearHistoryButton",
  "toast",
];

let toastTimer = null;

export function getElements(document) {
  const ui = Object.fromEntries(ELEMENT_IDS.map((id) => [id, document.getElementById(id)]));
  const missing = ELEMENT_IDS.filter((id) => !ui[id]);
  if (missing.length) throw new Error(`Missing required elements: ${missing.join(", ")}`);
  return ui;
}

export function showState(ui, state) {
  ui.result.dataset.state = state;
  ui.result.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  for (const panel of ui.result.querySelectorAll("[data-state-panel]")) {
    panel.hidden = panel.dataset.statePanel !== state;
  }
}

export function setServiceStatus(ui, state, text) {
  ui.serviceStatus.dataset.state = state;
  ui.serviceStatusText.textContent = text;
}

export function renderError(ui, { title, code, detail }) {
  ui.errorTitle.textContent = title;
  ui.errorCode.textContent = code ? `Error code: ${code}` : "";
  ui.errorCode.hidden = !code;
  ui.errorDetail.textContent = detail ?? "";
  ui.errorDetail.hidden = !detail;
}

export function renderPrediction(ui, { prediction, url, image, threshold, selectedFace = 0, onSelectFace }) {
  const face = prediction.faces[selectedFace] ?? null;
  const scores = face?.scores.length ? face.scores : prediction.scores;
  const top = face?.scores.length ? face.top : prediction.top;

  ui.topGlyph.textContent = top.emoji || emojiForLabel(top.label);
  ui.topLabel.textContent = humanizeLabel(top.label);
  ui.topConfidence.textContent = formatPercent(top.score);
  ui.topConfidence.style.color = colorForLabel(top.label);

  const isUncertain = Number.isFinite(threshold) && top.score < threshold;
  ui.lowConfidenceWarning.hidden = !isUncertain;
  if (isUncertain) {
    ui.lowConfidenceWarning.textContent = `The leading emotion is below the ${formatPercent(
      threshold,
      0,
    )} warning threshold, so treat this prediction as uncertain.`;
  }

  renderFaces(ui, prediction.faces, selectedFace, onSelectFace);
  renderBars(ui.bars, scores, top.label);
  renderAffect(ui, prediction.affect);
  renderMeta(ui.meta, buildMeta({ prediction, url, image, scores, top }));
  renderOverlay(ui, prediction.faces, selectedFace);
}

export function renderBars(container, scores, topLabel) {
  container.replaceChildren(
    ...scores.map((entry) => {
      const item = document.createElement("div");
      item.className = "bar";
      item.setAttribute("role", "listitem");
      item.dataset.label = entry.label;
      if (entry.label === topLabel) item.dataset.top = "true";
      item.style.setProperty("--bar-color", colorForLabel(entry.label));

      const head = document.createElement("div");
      head.className = "bar-head";

      const label = document.createElement("span");
      label.className = "bar-label";
      const swatch = document.createElement("i");
      swatch.setAttribute("aria-hidden", "true");
      label.append(swatch, document.createTextNode(humanizeLabel(entry.label)));

      const score = document.createElement("span");
      score.className = "bar-score";
      score.textContent = formatPercent(entry.score);

      head.append(label, score);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.setProperty("--fill", String(entry.score));
      track.append(fill);

      item.append(head, track);
      return item;
    }),
  );
}

export function renderAffect(ui, affect) {
  const hasAffect = Boolean(affect) && (affect.valence !== null || affect.arousal !== null);
  ui.affectPanel.hidden = !hasAffect;
  if (!hasAffect) return;

  ui.affectValence.textContent = formatSigned(affect.valence);
  ui.affectArousal.textContent = formatSigned(affect.arousal);
  const valenceFill = clamp01(((affect.valence ?? 0) + 1) / 2);
  ui.affectValenceBar.style.setProperty("--fill", String(valenceFill));
  ui.affectArousalBar.style.setProperty("--fill", String(clamp01(affect.arousal ?? 0)));
}

export function renderFaces(ui, faces, selectedFace, onSelectFace) {
  const selectable = faces.filter((face) => face.scores.length > 0);
  if (selectable.length < 2) {
    ui.facesList.hidden = true;
    ui.facesList.replaceChildren();
    return;
  }

  ui.facesList.hidden = false;
  ui.facesList.replaceChildren(
    ...faces.map((face, index) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", index === selectedFace ? "true" : "false");
      button.dataset.faceIndex = String(index);
      button.textContent = face.scores.length
        ? `Face ${index + 1} · ${humanizeLabel(face.top.label)} ${formatPercent(face.top.score)}`
        : `Face ${index + 1} · no scores`;
      button.addEventListener("click", () => onSelectFace(index));
      item.append(button);
      return item;
    }),
  );
}

export function renderOverlay(ui, faces, selectedFace) {
  const overlay = ui.facesOverlay;
  overlay.replaceChildren();

  const image = ui.previewImage;
  const naturalWidth = image.naturalWidth || 1;
  const naturalHeight = image.naturalHeight || 1;

  faces.forEach((face, index) => {
    if (!face.box) return;
    const { x, y, w, h, unit } = face.box;
    const toPercent = (value, span) => (unit === "pixel" ? (value / span) * 100 : value * 100);

    const element = document.createElement("div");
    element.className = "face-box";
    element.dataset.selected = String(index === selectedFace);
    element.style.left = `${toPercent(x, naturalWidth)}%`;
    element.style.top = `${toPercent(y, naturalHeight)}%`;
    element.style.width = `${toPercent(w, naturalWidth)}%`;
    element.style.height = `${toPercent(h, naturalHeight)}%`;

    const caption = document.createElement("span");
    caption.textContent = face.scores.length ? humanizeLabel(face.top.label) : "Face";
    element.append(caption);
    overlay.append(element);
  });
}

export function renderMeta(container, entries) {
  container.replaceChildren(
    ...entries.flatMap(({ term, value }) => {
      if (value === null || value === undefined || value === "") return [];
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      return [dt, dd];
    }),
  );
}

export function renderHistory(ui, entries, onSelect) {
  const hasEntries = entries.length > 0;
  ui.historyEmpty.hidden = hasEntries;
  ui.historyList.replaceChildren(
    ...entries.map((entry) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.entryId = entry.id;
      button.title = entry.blob
        ? "Re-open this result"
        : "The image for this entry is no longer available. Select it again to analyse";

      const media = entry.thumbnail
        ? Object.assign(document.createElement("img"), { src: entry.thumbnail, alt: "" })
        : Object.assign(document.createElement("span"), {
            className: "history-glyph",
            textContent: emojiForLabel(entry.label),
          });

      const text = document.createElement("span");
      text.className = "history-text";
      const label = document.createElement("span");
      label.className = "history-label";
      label.textContent = humanizeLabel(entry.label);
      const sub = document.createElement("span");
      sub.className = "history-sub";
      sub.textContent = [entry.filename, formatPercent(entry.confidence)].filter(Boolean).join(" · ");
      text.append(label, sub);

      const when = document.createElement("span");
      when.className = "history-when";
      when.textContent = formatTimestamp(entry.timestamp);

      button.append(media, text, when);
      button.addEventListener("click", () => onSelect(entry));
      item.append(button);
      return item;
    }),
  );
}

export function showToast(ui, message) {
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    ui.toast.hidden = true;
    toastTimer = null;
  }, 2800);
}

export function renderPreview(ui, image) {
  ui.facesOverlay.replaceChildren();

  if (!image) {
    ui.previewImage.hidden = true;
    ui.previewImage.removeAttribute("src");
    ui.previewEmpty.hidden = false;
    ui.previewCaption.textContent = "";
    return;
  }

  ui.previewImage.hidden = false;
  ui.previewImage.src = image.url;
  ui.previewImage.alt = image.name ? `Selected image: ${image.name}` : "Selected image";
  ui.previewEmpty.hidden = true;
  ui.previewCaption.textContent = [image.name, formatBytes(image.size)].filter(Boolean).join(" · ");
}

function buildMeta({ prediction, url, image, scores, top }) {
  const runnerUp = scores[1] ?? null;
  const model = prediction.model ?? {};
  const labels = model.labels ?? null;

  return [
    { term: "Model", value: model.name ?? "not reported" },
    { term: "Backend", value: model.backend ?? null },
    { term: "Device", value: model.device ?? null },
    { term: "Neural", value: model.neural === null || model.neural === undefined ? null : model.neural ? "yes" : "no" },
    { term: "Inference time", value: formatMilliseconds(prediction.latencyMs) },
    { term: "Round trip", value: formatMilliseconds(prediction.roundTripMs) },
    { term: "Classes scored", value: scores.length },
    { term: "Vocabulary", value: labels ? `${labels.length} labels` : prediction.count ? `${prediction.count} labels` : null },
    { term: "Scores", value: prediction.distributionReported ? "full distribution" : "top-1 only" },
    {
      term: "Top-1 margin",
      value: runnerUp ? formatPercent(Math.max(0, top.score - runnerUp.score)) : null,
    },
    { term: "Face detected", value: prediction.faceDetected === null ? null : prediction.faceDetected ? "yes" : "no" },
    {
      term: "Source image",
      value: prediction.image
        ? [`${prediction.image.width}×${prediction.image.height}`, prediction.image.mode].filter(Boolean).join(" px, ")
        : null,
    },
    {
      term: "Model input",
      value: prediction.modelInput ? `${prediction.modelInput.width}×${prediction.modelInput.height}` : null,
    },
    { term: "Endpoint", value: url ?? null },
    { term: "Uploaded", value: image?.name ?? null },
    { term: "Analysed at", value: formatClock(new Date()) },
  ];
}

function formatSigned(value) {
  if (!Number.isFinite(value)) return "—";
  const rounded = value.toFixed(2);
  return value > 0 ? `+${rounded}` : rounded;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}
