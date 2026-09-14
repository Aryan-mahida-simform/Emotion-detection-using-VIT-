import { checkHealth, InferenceError, resolvePredictUrl, runInference } from "./api.js";
import {
  clearSettings,
  loadSettings,
  normalizeSettings,
  readQuerySettings,
  saveSettings,
} from "./config.js";
import { addEntry, clearHistory, loadHistory, makeEntry, saveHistory } from "./history.js";
import { ModelOutputError } from "./parse.js";
import { formatBytes } from "./format.js";
import * as view from "./render.js";
import { createThumbnail } from "./thumbnail.js";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const HEALTH_TIMEOUT_MS = 4000;

const ui = view.getElements(document);
window.__emotionAppBooted = true;

const app = {
  settings: normalizeSettings({ ...loadSettings(), ...readQuerySettings() }),
  image: null,
  result: null,
  controller: null,
  stream: null,
  selectedFace: 0,
  history: [],
};

app.history = loadHistory(app.settings.historyLimit);

function init() {
  ui.endpointInput.value = app.settings.endpoint;
  ui.fieldInput.value = app.settings.uploadField;
  ui.thresholdInput.value = String(Math.round(app.settings.confidenceWarningThreshold * 100));
  updateSettingsHint();
  bindEvents();
  view.renderPreview(ui, null);
  view.renderHistory(ui, app.history, restoreEntry);
  view.showState(ui, "idle");
  void checkService();
}

function updateSettingsHint() {
  ui.settingsHint.textContent = `Timeout ${Math.round(app.settings.requestTimeoutMs / 1000)} s · multipart field "${
    app.settings.uploadField
  }" · health check ${app.settings.healthEndpoint || "derived"} · settings are stored in this browser.`;
}

function bindEvents() {
  ui.dropzone.addEventListener("click", () => ui.fileInput.click());
  ui.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      ui.fileInput.click();
    }
  });
  ui.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    ui.dropzone.dataset.active = "true";
  });
  ui.dropzone.addEventListener("dragleave", () => {
    delete ui.dropzone.dataset.active;
  });
  ui.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    delete ui.dropzone.dataset.active;
    const file = event.dataTransfer?.files?.[0];
    if (file) void acceptFile(file);
  });

  ui.fileInput.addEventListener("change", () => {
    const file = ui.fileInput.files?.[0];
    if (file) void acceptFile(file);
    ui.fileInput.value = "";
  });

  document.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData?.items ?? [])].find((entry) => entry.type.startsWith("image/"));
    const file = item?.getAsFile();
    if (file) void acceptFile(file);
  });

  ui.analyseButton.addEventListener("click", () => void analyse());
  ui.retryButton.addEventListener("click", () => void analyse());
  ui.cancelButton.addEventListener("click", () => {
    app.controller?.abort();
    view.showToast(ui, "Cancelling…");
  });
  ui.clearButton.addEventListener("click", resetWorkspace);

  ui.cameraButton.addEventListener("click", () => void startCamera());
  ui.captureButton.addEventListener("click", () => void captureFrame());
  ui.stopCameraButton.addEventListener("click", stopCamera);

  ui.saveSettingsButton.addEventListener("click", () => {
    app.settings = saveSettings({
      endpoint: ui.endpointInput.value,
      uploadField: ui.fieldInput.value,
      confidenceWarningThreshold: Number(ui.thresholdInput.value) / 100,
    });
    ui.endpointInput.value = app.settings.endpoint;
    ui.fieldInput.value = app.settings.uploadField;
    ui.thresholdInput.value = String(Math.round(app.settings.confidenceWarningThreshold * 100));
    if (app.result) renderResult();
    updateSettingsHint();
    view.showToast(ui, "Settings saved.");
    void checkService();
  });

  ui.resetSettingsButton.addEventListener("click", () => {
    app.settings = normalizeSettings(clearSettings());
    ui.endpointInput.value = app.settings.endpoint;
    ui.fieldInput.value = app.settings.uploadField;
    ui.thresholdInput.value = String(Math.round(app.settings.confidenceWarningThreshold * 100));
    updateSettingsHint();
    view.showToast(ui, "Settings reset.");
    void checkService();
  });

  ui.copyJsonButton.addEventListener("click", () => void copyJson());
  ui.exportJsonButton.addEventListener("click", exportJson);
  ui.clearHistoryButton.addEventListener("click", () => {
    clearHistory();
    app.history = [];
    view.renderHistory(ui, app.history, restoreEntry);
    view.showToast(ui, "History cleared.");
  });

  window.addEventListener("beforeunload", () => {
    if (app.image?.url) URL.revokeObjectURL(app.image.url);
    stopCamera();
  });
}

async function acceptFile(file) {
  if (!file.type.startsWith("image/")) {
    view.showToast(ui, "That file is not an image.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    view.showToast(ui, `Images must be ${formatBytes(MAX_UPLOAD_BYTES)} or smaller.`);
    return;
  }

  const thumbnail = await createThumbnail(file);
  setImage({ blob: file, name: file.name, size: file.size, thumbnail });
}

async function captureFrame() {
  const video = ui.cameraVideo;
  if (!video.videoWidth) {
    view.showToast(ui, "The camera is not streaming yet.");
    return;
  }

  const canvas = ui.cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) {
    view.showToast(ui, "Could not capture that frame.");
    return;
  }

  setImage({
    blob,
    name: `camera-${new Date().toISOString().replace(/[:.]/g, "-")}.png`,
    size: blob.size,
    thumbnail: await createThumbnail(blob),
  });
  stopCamera();
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    view.showToast(ui, "This browser cannot access a camera.");
    return;
  }

  try {
    app.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    ui.cameraVideo.srcObject = app.stream;
    await ui.cameraVideo.play();
    ui.cameraPanel.hidden = false;
    ui.cameraButton.disabled = true;
  } catch (error) {
    view.showToast(ui, `Camera unavailable: ${error?.message ?? "permission denied"}.`);
  }
}

function stopCamera() {
  for (const track of app.stream?.getTracks() ?? []) track.stop();
  app.stream = null;
  ui.cameraVideo.srcObject = null;
  ui.cameraPanel.hidden = true;
  ui.cameraButton.disabled = false;
}

function setImage({ blob, name, size, thumbnail }) {
  if (app.image?.url) URL.revokeObjectURL(app.image.url);
  app.image = { blob, name, size, thumbnail, url: URL.createObjectURL(blob) };
  app.result = null;
  app.selectedFace = 0;

  view.renderPreview(ui, app.image);
  ui.analyseButton.disabled = false;
  view.showState(ui, "idle");
}

function resetWorkspace() {
  app.controller?.abort();
  stopCamera();
  if (app.image?.url) URL.revokeObjectURL(app.image.url);
  app.image = null;
  app.result = null;
  app.selectedFace = 0;
  ui.analyseButton.disabled = true;
  view.renderPreview(ui, null);
  view.showState(ui, "idle");
}

async function analyse() {
  if (!app.image) {
    view.showToast(ui, "Choose an image first.");
    return;
  }

  const controller = new AbortController();
  app.controller = controller;
  view.showState(ui, "loading");

  try {
    const { prediction, raw, url } = await runInference({
      endpoint: app.settings.endpoint,
      blob: app.image.blob,
      filename: app.image.name,
      fieldName: app.settings.uploadField,
      timeoutMs: app.settings.requestTimeoutMs,
      signal: controller.signal,
    });

    app.result = { prediction, raw, url };
    app.selectedFace = 0;
    renderResult();

    const entry = makeEntry({ prediction, image: app.image, url, blob: app.image.blob });
    entry.thumbnail = app.image.thumbnail ?? null;
    app.history = addEntry(app.history, entry, app.settings.historyLimit);
    saveHistory(app.history, app.settings.historyLimit);
    view.renderHistory(ui, app.history, restoreEntry);
    view.showState(ui, "success");
  } catch (error) {
    if (error instanceof InferenceError && error.cancelled) {
      view.showState(ui, app.result ? "success" : "idle");
      return;
    }
    view.renderError(ui, describeError(error));
    view.showState(ui, "error");
  } finally {
    app.controller = null;
  }
}

function describeError(error) {
  if (error instanceof InferenceError) return { title: error.message, detail: error.detail };
  if (error instanceof ModelOutputError) {
    return { title: error.message, detail: error.detail ? truncateJson(error.detail) : null };
  }
  return { title: "Unexpected failure while reading the model output.", detail: error?.message ?? null };
}

function truncateJson(value) {
  try {
    const text = JSON.stringify(value, null, 2);
    return text.length > 1200 ? `${text.slice(0, 1200)}…` : text;
  } catch {
    return null;
  }
}

function renderResult() {
  view.renderPrediction(ui, {
    prediction: app.result.prediction,
    url: app.result.url,
    image: app.image,
    threshold: app.settings.confidenceWarningThreshold,
    selectedFace: app.selectedFace,
    onSelectFace: selectFace,
  });
}

function selectFace(index) {
  app.selectedFace = index;
  renderResult();
}

function restoreEntry(entry) {
  if (!entry.blob) {
    view.showToast(ui, "That image is no longer in memory — select it again to re-run the model.");
    return;
  }

  app.image = { blob: entry.blob, name: entry.filename, size: entry.blob.size, thumbnail: entry.thumbnail, url: URL.createObjectURL(entry.blob) };
  view.renderPreview(ui, app.image);
  ui.analyseButton.disabled = false;
  view.showToast(ui, "Image restored — press Analyse to run it again.");
}

async function copyJson() {
  if (!app.result) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(exportPayload(), null, 2));
    view.showToast(ui, "Model output copied as JSON.");
  } catch {
    view.showToast(ui, "Clipboard blocked by the browser.");
  }
}

function exportJson() {
  if (!app.result) return;
  const blob = new Blob([JSON.stringify(exportPayload(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `emotion-output-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function exportPayload() {
  const { prediction, raw, url } = app.result;
  return {
    endpoint: url,
    analysedAt: new Date().toISOString(),
    image: app.image ? { name: app.image.name, bytes: app.image.size } : null,
    request: {
      threshold: app.settings.confidenceWarningThreshold,
      uploadField: app.settings.uploadField,
    },
    prediction: {
      top: prediction.top,
      scores: prediction.scores,
      distributionReported: prediction.distributionReported,
      affect: prediction.affect,
      faces: prediction.faces.map(({ index, box, top }) => ({ index, box, top })),
      faceDetected: prediction.faceDetected,
      model: prediction.model,
      latencyMs: prediction.latencyMs,
      roundTripMs: prediction.roundTripMs,
      image: prediction.image,
      modelInput: prediction.modelInput,
    },
    modelResponse: raw,
  };
}

async function checkService() {
  view.setServiceStatus(ui, "checking", "Checking the model service…");
  const health = await checkHealth({
    healthEndpoint: app.settings.healthEndpoint,
    predictUrl: resolvePredictUrl(app.settings.endpoint),
    timeoutMs: HEALTH_TIMEOUT_MS,
  });

  if (health.ok) {
    view.setServiceStatus(ui, "online", `${health.model ?? "Model service"} ready`);
    return;
  }
  view.setServiceStatus(ui, "offline", "Model service not reachable. Check the settings panel");
}

init();
eck the settings panel");
}

init();
