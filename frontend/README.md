# Emotion detection with ViT: frontend

Browser frontend that shows the output of a Vision Transformer (ViT) emotion classifier. You pick a
face image, the page posts it to the model service, and the response is drawn as a ranked list of
emotion probabilities with the winning emotion called out.

Plain HTML, CSS, and ES modules. No build step is required to run it and there are no runtime
dependencies.

## Run it

```bash
npm start
```

That starts a mock model service and serves this folder on <http://localhost:8080>, so `/predict`
resolves with no configuration.

To serve the frontend on its own:

```bash
npm run static
```

Any static file server works. Opening `index.html` from the filesystem does not, because the page
loads ES modules and browsers block those over the `file:` scheme. The page detects that case and
shows a banner telling you to serve the folder over HTTP.

## The model service contract

The two services this frontend targets live in the `app/` and `backend/` trees. Both accept a
`POST` with a `multipart/form-data` body where the image part is named `file`. That name is a
setting, because a service wired up with a different field name would otherwise fail with an opaque
422.

```bash
curl -F "file=@face.png" http://localhost:8000/predict
```

### The `app/` shape

`app/schemas.py` returns a top label, a ranked score list, a probability map, an affect pair, and a
nested model descriptor:

```json
{
  "label": "happy",
  "confidence": 0.93,
  "scores": [
    { "label": "happy", "probability": 0.93 },
    { "label": "neutral", "probability": 0.06 }
  ],
  "probabilities": { "angry": 0.01, "happy": 0.93, "neutral": 0.06 },
  "affect": { "valence": 0.71, "arousal": 0.58 },
  "model": {
    "backend": "vit",
    "model_id": "vit-emotion-base",
    "device": "cuda",
    "neural": true,
    "labels": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
  },
  "latency_ms": 41.7
}
```

The affect pair comes from the circumplex coordinates in `app/emotions.py`, so the page draws
valence on a -1 to +1 track and arousal on a 0 to 1 track.

### The `backend/` shape

`backend/app/schemas/prediction.py` returns a label and confidence, plus an emoji per score, a source
image descriptor, a face crop in pixel coordinates, and the model input size:

```json
{
  "label": "sad",
  "emoji": "😢",
  "confidence": 0.64,
  "label_id": 5,
  "probabilities": { "sad": 0.64, "neutral": 0.3 },
  "scores": [{ "label": "sad", "emoji": "😢", "probability": 0.64 }],
  "image": { "width": 48, "height": 48, "mode": "RGB" },
  "model_name": "vit-emotion-base",
  "device": "cpu",
  "elapsed_ms": 88.25,
  "face_detected": true,
  "face_crop": { "x": 12, "y": 10, "width": 24, "height": 26 },
  "model_input_size": { "width": 224, "height": 224 }
}
```

The `face_crop` rectangle is drawn over the preview. Box values above 1 are read as pixels and
values at or below 1 as fractions, so both conventions render correctly.

### Errors

Both services return the same envelope, which the page reads for the error code and message:

```json
{ "error": { "code": "model_unavailable", "message": "no model weights are loaded", "details": {} } }
```

The page separates four failure modes and shows the raw body for the last three:

- The service is unreachable.
- The request timed out. The default is 30 seconds.
- The service answered with a non-2xx status. The envelope's code and message are shown.
- The body was not JSON, or held no emotion scores.

## Settings

Open Model service settings in the page to set the predict endpoint, the multipart field name, and
the low-confidence threshold. Settings are stored in this browser.

A URL query string overrides the stored values, which is useful for demos:

| Parameter | Meaning | Example |
| --- | --- | --- |
| `endpoint` or `api` | Predict endpoint, full URL or path | `?endpoint=http://localhost:8000` |
| `health` | Health endpoint used for the status pill | `?health=/healthz` |
| `field` | Multipart field name for the image | `?field=image` |
| `threshold` | Low-confidence threshold, `0.4` or `40` | `?threshold=45` |
| `timeout` | Request timeout in milliseconds | `?timeout=60000` |

When the endpoint has no inference verb on the end, the page appends `/predict`. So
`?endpoint=http://localhost:8000` posts to `http://localhost:8000/predict`, while
`?endpoint=http://localhost:8000/api/predict` is used unchanged.

## Other things the page does

- Drag and drop, file picker, clipboard paste, and webcam capture all feed the same input.
- Each finished run is kept in the recent inferences list with a thumbnail. The image itself lives
  in memory only, so re-analysing an entry from a previous visit asks you to pick the file again.
- The result can be copied or downloaded as JSON, including the raw model response.
- A warning appears when the winning emotion falls below the confidence threshold.
- A status pill in the header polls the health endpoint on load and when settings change. A
  `ready: false` payload counts as offline.

## Layout

```
index.html            page markup and the state panels
styles.css            all styling
js/app.js             wiring: input, inference, settings, history
js/api.js             fetch calls, endpoint resolution, timeouts, errors
js/parse.js           turns either service's response into one prediction model
js/render.js          DOM rendering for every result view
js/emotions.js        canonical emotions, the backend's label aliases, colours, emoji
js/format.js          percentage, duration, byte, and time formatting
js/config.js          settings defaults, validation, URL overrides
js/history.js         recent inference records in localStorage
js/thumbnail.js       canvas thumbnail generation
tools/mock-api.mjs    mock model service plus static file server
scripts/build.mjs     copies the static frontend into dist/
tests/                node:test suites
```

`js/parse.js` is the only file that knows the wire format. It accepts either contract above, plus a
keyed probability map, a list of label and score objects, a bare score array paired with a `labels`
array, a per-face array, a top-1 only response, and one wrapper object around any of those.
Everything downstream works with one shape, so the renderer never sorts or normalises.

Scale is normalised there too. Values that do not sum to one are rescaled, values above one are read
as a 0-100 scale, and values outside that range are read as logits and passed through a softmax. The
bars always sum to 100%.

## Develop

```bash
npm test        # 56 tests over parsing, formatting, config, API, history, and markup
npm run build   # copies index.html, styles.css, and js/ into dist/
```

The mock service takes flags for the failure paths:

```bash
node tools/mock-api.mjs --port 9000 --delay 1500   # slow inference, to see the loading state
node tools/mock-api.mjs --port 9000 --fail 503     # failing inference, to see the error state
```

It answers `/predict` with the `app/` shape and `/api/predict` with the `backend/` shape. It serves
only `index.html`, `styles.css`, and `js/`, so the rest of the repository is not exposed. Scores
derive from a hash of the uploaded bytes, so the same image always gets the same prediction.

## Browser support

Current Chromium, Firefox, and Safari. Webcam capture needs a secure context, so it works on
`localhost` and over HTTPS. The styling uses `color-mix()` and the thumbnail path uses
`createImageBitmap`, both of which need a current browser.
