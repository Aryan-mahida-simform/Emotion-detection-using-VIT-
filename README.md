# Emotion detection using ViT

A FastAPI backend that classifies facial emotion with a Vision Transformer and
reports accuracy, per-class metrics and latency for that model.

The service answers three questions:

- What emotion does this face show? `POST /predict` returns a ranked label
  distribution, a valence and arousal score, and the model that produced it.
- Which model is running? `GET /model/info` reports the backend, checkpoint id,
  device, and the label list in output order.
- How good is that model? `python -m scripts.evaluate_model --data-dir DIR`
  scores a labelled directory and prints accuracy, per-class precision, recall
  and F1, top-k accuracy, a confusion matrix, and latency percentiles.

## Layout

```
app/
  main.py                 application factory, error handlers, lifespan
  config.py               EMOTION_API_* settings, validated with pydantic
  emotions.py             label vocabulary, aliases, circumplex anchors
  imaging.py              decode, verify, reorient and downscale uploads
  scoring.py              softmax, ranking, affine maths, percentiles
  schemas.py              request and response models
  uploads.py              size-capped body reads, base64 decoding
  api/                    health, model and prediction routes
  backends/               classifier adapters behind one contract
  services/predictor.py   decode, classify, shape the response
  evaluation/             metrics, dataset walker, CLI runner
scripts/
  evaluate_model.py       score a labelled dataset
  make_samples.py         write placeholder images for smoke tests
tests/                    152 tests, 3 of them need checkpoint weights
```

## Install

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` covers the API and the tests. PyTorch and transformers are
separate, because the service runs without them:

```bash
python -m pip install -r requirements-vit.txt
```

## Run

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# or
make run
```

Interactive docs are at `/docs`.

## Backends

The service resolves one backend at startup, following `EMOTION_API_BACKEND`.

| value | behaviour |
| --- | --- |
| `auto` | Load the ViT checkpoint. If that fails, log the reason and run the reference backend. |
| `vit` | Load the ViT checkpoint. Fail startup if it cannot load. |
| `reference` | Never load a checkpoint. Used by the tests. |

`vit` wraps any `ViTForImageClassification` checkpoint. The default is
`dima806/facial_emotions_image_detection`, which ships seven FER2013 classes.
The adapter reads `config.id2label` and keeps the checkpoint's own output
order, because swapping positions silently would produce confident nonsense.
Labels are normalised to lowercase slugs (`Happiness` and `happiness` both
become `happy`), and a checkpoint whose label map is missing, short, or holds
duplicates falls back to the canonical set.

Downloads are off by default. Set `EMOTION_API_ALLOW_MODEL_DOWNLOAD=true` to let
`from_pretrained` reach the Hugging Face hub, or pre-populate the cache with
`huggingface-cli download <model-id>`.

### The reference backend

`reference` computes eight image statistics with Pillow and combines them with a
fixed weight table. It is not a trained model and its output says nothing about
a person's emotional state. It exists so the API, the tests and the evaluation
harness run on a machine without torch, and so the service answers with a
labelled result instead of a 500 when the checkpoint is missing. Every response
names the backend that produced it, and `/model/info` reports the fallback
reason.

## Endpoints

| method | path | purpose |
| --- | --- | --- |
| GET | `/` | service name and endpoint list |
| GET | `/health` | liveness, active backend, uptime |
| GET | `/health/ready` | 200 when the backend has loaded, 503 otherwise |
| GET | `/model/info` | active backend, settings and upload modes |
| GET | `/model/labels` | label set, aliases and affect anchors |
| POST | `/predict` | multipart file upload |
| POST | `/predict/raw` | raw image bytes as the request body |
| POST | `/predict/base64` | JSON body with a base64 image |
| POST | `/predict/batch` | several files, per-image results |

A prediction looks like this:

```json
{
  "label": "sad",
  "confidence": 0.812344,
  "scores": [
    {"label": "sad", "probability": 0.812344},
    {"label": "neutral", "probability": 0.091233},
    {"label": "angry", "probability": 0.051002}
  ],
  "probabilities": {
    "sad": 0.812344,
    "neutral": 0.091233,
    "angry": 0.051002,
    "fear": 0.021884,
    "disgust": 0.013457,
    "surprise": 0.006012,
    "happy": 0.004068
  },
  "affect": {"valence": -0.539911, "arousal": 0.295317},
  "model": {
    "backend": "vit",
    "model_id": "dima806/facial_emotions_image_detection",
    "device": "cpu",
    "neural": true,
    "labels": ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]
  },
  "latency_ms": 214.77
}
```

`probabilities` always carries every label in the backend's order. `scores` is
the top `top_k` slice of the same ranking. `valence` and `arousal` are the mean
of the Russell circumplex anchors in `app/emotions.py`, weighted by probability.

Errors use one shape and a stable code:

```json
{"error": {"code": "invalid_image", "message": "...", "details": {}}}
```

| status | code | cause |
| --- | --- | --- |
| 413 | `payload_too_large` | image or batch over the configured limit |
| 415 | `unsupported_media_type` | content type or image format not accepted |
| 422 | `invalid_image` | body is not a decodable image |
| 422 | `invalid_request` | body or query does not match the endpoint schema |
| 502 | `model_output_invalid` | backend returned the wrong number of scores |
| 503 | `not_ready` | backend has not loaded |
| 503 | `backend_unavailable` | backend could not be resolved |

## Configuration

Every setting is an `EMOTION_API_*` environment variable with a default in
`app/config.py`. See `.env.example`.

| variable | default | meaning |
| --- | --- | --- |
| `EMOTION_API_BACKEND` | `auto` | `auto`, `vit` or `reference` |
| `EMOTION_API_MODEL_ID` | `dima806/facial_emotions_image_detection` | checkpoint id |
| `EMOTION_API_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `EMOTION_API_IMAGE_SIZE` | `224` | side of the warmup probe |
| `EMOTION_API_LOGIT_TEMPERATURE` | `1.0` | softmax temperature |
| `EMOTION_API_DEFAULT_TOP_K` | `3` | ranked classes returned by default |
| `EMOTION_API_MAX_IMAGE_BYTES` | `8388608` | per-image upload cap |
| `EMOTION_API_MAX_IMAGE_SIDE` | `2048` | images larger than this are downscaled |
| `EMOTION_API_MAX_BATCH_SIZE` | `8` | files accepted by `/predict/batch` |
| `EMOTION_API_WARMUP_ON_START` | `true` | run one image through the backend at startup |
| `EMOTION_API_ALLOW_MODEL_DOWNLOAD` | `false` | permit hub downloads |
| `EMOTION_API_CORS_ORIGINS` | `*` | comma-separated or JSON list |

## Model testing

Point the evaluator at an ImageFolder-style directory, one subdirectory per
emotion:

```
data/fer2013/
  angry/    *.png
  disgust/  *.png
  ...
```

```bash
python -m scripts.make_samples --output data/sample   # placeholder images
python -m scripts.evaluate_model --data-dir data/fer2013 --top-k 2
```

The runner prints accuracy, top-k accuracy, macro and weighted F1, per-class
precision and recall, mean confidence, the number of confident mistakes,
latency p50 and p95, and it writes the full report, confusion matrix and skip
list to JSON when you pass `--json-out`. `--min-accuracy` makes it exit 1 below
a threshold, which is how you gate a checkpoint in CI.

For a real checkpoint:

```bash
export EMOTION_API_BACKEND=vit
python -m scripts.evaluate_model --data-dir data/fer2013 --min-accuracy 0.60 --json-out reports/vit.json
```

`scripts/make_samples.py` writes colour patches, not faces. Accuracy measured on
them is meaningless, and the runner says so in its output. Use them to check
that the transport, the metrics and the CLI work before you have a dataset.

## Tests

```bash
make install-dev
python -m pytest                    # 149 pass, 3 skip, seconds, no network
```

The default suite needs no weights and no network. It covers the label
vocabulary, settings validation, image decoding, softmax and ranking maths, the
reference backend, the backend contract, the service layer, every endpoint,
the metrics against hand-computed values, and the evaluator CLI. API tests that
assert exact probabilities inject a stub backend whose logits they choose, so
1/7 per class is checked as 1/7 rather than within a tolerance.

Checkpoint tests are marked `vit` and skip unless you name a model:

```bash
pip install -r requirements-vit.txt
EMOTION_API_TEST_MODEL_ID=dima806/facial_emotions_image_detection \
EMOTION_API_TEST_ALLOW_DOWNLOAD=1 \
python -m pytest -m vit
```

They assert that the checkpoint loads, reports one label per output, and
returns finite scores whose softmax sums to 1.

## Docker

```bash
docker build -t emotion-api .                          # reference backend only
docker build --build-arg WITH_VIT=true -t emotion-api . # with torch
docker run -p 8000:8000 emotion-api
```

## Limits

- The API classifies the whole image. It does not detect or crop a face, so a
  photo with several people returns one label for the frame.
- Emotion labels are the model's guess, not a measurement of a person. The
  FER2013 training set is small, posed and heavily imbalanced toward neutral,
  and accuracy on candid photographs is lower than the published numbers.
- A checkpoint that reports labels the canonical set does not contain keeps its
  own names. `probabilities` can then hold keys outside the seven emotions, and
  those keys contribute nothing to valence or arousal.
