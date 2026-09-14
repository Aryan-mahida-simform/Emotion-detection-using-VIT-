# 2. The frontend reads both service contracts instead of pinning one

Date: 2026-05-01

## Status

Accepted, and narrowed by [3](0003-frontend-and-api-are-one-service.md): the page now posts to the
`app/` service, and the second service named below no longer exists in the repository. The permissive
parser stays, so the page still renders a response that only reports a top class.

## Context

Two FastAPI services for this model exist in the repository as separate workstreams, and they
describe the same prediction differently.

`app/schemas.py` returns a top label, a ranked `scores` list, a `probabilities` map, an `affect`
pair, and a nested `model` descriptor holding the backend, model id, device, and label list.

`backend/app/schemas/prediction.py` returns a label and confidence, plus an emoji per score, a
source `image` descriptor, a `face_crop` in pixel coordinates, `elapsed_ms`, and `model_input_size`.

They agree on the multipart field name, `file`, and on the error envelope,
`{"error": {"code", "message", "details"}}`. They disagree on nearly everything else, including the
path: `/predict` against `/api/predict`.

Only one of them can be the frontend's target on the day it is wired up, and nothing in the
repository says which. A frontend that read one exactly would fail on the other, and the failure
would look like a frontend bug.

## Decision

`js/parse.js` reads both, plus the looser shapes a hand-rolled service tends to emit: a keyed
probability map, a list of label and score objects, a bare score array paired with a `labels` array,
a per-face array, a top-1 only response, and one wrapper object around any of those. Everything
downstream works with one shape:

```
{ faces, scores, distributionReported, top, affect, model, latencyMs,
  image, modelInput, faceDetected, count }
```

`top` is a `{ label, score, emoji }` entry and `scores` is sorted descending, so the renderer never
sorts or normalises.

The multipart field name, the endpoint, and the health endpoint are settings, defaulting to `file`,
`/predict`, and `/health`. A wrong field name produces a 422 from FastAPI with a validation body
that names no field, so making it settable turns an unreadable failure into a one-line fix.

Label normalisation mirrors `app/emotions.py`. Its alias table maps `happiness`, `joy`, and `joyful`
onto `happy`, and `anger` and `rage` onto `angry`. The frontend applies the same aliases so that an
alias and its canonical name cannot both appear as separate bars, splitting the probability mass.

Scale reconciliation also lives in `parse.js`. Values that do not sum to one are rescaled, values
above one are read as a 0-100 scale, and values outside that range are read as logits and passed
through a softmax. Box coordinates above 1 are read as pixels and those at or below 1 as fractions,
which is why `face_crop` from the `backend/` service and a fractional `box` both render.

## Consequences

The frontend works against either service on the first run, and against a third one that follows
neither schema exactly. The cost is roughly 300 lines in `js/parse.js` and its test suite, and the
looseness means a genuinely malformed response can be read as a valid one.

That risk is bounded by the display. Every result is drawn as percentages that sum to 100, and both
rescaling and softmax preserve order, so a misread scale changes no ranking. `distributionReported`
records whether the service sent a full distribution or only a top class, and the result panel
labels the latter as partial, so a single bar is never mistaken for a confident win.

`tests/parse.test.mjs` holds both real payloads verbatim, taken from the two schema files. If either
service changes shape, that is where it fails, and the fixtures say which contract broke.

If one contract is later retired, `js/parse.js` is the only file to narrow and the fixtures name the
shapes that would be dropped.
