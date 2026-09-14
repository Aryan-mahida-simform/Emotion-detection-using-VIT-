# 1. The frontend normalises whatever response shape the model service returns

Date: 2026-05-01

## Status

Accepted

## Context

The repository held only a README title when this frontend was built. No inference server, no
serialisation code, and no sample response existed, so the wire format of the ViT emotion model is
unknown. The model itself is not in this repository.

A frontend that only reads one exact JSON shape would break the moment the real service is wired up,
and the person wiring it up would have no way to tell whether the frontend or the server was wrong.
Two candidate directions were available:

1. Pin one format, document it as a contract, and require the model service to match it.
2. Accept the formats a PyTorch or FastAPI emotion service plausibly emits and reconcile them in one
   module.

## Decision

We take direction 2. `js/parse.js` is the only place that knows about wire format. It accepts a
keyed probability map, a list of label and score objects, a bare score array paired with a `labels`
array, a per-face array, a top-1 only response, and one level of wrapper object around any of those.
Everything downstream of `parsePrediction` works with a single shape:

```
{ faces, scores, distributionReported, top, latencyMs, model, device }
```

`top` is `{ label, score }` and `scores` is sorted descending, so the renderer never sorts or
normalises anything.

Scale reconciliation also lives in `parse.js`. Values that do not sum to one are rescaled to sum to
one. Values above one are read as a 0-100 scale. Values outside 0-100 are read as logits and passed
through a softmax. Duplicate labels after lowercasing are summed rather than dropped, because a
service that emits both `happy` and `Happy` should not lose probability mass.

The request direction is pinned instead of flexible. The page posts `multipart/form-data` with the
file field named `image`, which is what FastAPI's `UploadFile` and Flask's `request.files` expect by
default.

## Consequences

Pinning the request and tolerating the response costs about 250 lines in `js/parse.js` and a test
suite of 15 cases, and it removes a coordination round trip with whoever deploys the model. The
frontend works on the first try against a service that reports probabilities, against one that
reports logits, and against one that only returns a predicted class.

The cost is that a genuinely malformed response can be interpreted as a valid one. A logits vector
that happens to sit inside 0-100 is read as a percentage scale rather than as logits. We accept this
because the failure mode is visible: the bars are always shown as percentages that sum to 100, and a
wrong scale changes the ranking of nothing, since both rescaling and softmax are monotonic.

`distributionReported` records whether the service sent a full distribution or only a top class. The
result panel reads it to label the result as partial, so a user is never shown a single bar that
looks like a confident one-horse race.

If the model service later publishes a fixed schema, `parse.js` is the only file to narrow, and the
tests in `tests/parse.test.mjs` already cover the shapes that would be dropped.
