# 1. Model boundary and label vocabulary

Date: 2026-09-14

## Status

Accepted

## Context

The repository held a README title and nothing else, so the backend had to
choose where the model stops and the service starts. Three questions came up
while building it, and each one has a plausible answer that later causes a
silent wrong result.

Where does softmax run? A Hugging Face `ViTForImageClassification` returns
logits. Calling `softmax` in each place that needs probabilities (the endpoint,
the evaluator, the batch loop) means three places to keep in step, and a
backend that already returns probabilities gets softmaxed twice.

What is the label order? `config.id2label` is a dict keyed by output index. The
default checkpoint
`dima806/facial_emotions_image_detection` orders its outputs as
`sad, disgust, angry, neutral, fear, surprise, happy`, not in FER2013 order. Any
code that assumes FER2013 positions reads `index 3` as `happy` when the model
means `neutral`, and reports it with high confidence.

What happens when the checkpoint is missing? The machine that runs the tests
has no torch and no weights. A service that refuses to start there cannot be
tested, smoke-checked, or deployed before the weights are ready.

## Decision

Backends return logits and nothing else. `EmotionBackend.predict` returns one
raw score row per image, in the order the backend declares. `softmax` runs
exactly once, in `EmotionPredictor.build_prediction`, which also applies the
configured temperature. A backend that wants to expose probabilities
implements `EmotionBackend` and converts its own output to logits first.

Each backend owns its label order and publishes it through
`descriptor.labels`. The ViT adapter reads `config.id2label` and keeps that
order, so `labels[i]` is the class for output `i`. If the map is missing, short,
or produces duplicate names after normalisation, the adapter substitutes the
canonical seven and logs the substitution rather than guessing positions.
`POST /model/info` reports the order in use.

`app/emotions.py` holds the canonical vocabulary: the seven FER2013 labels,
aliases that map checkpoint spellings onto them (`happiness` to `happy`,
`contempt` to `disgust`), and Russell circumplex anchors used to turn a
distribution into valence and arousal. Dataset directory names and checkpoint
label strings both pass through `normalize_label`, so `Sad/` and `sadness` land
on the same class.

Startup resolution follows `EMOTION_API_BACKEND`. `auto` tries the checkpoint
and falls back to the reference backend, recording the reason in
`/model/info`. `vit` treats a missing checkpoint as a startup error, because the
operator asked for it by name. Every response names the backend that answered,
and `ReferenceBackend` declares `neural=false` so a caller can tell a placeholder
answer from a model answer.

## Consequences

A new backend implements one method and one descriptor. It cannot get the
softmax or the label order wrong, because neither is its job.

The API can be installed, started, tested and evaluated without torch,
transformers, or network access. `auto` keeps a service deployed without
weights answering requests instead of failing its health check.

Reading a probability requires the label list from the same response, since
`probabilities` is an object keyed by label rather than an array sized by class
count. Evaluating a checkpoint that reports labels outside the canonical seven
produces per-class rows for those extra labels, and those labels contribute
nothing to valence or arousal.

The reference backend is a fixed weight table over eight Pillow statistics. It
is not trained, it is not a measurement of emotion, and the docs say so
wherever it can be reached. Its job is to keep the transport, the metrics and
the CLI exercised, and to keep `auto` honest about which answer a caller got.
