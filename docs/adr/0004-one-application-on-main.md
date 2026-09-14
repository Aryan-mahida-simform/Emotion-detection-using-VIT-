# 4. `main` carries the `app/` service and the dependency-free page

Date: 2026-09-14

## Status

Accepted. Narrows [2](0002-frontend-normalises-model-response-shapes.md), whose status note already
asserted that the `backend/` service "no longer exists in the repository". Until this decision that
was only true of the parser's assumptions, not of `main`.

## Context

Twenty-eight branches held two complete, independently built applications for the same model, plus
the skeleton branches that led to each. `main` held a one-line `README.md`.

One lineage is a FastAPI package under `app/` with a browser page of plain HTML, CSS and ES modules
in `frontend/`, no build step and no runtime dependencies. The other is a FastAPI package under
`backend/` with a React and TypeScript app in `frontend/`, built by Vite into a committed
`frontend/dist`, plus a separate `training/` pipeline that fine-tunes BEiT and exports the weights.

They disagree on the package name, the route prefix (`/predict` against `/api/predict`), the front
end's toolchain, and the prediction JSON. A checkout could only run one of them, and `main` named
neither. Every reader had to rediscover the choice by reading branch history.

## Decision

`main` is fast-forwarded onto the `app/` lineage, `feature/pex-ai-0d713bbe-integration` at
`8203ced`. It is the only lineage that is complete on its own: the API serves the page from its own
process ([3](0003-frontend-and-api-are-one-service.md)), the page needs no toolchain to run, and the
service answers without torch by falling back to the reference backend. Nothing was lost in the
move, because the merge was a fast-forward and the old `main` tip is an ancestor.

The `backend/` and React lineage stays on its own branches. It is not deleted and it is not merged.
Its `training/` pipeline is the part with no counterpart here, so anyone returning to it should
expect the two trees to be treated as separate applications rather than reconciled file by file.

The FastAPI package is therefore `app/`, and the page is the ES-module tree in `frontend/` served
unbuilt. `training/` is not on `main`.

## Consequences

`git clone` then `.venv/bin/python -m uvicorn app.main:app` yields a working app with no Node step,
which is what `README.md` documents. `scripts/dev.sh` starts the same process and, beside it, a
static server for the page on a second port, so the page can be reloaded without restarting the API;
the page reaches the API across origins through the endpoint query string that
`EMOTION_API_CORS_ORIGINS` already permits.

Two costs follow. First, `requirements-vit.txt` is the only route to real ViT inference, and with
neither torch nor a checkpoint present the service answers from the reference backend and says so in
`fallback_reason`; the default suite therefore tests the reference backend rather than the model.
Second, fine-tuning is not available on `main`, so a checkpoint must come from the hub or from the
branch that still carries `training/`.
