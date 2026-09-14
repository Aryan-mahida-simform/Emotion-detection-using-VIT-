# 3. The API serves the page, so the front end and the back end share one origin

Date: 2026-09-14

## Status

Accepted. This settles the open question in [2](0002-frontend-normalises-model-response-shapes.md):
which service the page targets.

## Context

The page and the API were built as separate workstreams. The page posts to the
relative paths `/predict` and `/health` by default, and it is plain ES modules
with no build step and no runtime dependencies. The API lived in a different tree
and answered its service banner at `/`.

Nothing joined them. Served as they were, the page on `http://localhost:8080`
would post to `http://localhost:8080/predict` and reach nothing, and both
`/predict` and `/model/info` on the API would need their origin repeated in the
page's settings before a single prediction worked.

Three ways to close that gap looked reasonable:

- Serve the page from a second process and give it the API origin, either through
  the settings panel or through CORS plus an absolute endpoint.
- Add a dev proxy, the way a Vite project does, and accept that production needs a
  web server in front of both.
- Serve the page from the API process itself.

## Decision

`app/spa.py` serves the page from the API process, at `/`. The page's relative
defaults then resolve against the origin it was loaded from, and the connection
needs no configuration.

The served files are `index.html`, `styles.css`, and `js`, taken from
`frontend/dist` when the page has been built and from `frontend/` when it has not.
Every other path answers 404, so the test suite, the mock service, and
`package.json` stay off the wire. The allowlist check runs on the resolved path,
not on the raw request string: checking the request string first lets `js/../package.json`
and its percent-encoded forms past the check, because `Path.resolve` collapses the
dot segments after it. `tests/test_frontend_integration.py` requests both forms.

The API routes are registered before the catch-all, so `/predict`, `/health`,
`/model/info`, and the rest keep their behaviour and their JSON.

The service banner moved from `/` to `/service`, because `/` now belongs to the
page. `/service` keeps the banner in the same shape and adds a `web_app` entry when
a page is being served. Requests to `/` reach the banner only when no page was
found, which keeps a checkout without a front end useful.

CORS origins stay in the settings for the case where the page is served elsewhere,
and are not needed when it is served from this process.

## Consequences

One process, one port, one origin. `uvicorn app.main:app` serves a working page at
`http://127.0.0.1:8000/` and a working API beside it, with no settings to fill in
and no second process to start. The dev-server proxy and its two ports are not
needed to run the app, and the page's settings panel becomes a way to point at a
different service rather than a required first step.

The cost is one exception to the page's own routing. `/` no longer returns the
banner, so a caller that read service metadata from `/` must read it from
`/service`. No route under the API's documented prefixes changed.

The page's response parsing did not change. `app/schemas.py` already produced the
shape `js/parse.js` reads first, which is why the page renders a live prediction
untouched. `tests/test_frontend_integration.py` holds that on both sides: it drives
the API with a stub whose top class it chooses, and runs the page's own
`js/parse.js` over the real response through node, so a change to either side fails
in CI rather than in the browser.
