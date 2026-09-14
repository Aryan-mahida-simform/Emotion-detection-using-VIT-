"""Serve the browser page from the same origin as the JSON API.

The page posts to the relative paths ``/predict`` and ``/health``. Serving it
from this process is what makes those defaults resolve with no configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "frontend" / "dist"
SOURCE_DIR = REPO_ROOT / "frontend"
PUBLIC_ENTRIES = ("index.html", "styles.css", "js")


def find_frontend(configured: Optional[str] = None) -> Optional[Path]:
    """Return the directory that holds ``index.html``, or None when there is none."""

    candidates = [Path(configured)] if configured else [BUNDLE_DIR, SOURCE_DIR]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    return None


def resolve_asset(root: Path, requested: str) -> Optional[Path]:
    """Return one file the page is allowed to expose, or None."""

    if not requested:
        return root / "index.html"
    candidate = (root / requested).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    if candidate.relative_to(root).parts[0] not in PUBLIC_ENTRIES:
        return None
    return candidate


def mount_frontend(app: FastAPI, root: Optional[Path]) -> None:
    """Serve the page at ``/`` and its files beside it.

    Only ``index.html``, ``styles.css`` and ``js`` are reachable, so the rest of
    the checkout stays private. API routes are registered first and win.
    """

    if root is None:
        logger.info("no front end found; the service banner answers at / as well as /service")
        return

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(root / "index.html")

    @app.get("/{asset_path:path}", include_in_schema=False)
    def asset(asset_path: str) -> FileResponse:
        found = resolve_asset(root, asset_path)
        if found is None:
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(found)

    logger.info("serving the front end from %s", root)
