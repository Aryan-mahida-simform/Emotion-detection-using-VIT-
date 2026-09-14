"""The page and the JSON API answer on one origin.

The page's own modules are loaded here through node, so a change to either the
API response or the page's parser fails this file rather than only the browser.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.emotions import CANONICAL_EMOTIONS

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is needed to run the page's modules")

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "frontend"

CONFIG_PROBE = """
import { pathToFileURL } from "node:url";

const [configPath, apiPath] = process.argv.slice(2);
const { normalizeSettings } = await import(pathToFileURL(configPath).href);
const { resolveHealthUrl, resolvePredictUrl } = await import(pathToFileURL(apiPath).href);
const settings = normalizeSettings();
process.stdout.write(
  JSON.stringify({
    predict: resolvePredictUrl(settings.endpoint),
    health: resolveHealthUrl(settings.healthEndpoint),
    field: settings.uploadField,
  })
);
"""

PARSE_PROBE = """
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const [payloadPath, parsePath] = process.argv.slice(2);
const { parsePrediction } = await import(pathToFileURL(parsePath).href);
const payload = JSON.parse(await readFile(payloadPath, "utf8"));
const prediction = parsePrediction(payload);
process.stdout.write(
  JSON.stringify({
    top: prediction.top.label,
    scores: prediction.scores.length,
    distributionReported: prediction.distributionReported,
    latencyMs: prediction.latencyMs,
  })
);
"""


def run_probe(script: str, *arguments: Path) -> Any:
    """Run one page module in node and read back the JSON it prints."""

    with tempfile.TemporaryDirectory() as directory:
        probe = Path(directory) / "probe.mjs"
        probe.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [NODE, str(probe), *(str(argument) for argument in arguments)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


PRIVATE_PATHS = (
    "/package.json",
    "/README.md",
    "/tests/parse.test.mjs",
    "/tools/mock-api.mjs",
)

ALLOWLIST_ESCAPE_PATHS = (
    "/js/%2e%2e/package.json",
    "/js/%2e%2e%2fREADME.md",
    "/js/%2e%2e/tests/parse.test.mjs",
)


def test_the_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<script type="module" src="js/app.js">' in response.text


def test_the_page_files_are_served_beside_it(client: TestClient) -> None:
    for path in ("/styles.css", "/js/app.js", "/js/api.js", "/js/parse.js"):
        assert client.get(path).status_code == 200, path


@pytest.mark.parametrize("path", PRIVATE_PATHS + ALLOWLIST_ESCAPE_PATHS)
def test_the_rest_of_the_checkout_stays_private(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404, path


def test_the_api_routes_keep_answering(client: TestClient, png_bytes: bytes) -> None:
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/predict", files={"file": ("face.png", png_bytes, "image/png")})
    assert response.status_code == 200
    assert response.json()["label"] in CANONICAL_EMOTIONS


def test_the_banner_names_the_page(client: TestClient) -> None:
    assert client.get("/service").json()["endpoints"]["web_app"] == "/"


@needs_node
def test_the_page_defaults_point_at_the_paths_this_service_serves(client: TestClient) -> None:
    endpoints = client.get("/service").json()["endpoints"]
    probe = run_probe(CONFIG_PROBE, FRONTEND / "js/config.js", FRONTEND / "js/api.js")

    assert probe["predict"] == endpoints["predict_upload"] == "/predict"
    assert probe["health"] == endpoints["health"] == "/health"
    assert probe["field"] == "file"


@needs_node
def test_the_page_parser_reads_a_live_prediction(stub_client, png_bytes: bytes, tmp_path: Path) -> None:
    client = stub_client(logits=[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    response = client.post(
        "/predict?top_k=7", files={"file": ("face.png", png_bytes, "image/png")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "happy"

    body = tmp_path / "prediction.json"
    body.write_text(json.dumps(payload), encoding="utf-8")
    probe = run_probe(PARSE_PROBE, body, FRONTEND / "js/parse.js")

    assert probe["top"] == payload["label"] == "happy"
    assert probe["scores"] == len(payload["scores"]) == len(CANONICAL_EMOTIONS)
    assert probe["distributionReported"] is True
    assert probe["latencyMs"] == pytest.approx(payload["latency_ms"])
