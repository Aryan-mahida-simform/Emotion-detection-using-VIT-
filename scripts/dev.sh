#!/usr/bin/env bash
#
# Start the API and the browser page as two local servers and print their URLs.
#
# The API already serves the page at "/" (docs/adr/0003), so one process is
# enough to use the app. This script also exposes the page on its own port,
# pointed at the API, so the two halves can be opened and reloaded separately.
#
# Usage: scripts/dev.sh [--api-only | --web-only]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-8080}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="${PYTHON:-$VENV_DIR/bin/python}"

mode=both
case "${1:-}" in
  --api-only) mode=api ;;
  --web-only) mode=web ;;
  "") ;;
  *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
esac

say() { printf '%s\n' "$*"; }

# The service's libraries live in a venv, never in the system interpreter.
if [ ! -x "$PYTHON" ]; then
  say "creating the virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON" -c 'import fastapi, uvicorn, PIL, multipart' >/dev/null 2>&1; then
  say "installing the API and test dependencies into $VENV_DIR"
  "$PYTHON" -m pip install --upgrade pip >/dev/null
  "$PYTHON" -m pip install -r requirements-dev.txt
fi

# The page runs unbundled, so a build is optional. Do it when node is present
# so the API serves frontend/dist instead of the sources.
if [ "$mode" != api ] && command -v node >/dev/null 2>&1 && [ ! -d frontend/dist ]; then
  node frontend/scripts/build.mjs >/dev/null
fi

port_free() {
  "$PYTHON" - "$1" <<'PY'
import socket, sys

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

pids=()
shutdown() {
  trap - INT TERM EXIT
  say ""
  say "stopping"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap shutdown INT TERM EXIT

if [ "$mode" != web ]; then
  if ! port_free "$API_PORT"; then
    say "port $API_PORT is already in use; set API_PORT to pick another" >&2
    exit 1
  fi
  "$PYTHON" -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --log-level warning &
  pids+=("$!")
fi

if [ "$mode" != api ]; then
  if ! port_free "$WEB_PORT"; then
    say "port $WEB_PORT is already in use; set WEB_PORT to pick another" >&2
    exit 1
  fi
  "$PYTHON" -m http.server "$WEB_PORT" --bind "$API_HOST" --directory frontend >/dev/null 2>&1 &
  pids+=("$!")
fi

url_answers() {
  "$PYTHON" - "$1" <<'PY' >/dev/null 2>&1
import sys, urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    response.read()
PY
}

wait_for() {
  local url="$1" deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if url_answers "$url"; then
      return 0
    fi
    sleep 0.3
  done
  return 1
}

if [ "$mode" != web ]; then
  if wait_for "http://$API_HOST:$API_PORT/health"; then
    say "API is up"
  else
    say "the API did not answer on port $API_PORT" >&2
    exit 1
  fi
fi

if [ "$mode" != api ]; then
  if wait_for "http://$API_HOST:$WEB_PORT/"; then
    say "page server is up"
  else
    say "the page did not answer on port $WEB_PORT" >&2
    exit 1
  fi
fi

say ""
say "URLs"
if [ "$mode" != web ]; then
  say "  page        http://$API_HOST:$API_PORT/          served by the API"
  say "  API docs    http://$API_HOST:$API_PORT/docs"
fi
if [ "$mode" != api ]; then
  say "  page        http://$API_HOST:$WEB_PORT/?endpoint=http://$API_HOST:$API_PORT"
  say "              the same page on its own port, posting to the API above"
fi

if [ "$mode" != web ]; then
  "$PYTHON" - "$API_HOST" "$API_PORT" <<'PY'
import json, sys, urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/model/info", timeout=5) as response:
        info = json.load(response)
except Exception as exc:
    print(f"\nmodel: unavailable ({exc})")
else:
    print(f"\nmodel: backend={info.get('backend')} model={info.get('model_id')} device={info.get('device')}")
    for key in ("fallback_reason", "detail"):
        if info.get(key):
            print(f"       {key}: {info[key]}")
PY
fi

say ""
say "Ctrl-C stops both."
wait
