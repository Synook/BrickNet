#!/bin/bash
# Cloud-session setup for hand-authoring BrickNet models (scripts/author.py):
#   .venv/                       bricknet (editable) + pybullet, Playwright, Pillow, bricknet-render + bpy, linters
#   scripts/viewer/node_modules  three.js for the WebGL viewer
#   <BrickNet data dir>/cvx      convex colliders (collision checks), 191 MiB download
#   <BrickNet data dir>/glb      part meshes (both renderers), 323 MiB download
# Idempotent: finished steps are skipped. The first run compiles pybullet from source (~10 min).
# The mesh downloads need network access to keeper.mpdl.mpg.de.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
VENV="$CLAUDE_PROJECT_DIR/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
# the venv's own setuptools: Debian's patched one cannot build pybullet
"$PY" -m pip install -q --upgrade pip setuptools wheel
"$PY" -m pip install -q -e . \
  "playwright==1.56.0" "pillow" \
  "bricknet-render==0.1.0" "bpy==5.2.2" \
  "black==24.10.0" "flake8==7.1.1"

npm install --prefix scripts/viewer --no-audit --no-fund --loglevel=error

# importing pybullet prints a banner to stdout; the path is the last line
DATA="$("$PY" -c 'from bricknet.collision import data_dir; print(data_dir())' | tail -n 1)"
has() { [ -n "$(find "$1" -maxdepth 1 -name "$2" -print -quit 2>/dev/null)" ]; }
if ! has "$DATA/cvx" '*.cvx.obj'; then
  "$PY" -m bricknet fetch-meshes || echo "warning: collider download failed; collision checks unavailable" >&2
fi
if ! has "$DATA/glb" '*.glb'; then
  "$PY" -m bricknet_render fetch-glbs || echo "warning: GLB download failed; rendering unavailable" >&2
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export VIRTUAL_ENV=\"$VENV\"" >> "$CLAUDE_ENV_FILE"
  echo "export PATH=\"$VENV/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
