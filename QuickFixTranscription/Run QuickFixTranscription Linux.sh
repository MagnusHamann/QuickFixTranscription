#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

PYTHON=""
if [ -x ./.venv/bin/python ]; then
    PYTHON="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
fi

if [ -n "$PYTHON" ]; then
    "$PYTHON" -m transcription.model_setup --yes || echo "Default Whisper model setup did not complete. You can still choose a local model in the app."
    exec "$PYTHON" ./main.py "$@"
fi

echo "python3 was not found."
echo "Run ./agent-bootstrap.sh first, then run this again."
exit 1
