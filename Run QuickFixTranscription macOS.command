#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
DEPENDENCY_ROOT="$(dirname "$(pwd)")/QuickFixAppDependencies"
VENV_PYTHON="$DEPENDENCY_ROOT/.venvs/QuickFixTranscription/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    ./agent-bootstrap.sh --yes || {
        echo "Setup did not complete. Run ./agent-bootstrap.sh manually to see details."
        read -r _
        exit 1
    }
fi

PYTHON=""
if [ -x "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
fi

if [ -n "$PYTHON" ]; then
    if [ -x "$VENV_PYTHON" ]; then
        "$PYTHON" -m pip install -r requirements.txt || echo "Python package setup did not complete. The app may be missing optional local features."
    fi
    "$PYTHON" -m transcription.model_setup --yes || echo "Runtime asset setup did not complete. You can still choose local paths in the app."
    exec "$PYTHON" ./main.py "$@"
fi

echo "python3 was not found."
echo "Run ./agent-bootstrap.sh first, then run this again."
read -r _
exit 1
