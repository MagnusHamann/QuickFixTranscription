#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
DEPENDENCY_ROOT="$(dirname "$(pwd)")/QuickFixAppDependencies"
VENV_PYTHON="$DEPENDENCY_ROOT/.venvs/QuickFixTranscription/bin/python"

python_ready() {
    python_path="$1"
    [ -x "$python_path" ] && "$python_path" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

whisper_ready() {
    for candidate in \
        "$DEPENDENCY_ROOT/.tools/whisper/whisper-cpp" \
        "$DEPENDENCY_ROOT/.tools/whisper/whisper-cli" \
        "$DEPENDENCY_ROOT/.tools/whisper/main"; do
        [ -x "$candidate" ] && return 0
    done
    command -v whisper-cpp >/dev/null 2>&1 || command -v whisper-cli >/dev/null 2>&1 || command -v whisper >/dev/null 2>&1
}

needs_bootstrap=0
if ! python_ready "$VENV_PYTHON"; then
    needs_bootstrap=1
fi
if ! whisper_ready; then
    needs_bootstrap=1
fi

if [ "$needs_bootstrap" -eq 1 ]; then
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
