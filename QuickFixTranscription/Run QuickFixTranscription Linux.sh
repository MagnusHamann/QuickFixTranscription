#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

if [ -x ./.venv/bin/python ]; then
    exec ./.venv/bin/python ./main.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 ./main.py "$@"
fi

echo "python3 was not found."
echo "Run ./agent-bootstrap.sh first, then run this again."
exit 1
