#!/usr/bin/env sh
set -eu

ASSUME_YES=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes|-y)
            ASSUME_YES=1
            ;;
        --help|-h)
            cat <<'HELP'
Usage: ./agent-bootstrap.sh [--yes]

Installs/checks Python, FFmpeg, creates .venv, and installs Python dependencies
for QuickFixTranscription.
HELP
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

cd "$(dirname "$0")"

confirm() {
    if [ "$ASSUME_YES" -eq 1 ]; then
        return 0
    fi
    printf "%s [y/N] " "$1"
    read -r answer
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

is_root() {
    [ "$(id -u)" -eq 0 ]
}

as_admin() {
    if is_root; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "Administrator rights are required and sudo is not installed." >&2
        echo "Run as root, install sudo, or ask the user/admin to grant privileges." >&2
        exit 1
    fi
}

package_manager() {
    for manager in apt-get dnf pacman zypper apk; do
        if command -v "$manager" >/dev/null 2>&1; then
            echo "$manager"
            return 0
        fi
    done
    return 1
}

install_linux_package() {
    manager="$1"
    shift
    case "$manager" in
        apt-get)
            as_admin apt-get update
            as_admin apt-get install -y "$@"
            ;;
        dnf)
            as_admin dnf install -y "$@"
            ;;
        pacman)
            as_admin pacman -Sy --needed --noconfirm "$@"
            ;;
        zypper)
            as_admin zypper --non-interactive install "$@"
            ;;
        apk)
            as_admin apk add "$@"
            ;;
        *)
            echo "Unsupported package manager: $manager" >&2
            exit 1
            ;;
    esac
}

install_sudo_if_possible() {
    if command -v sudo >/dev/null 2>&1 || ! is_root; then
        return 0
    fi
    manager="$(package_manager || true)"
    if [ -n "$manager" ]; then
        echo "sudo is missing and this process is root; installing sudo."
        install_linux_package "$manager" sudo
    fi
}

install_linux_prereqs() {
    manager="$(package_manager || true)"
    if [ -z "$manager" ]; then
        echo "No supported Linux package manager was found." >&2
        echo "Install python3, python3-venv, pip, and ffmpeg manually." >&2
        exit 1
    fi

    install_sudo_if_possible

    case "$manager" in
        apt-get)
            install_linux_package "$manager" python3 python3-venv python3-pip ffmpeg
            ;;
        dnf)
            install_linux_package "$manager" python3 python3-pip ffmpeg
            ;;
        pacman)
            install_linux_package "$manager" python python-pip ffmpeg
            ;;
        zypper)
            install_linux_package "$manager" python3 python3-pip ffmpeg
            ;;
        apk)
            install_linux_package "$manager" python3 py3-pip ffmpeg
            ;;
    esac
}

install_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        return 0
    fi
    if ! confirm "Homebrew is missing. Install Homebrew now?"; then
        echo "Homebrew is needed for automated macOS dependency setup." >&2
        exit 1
    fi
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

install_macos_prereqs() {
    install_homebrew
    brew install python ffmpeg
}

case "$(uname -s)" in
    Linux)
        install_linux_prereqs
        ;;
    Darwin)
        install_macos_prereqs
        ;;
    *)
        echo "agent-bootstrap.sh supports Linux and macOS." >&2
        exit 1
        ;;
esac

python3 -m venv .venv
. ./.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m transcription.model_setup --yes || echo "Default Whisper model setup did not complete. You can still choose a local model in the app."

echo "Bootstrap complete."
echo "Run QuickFixTranscription with: ./Run QuickFixTranscription Linux.sh"
echo "On macOS, use: ./Run QuickFixTranscription macOS.command"
