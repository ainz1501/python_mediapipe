#!/usr/bin/env bash
# Set up Python dependencies required by triangulate3DHPE_show_result.py
# Usage (with optional overrides):
#   chmod +x setup_triangulate3dhpe_show_result.sh
#   ./setup_triangulate3dhpe_show_result.sh [--no-system-packages]

set -euo pipefail

NEED_SYSTEM_PACKAGES=1

for arg in "$@"; do
    case "$arg" in
        --no-system-packages)
            NEED_SYSTEM_PACKAGES=0
            shift
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [[ $NEED_SYSTEM_PACKAGES -eq 1 && -x "$(command -v sudo)" && -x "$(command -v apt-get)" ]]; then
    echo "Installing required system packages via apt-get..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-dev libgl1 libglib2.0-0
else
    echo "Skipping system package installation."
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable '$PYTHON_BIN' not found. Set PYTHON_BIN to a valid interpreter." >&2
    exit 1
fi

VENV_DIR=${VENV_DIR:-.venv_triangulate3dhpe}
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at '$VENV_DIR'..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at '$VENV_DIR'."
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade \
    mediapipe \
    opencv-python \
    matplotlib \
    numpy

echo "All dependencies installed successfully."
echo "Activate the virtual environment with: source \"$VENV_DIR/bin/activate\""
