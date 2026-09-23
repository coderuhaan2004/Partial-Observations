#!/usr/bin/env bash

#############################################################################
# Install PyADI-IIO and the application dependencies without building Python.
#
# The supported Python version must already be installed and available as
# python3, or supplied through PYTHON_BIN.
#
# Usage:
#   bash install-no-python.sh
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python3  VENV_DIR=/path/to/.venv  MAKE_JOBS=2
#############################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
MAKE_JOBS="${MAKE_JOBS:-2}"

log() { echo -e "${GREEN}$*${NC}"; }
step() { echo -e "${YELLOW}$*${NC}"; }
fail() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

[[ "${EUID}" -eq 0 ]] && fail "Run this script as a normal user with sudo access, not as root."
[[ "${OSTYPE}" == linux-gnu* ]] || fail "This script is designed for Linux."
command -v sudo >/dev/null 2>&1 || fail "sudo is required."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    fail "Supported Python was not found: ${PYTHON_BIN}"

python_version="$("${PYTHON_BIN}" --version 2>&1)"

log "========================================"
log "PyADI-IIO Installation"
log "========================================"

step "[1/6] Updating system packages..."
sudo apt-get update

step "[2/6] Installing compilers and native library dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git pkg-config \
    libusb-1.0-0-dev libxml2-dev bison flex \
    libavahi-common-dev libavahi-client-dev libaio-dev \
    uuid-dev ca-certificates \
    python3-venv python3-pip python3-tk

"${PYTHON_BIN}" -c 'import venv' >/dev/null 2>&1 || \
    fail "${python_version} does not provide the venv module after installing python3-venv."

build_libiio() {
    local build_dir
    build_dir="$(mktemp -d)"

    step "Building libiio from source..."
    git clone --depth 1 --branch libiio-v0 \
        https://github.com/analogdevicesinc/libiio.git "${build_dir}/libiio"
    cmake -S "${build_dir}/libiio" -B "${build_dir}/libiio/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build "${build_dir}/libiio/build" -j"${MAKE_JOBS}"
    sudo cmake --install "${build_dir}/libiio/build"
    rm -rf "${build_dir}"
}

build_libad9361() {
    local build_dir
    build_dir="$(mktemp -d)"

    step "Building libad9361-iio from source..."
    git clone --depth 1 \
        https://github.com/analogdevicesinc/libad9361-iio.git \
        "${build_dir}/libad9361-iio"
    cmake -S "${build_dir}/libad9361-iio" -B "${build_dir}/libad9361-iio/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build "${build_dir}/libad9361-iio/build" -j"${MAKE_JOBS}"
    sudo cmake --install "${build_dir}/libad9361-iio/build"
    rm -rf "${build_dir}"
}

step "[3/6] Building libiio and libad9361-iio from source..."
build_libiio
build_libad9361
sudo ldconfig

step "[4/6] Creating the Python virtual environment with ${python_version}..."
if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel

step "[5/6] Installing Python packages from pip..."
requirements_file="${SCRIPT_DIR}/requirements.txt"
if [[ -f "${requirements_file}" ]]; then
    grep -Ev '^[[:space:]]*(tkinter|pyadi-iio)([<>=!~].*)?[[:space:]]*$' \
        "${requirements_file}" > "${VENV_DIR}/requirements-pip.txt"
    python3 -m pip install -r "${VENV_DIR}/requirements-pip.txt"
fi
python3 -m pip install pyadi-iio pylibiio==0.23.1 scipy

step "[6/6] Verifying the installation..."
python3 - <<'PY'
import adi
import iio
import matplotlib
import numpy
import tkinter
import yaml

matplotlib.use("TkAgg")
print("PyADI-IIO, NumPy, SciPy, Matplotlib, PyYAML, pylibiio, and Tk are ready.")
PY

log "========================================"
log "Installation completed successfully"
log "========================================"
echo "Virtual environment: ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
echo "Run the application with: python3 ${SCRIPT_DIR}/src/secondary-user.py <pluto-uri>"
python3 -m pip list
