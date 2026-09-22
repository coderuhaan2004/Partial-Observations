#!/usr/bin/env bash

#############################################################################
# Install PyADI-IIO and the application dependencies on a Jetson Nano
# running Ubuntu 18.04.
#
# Usage:
#   bash install.sh
#
# Optional environment variables:
#   PYTHON_VERSION=3.10.13  PYTHON_PREFIX=/usr/local
#   VENV_DIR=/path/to/.venv  MAKE_JOBS=2
#############################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.13}"
PYTHON_PREFIX="${PYTHON_PREFIX:-/usr/local}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
MAKE_JOBS="${MAKE_JOBS:-2}"
PYTHON_BIN="${PYTHON_PREFIX}/bin/python${PYTHON_VERSION%.*}"

log() { echo -e "${GREEN}$*${NC}"; }
step() { echo -e "${YELLOW}$*${NC}"; }
fail() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

[[ "${EUID}" -eq 0 ]] && fail "Run this script as a normal user with sudo access, not as root."
[[ "${OSTYPE}" == linux-gnu* ]] || fail "This script is designed for Linux."

source /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 18.04 ]] || \
    fail "Ubuntu 18.04 is required; found ${PRETTY_NAME:-unknown}."

if [[ "$(uname -m)" != aarch64 ]]; then
    echo -e "${YELLOW}Warning: this is not an aarch64 Jetson; continuing anyway.${NC}"
fi

log "========================================"
log "Jetson Nano PyADI-IIO Installation"
log "========================================"

step "[1/7] Updating system packages..."
sudo apt-get update

step "[2/7] Installing compilers, Python, Tk, and native libraries..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git pkg-config sudo \
    libusb-1.0-0-dev libxml2-dev bison flex \
    libavahi-common-dev libavahi-client-dev libaio-dev \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev libffi-dev liblzma-dev \
    libgdbm-dev libgdbm-compat-dev uuid-dev tk-dev \
    ca-certificates wget

build_python() {
    local archive="Python-${PYTHON_VERSION}.tgz"
    local source_dir="Python-${PYTHON_VERSION}"
    local build_dir
    build_dir="$(mktemp -d)"
    trap 'rm -rf "${build_dir}"' RETURN

    step "Building Python ${PYTHON_VERSION} from source..."
    wget -q --show-progress \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/${archive}" \
        -O "${build_dir}/${archive}"
    tar -xzf "${build_dir}/${archive}" -C "${build_dir}"
    cd "${build_dir}/${source_dir}"
    ./configure --prefix="${PYTHON_PREFIX}" --with-ensurepip=install
    make -j"${MAKE_JOBS}"
    sudo make altinstall
    cd "${SCRIPT_DIR}"
    log "Python ${PYTHON_VERSION} installed at ${PYTHON_BIN}"
}

step "[3/7] Installing Python ${PYTHON_VERSION}..."
if [[ ! -x "${PYTHON_BIN}" ]]; then
    build_python
else
    log "Python ${PYTHON_VERSION%.*} already exists; skipping the source build."
fi

"${PYTHON_BIN}" -c 'import tkinter, ssl, sqlite3; print("Python Tk/SSL/SQLite checks passed")'

build_libiio() {
    local build_dir
    build_dir="$(mktemp -d)"
    trap 'rm -rf "${build_dir}"' RETURN
    step "Building libiio from source..."
    git clone --depth 1 --branch libiio-v0 \
        https://github.com/analogdevicesinc/libiio.git "${build_dir}/libiio"
    cmake -S "${build_dir}/libiio" -B "${build_dir}/libiio/build" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}/libiio/build" -j"${MAKE_JOBS}"
    sudo cmake --install "${build_dir}/libiio/build"
}

build_libad9361() {
    local build_dir
    build_dir="$(mktemp -d)"
    trap 'rm -rf "${build_dir}"' RETURN
    step "Building libad9361-iio from source..."
    git clone --depth 1 \
        https://github.com/analogdevicesinc/libad9361-iio.git \
        "${build_dir}/libad9361-iio"
    cmake -S "${build_dir}/libad9361-iio" -B "${build_dir}/libad9361-iio/build" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}/libad9361-iio/build" -j"${MAKE_JOBS}"
    sudo cmake --install "${build_dir}/libad9361-iio/build"
}

step "[4/7] Installing libiio and libad9361-iio..."
build_libiio
build_libad9361
sudo ldconfig

step "[5/7] Creating the Python virtual environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

step "[6/7] Installing Python packages from pip..."
# tkinter is supplied by tk-dev, not PyPI. pyadi-iio is installed from PyPI.
requirements_file="${SCRIPT_DIR}/requirements.txt"
if [[ -f "${requirements_file}" ]]; then
    grep -Ev '^[[:space:]]*(tkinter|pyadi-iio)([<>=!~].*)?[[:space:]]*$' \
        "${requirements_file}" > "${VENV_DIR}/requirements-pip.txt"
    python -m pip install -r "${VENV_DIR}/requirements-pip.txt"
fi
python -m pip install pyadi-iio pylibiio==0.23.1 scipy

step "[7/7] Verifying the installation..."
python - <<'PY'
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
echo "Run the application with: python ${SCRIPT_DIR}/src/secondary-user.py <pluto-uri>"
python -m pip list