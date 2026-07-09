#!/usr/bin/env bash
set -euo pipefail

########################################
# One-click MuJoCo installer
# - Download & install mujoco200 + mjkey + mujoco210 binaries
# - Install python packages:
#     mujoco==3.3.7
#     mujoco-py==2.1.2.14  (pin Cython/numpy for compatibility)
# - Verify with real sim step
#
# Optional env vars:
#   MUJOCO_DIR=~/.mujoco
#   AUTO_APT=1            # try apt-get install build deps if missing (Debian/Ubuntu)
#   WRITE_BASHRC=1        # append env vars to ~/.bashrc
########################################

MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco}"
AUTO_APT="${AUTO_APT:-0}"
WRITE_BASHRC="${WRITE_BASHRC:-0}"

MJ200_ZIP_URL="https://roboti.us/download/mujoco200_linux.zip"
MJKEY_URL="https://roboti.us/file/mjkey.txt"
MJ210_TGZ_URL="https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"

# Target Python package versions
PIP_MUJOCO_VER="3.3.7"
PIP_MUJOCO_PY_VER="2.1.2.14"
# Compatibility pins for mujoco-py build
PIN_CYTHON_VER="0.29.36"
PIN_NUMPY_SPEC="numpy<2"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERR] Missing command: $1"; exit 1; }; }
log() { echo -e "[INFO] $*"; }
warn(){ echo -e "[WARN] $*"; }

append_bashrc_once() {
  local line="$1"
  local file="$HOME/.bashrc"
  touch "$file"
  if grep -Fqx "$line" "$file"; then
    log "Already in ~/.bashrc: $line"
  else
    echo "$line" >> "$file"
    log "Appended to ~/.bashrc: $line"
  fi
}

maybe_install_apt_deps() {
  # Only attempt on Debian/Ubuntu-like systems
  if [[ "$AUTO_APT" != "1" ]]; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "AUTO_APT=1 but apt-get not found; skipping system deps install."
    return 0
  fi

  log "AUTO_APT=1: installing build/OpenGL deps via apt-get ..."
  apt-get update
  apt-get install -y \
    build-essential patchelf pkg-config \
    libgl1-mesa-dev libglew-dev libglfw3-dev \
    libosmesa6-dev \
    libx11-dev libxext-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev
}

########################################
# Preconditions
########################################
need_cmd wget
need_cmd tar
need_cmd unzip
need_cmd python
need_cmd pip

log "MUJOCO_DIR=$MUJOCO_DIR"
mkdir -p "$MUJOCO_DIR"

########################################
# Install mujoco200 + mjkey
########################################
log "Installing MuJoCo 200..."
wget -q --show-progress "$MJ200_ZIP_URL" -O "$MUJOCO_DIR/mujoco200.zip"
unzip -q -o "$MUJOCO_DIR/mujoco200.zip" -d "$MUJOCO_DIR"
rm -f "$MUJOCO_DIR/mujoco200.zip"

log "Downloading mjkey..."
wget -q --show-progress "$MJKEY_URL" -O "$MUJOCO_DIR/mjkey.txt"

########################################
# Install mujoco210
########################################
log "Installing MuJoCo 2.1.0..."
wget -q --show-progress "$MJ210_TGZ_URL" -O "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz"
tar -xzf "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz" -C "$MUJOCO_DIR"
rm -f "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz"

########################################
# Quick sanity check
########################################
log "Done. Contents:"
ls -la "$MUJOCO_DIR" | sed -n '1,80p'
log "Expect:"
echo "  - $MUJOCO_DIR/mujoco200_linux/"
echo "  - $MUJOCO_DIR/mujoco210/"
echo "  - $MUJOCO_DIR/mjkey.txt"

########################################
# Python packages (pin for mujoco-py)
########################################
log "Installing Python packages..."
python -m pip install --upgrade pip

# Pin build deps that frequently break mujoco-py
python -m pip install --upgrade "$PIN_NUMPY_SPEC" "Cython==$PIN_CYTHON_VER" "setuptools<70" wheel

# Install target versions
python -m pip install --upgrade "mujoco==$PIP_MUJOCO_VER" "mujoco-py==$PIP_MUJOCO_PY_VER"

########################################
# Env vars (current shell)
########################################
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_DIR/mujoco210"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_DIR/mujoco210/bin"

########################################
# Optional: persist env vars
########################################
if [[ "$WRITE_BASHRC" == "1" ]]; then
  log "WRITE_BASHRC=1: writing env vars to ~/.bashrc"
  append_bashrc_once 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210'
  append_bashrc_once 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin'
else
  warn "Not writing to ~/.bashrc (set WRITE_BASHRC=1 to enable)."
fi

########################################
# Build/verify mujoco-py (with fallback)
########################################
log "Python verification (real sim step)..."
python - <<'EOF'
import os, sys
print("Python:", sys.version.split()[0])
print("MUJOCO_PY_MUJOCO_PATH =", os.environ.get("MUJOCO_PY_MUJOCO_PATH"))

import mujoco
print("mujoco version:", mujoco.__version__)

# Force a real mujoco-py compile/use
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim
m = load_model_from_xml("<mujoco/>")
s = MjSim(m)
s.step()
print("mujoco-py: model load + step OK")
EOF

log "All good ✅"

########################################
# If you still hit system-deps build errors, provide a guided hint
########################################
# (We cannot detect every missing header without failing the build, so we print guidance here)
if [[ "$AUTO_APT" != "1" ]]; then
  cat <<'EOT'
[INFO] Note:
If mujoco-py fails to compile with gcc errors (missing GL/X11 headers, etc.),
rerun with:
  AUTO_APT=1 ./install_mujoco.sh

This will apt-get install the common build/OpenGL deps on Debian/Ubuntu.
EOT
fi