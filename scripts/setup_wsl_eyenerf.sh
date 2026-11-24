#!/usr/bin/env bash
# Set up the EyeNeRF training environment inside WSL2 (Ubuntu 22.04).
#
# Prerequisites (on Windows, once):
#   1. Admin PowerShell:  wsl --install -d Ubuntu-22.04   then reboot.
#   2. A recent NVIDIA Windows driver (provides the GPU inside WSL; do NOT install
#      a Linux NVIDIA driver inside WSL).
#
# Usage (inside the Ubuntu shell, from the repo root):
#   bash scripts/setup_wsl_eyenerf.sh
#
# Installs: CUDA 12.1 toolkit (compiler only, for building tiny-cuda-nn /
# GroundingDINO), Miniforge, conda env "eyenerf" (Python 3.8, torch 2.2.1+cu121,
# nerfstudio 1.0.2, pins from eyenerf/environment.yml), tiny-cuda-nn, EyeNeRF,
# Segment Anything + ViT-H weights, GroundingDINO + weights, ExifTool.
# Safe to re-run: finished steps are skipped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EYENERF_DIR="$REPO_ROOT/eyenerf"
DEPS_DIR="${EYENERF_DEPS:-$HOME/eyenerf-deps}"
CONDA_DIR="$HOME/miniforge3"
ENV_NAME="eyenerf"
CUDA_VER="12-1"
CUDA_HOME_DIR="/usr/local/cuda-12.1"
TCNN_COMMIT="a77dc53ed770dd8ea6f78951d5febe175d0045e9"
GDINO_COMMIT="39b1472457b8264adc8581d354bb1d1956ec7ee7"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. sanity checks ---------------------------------------------------------
grep -qi microsoft /proc/version || die "Run this inside WSL (Ubuntu 22.04), not native Linux/Windows."
. /etc/os-release
[[ "$VERSION_ID" == "22.04" ]] || die "Ubuntu $VERSION_ID detected. Use Ubuntu 22.04: CUDA 12.1 needs gcc <= 12 (24.04 ships gcc 13)."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: update the NVIDIA driver on Windows, then 'wsl --shutdown'."
nvidia-smi --query-gpu=name --format=csv,noheader || die "GPU not visible inside WSL."
[[ -f "$EYENERF_DIR/pyproject.toml" ]] || die "eyenerf submodule is empty. Run: git submodule update --init"
case "$REPO_ROOT" in /mnt/*) echo "NOTE: repo is on a Windows drive ($REPO_ROOT). It works, but file I/O is slow;"
                             echo "      for faster training keep datasets/outputs under \$HOME (e.g. ~/eyenerf-data).";; esac
GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
echo "GPU compute capability: $GPU_ARCH"

# --- 1. system packages -------------------------------------------------------
log "System packages (sudo)"
sudo apt-get update -y
sudo apt-get install -y build-essential git wget curl unzip file ca-certificates libimage-exiftool-perl \
                        libgl1 libglib2.0-0
exiftool -ver

# --- 2. CUDA 12.1 toolkit (WSL repo: toolkit only, never the driver) ----------
if [[ ! -x "$CUDA_HOME_DIR/bin/nvcc" ]]; then
  log "CUDA $CUDA_VER toolkit"
  tmp="$(mktemp -d)"
  wget -q -O "$tmp/cuda-keyring.deb" \
    https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "$tmp/cuda-keyring.deb"
  rm -rf "$tmp"
  sudo apt-get update -y
  sudo apt-get install -y "cuda-toolkit-$CUDA_VER"
fi
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -1

# --- 3. Miniforge + conda env ---------------------------------------------------
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
  log "Miniforge"
  wget -q -O /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm /tmp/miniforge.sh
  "$CONDA_DIR/bin/conda" init bash >/dev/null
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda env '$ENV_NAME' (Python 3.8)"
  conda create -y -n "$ENV_NAME" python=3.8 pip
fi
conda activate "$ENV_NAME"

# Persist CUDA paths for every future 'conda activate eyenerf'.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<EOF
export CUDA_HOME=$CUDA_HOME_DIR
export PATH=$CUDA_HOME_DIR/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_HOME_DIR/lib64:\${LD_LIBRARY_PATH:-}
export TCNN_CUDA_ARCHITECTURES=$GPU_ARCH
EOF

# --- 4. pinned pip packages from eyenerf/environment.yml ----------------------
if ! python -c "import nerfstudio, torch; assert torch.__version__.startswith('2.2.1')" 2>/dev/null; then
  log "Pinned pip packages (torch 2.2.1+cu121, nerfstudio 1.0.2, ...) - this takes a while"
  req="$(mktemp --suffix=.txt)"
  # take only the '- pip:' section, strip the YAML list markers
  awk '/^ *- pip:/{f=1;next} f && /^ *- /{sub(/^ *- /,""); print}' "$EYENERF_DIR/environment.yml" > "$req"
  pip install --upgrade "pip<24.1"   # newer pip rejects some legacy metadata in these pins
  pip install -r "$req"
  rm "$req"
fi
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

# --- 5. tiny-cuda-nn -------------------------------------------------------------
if ! python -c "import tinycudann" 2>/dev/null; then
  log "tiny-cuda-nn (compiling for sm_$GPU_ARCH, ~10-20 min)"
  TCNN_CUDA_ARCHITECTURES="$GPU_ARCH" pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@$TCNN_COMMIT#subdirectory=bindings/torch"
fi

# --- 6. EyeNeRF itself -------------------------------------------------------------
log "EyeNeRF (editable)"
pip install --no-deps -e "$EYENERF_DIR"
pip install pyexiftool==0.5.6

# --- 7. Segment Anything + GroundingDINO (only needed for preprocessing own captures)
mkdir -p "$DEPS_DIR"
python -c "import segment_anything" 2>/dev/null || {
  log "Segment Anything"
  pip install "git+https://github.com/facebookresearch/segment-anything.git"
}
[[ -f "$DEPS_DIR/sam_vit_h_4b8939.pth" ]] || {
  log "SAM ViT-H weights (2.4 GB)"
  wget -q --show-progress -O "$DEPS_DIR/sam_vit_h_4b8939.pth" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
}
if [[ ! -d "$DEPS_DIR/GroundingDINO" ]]; then
  log "GroundingDINO @ ${GDINO_COMMIT:0:7}"
  git clone -q https://github.com/IDEA-Research/GroundingDINO.git "$DEPS_DIR/GroundingDINO"
  git -C "$DEPS_DIR/GroundingDINO" checkout -q "$GDINO_COMMIT"
fi
python -c "import groundingdino" 2>/dev/null || pip install --no-build-isolation -e "$DEPS_DIR/GroundingDINO"
[[ -f "$DEPS_DIR/GroundingDINO/weights/groundingdino_swint_ogc.pth" ]] || {
  mkdir -p "$DEPS_DIR/GroundingDINO/weights"
  wget -q --show-progress -O "$DEPS_DIR/GroundingDINO/weights/groundingdino_swint_ogc.pth" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
}

# --- 8. verify -------------------------------------------------------------------------
log "Verifying"
python - <<'PY'
import torch, tinycudann, nerfstudio, eyenerf.eyenerf_config, groundingdino, segment_anything, exiftool
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
print("tinycudann, nerfstudio, eyenerf, groundingdino, segment_anything, pyexiftool: OK")
PY
ns-train cornea --help >/dev/null && echo "ns-train cornea: OK"

cat <<EOF

Done. In a new shell:
  conda activate $ENV_NAME
  bash scripts/train_sample.sh          # download a sample capture and train

Preprocessing your own captures (from the eyenerf/ folder, it imports ellseg/ relative to cwd):
  cd eyenerf && python gen_masks_offline.py --data_path <tif_folder> \\
      --sam_path $DEPS_DIR/sam_vit_h_4b8939.pth --grounding_dino_prefix $DEPS_DIR/GroundingDINO
EOF
