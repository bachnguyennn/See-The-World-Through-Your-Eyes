#!/usr/bin/env bash
# Download the EyeNeRF sample capture and train on it (run inside WSL after
# setup_wsl_eyenerf.sh, with `conda activate eyenerf`).
#
# Usage:
#   bash scripts/train_sample.sh [DATA_ROOT] [EXPERIMENT_NAME]
#
# DATA_ROOT defaults to ~/eyenerf-data (inside WSL = fast disk). Outputs go to
# DATA_ROOT/outputs. Watch training at http://localhost:7007 in a Windows browser
# (WSL forwards localhost automatically).

set -euo pipefail

DATA_ROOT="${1:-$HOME/eyenerf-data}"
EXP="${2:-sample}"
SAMPLE_GDRIVE_ID="1YMHZUSifun5gA2sKHj8Jiu32dnqFDccr"   # from eyenerf/README.md

[[ "${CONDA_DEFAULT_ENV:-}" == "eyenerf" ]] || { echo "Run 'conda activate eyenerf' first." >&2; exit 1; }
mkdir -p "$DATA_ROOT"

# --- download + unpack (skipped if already there) -----------------------------
if ! find "$DATA_ROOT/sample" -name transforms_train.json -print -quit 2>/dev/null | grep -q .; then
  echo "==> Downloading sample capture from Google Drive"
  gdown --fuzzy "https://drive.google.com/file/d/$SAMPLE_GDRIVE_ID/view" -O "$DATA_ROOT/sample_download"
  mkdir -p "$DATA_ROOT/sample"
  case "$(file -b --mime-type "$DATA_ROOT/sample_download")" in
    application/zip)          unzip -q "$DATA_ROOT/sample_download" -d "$DATA_ROOT/sample" ;;
    application/x-tar|application/gzip|application/x-gzip) tar -xf "$DATA_ROOT/sample_download" -C "$DATA_ROOT/sample" ;;
    *) echo "Unknown archive type: $(file -b "$DATA_ROOT/sample_download")" >&2; exit 1 ;;
  esac
  rm "$DATA_ROOT/sample_download"
fi

# the dataset folder is whichever directory holds transforms_train.json
DATA_DIR="$(dirname "$(find "$DATA_ROOT/sample" -name transforms_train.json -print -quit)")"
[[ -f "$DATA_DIR/transforms_test.json" ]] || { echo "No transforms_test.json next to $DATA_DIR/transforms_train.json" >&2; exit 1; }
echo "==> Dataset: $DATA_DIR"

# --- train (command from eyenerf/README.md) -----------------------------------
# EyeNeRF opens ./config_hash.json (tiny-cuda-nn network config) relative to the
# working directory, so training must run from inside eyenerf/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/eyenerf"
echo "==> Training '$EXP' - open http://localhost:7007 to watch"
ns-train cornea \
  --data "$DATA_DIR" \
  --output_dir "$DATA_ROOT/outputs" \
  --experiment_name "$EXP" \
  --pipeline.model.rot-loss-mult 0.1 \
  --pipeline.model.use-texture-field True

CONFIG="$(ls -td "$DATA_ROOT/outputs/$EXP"/cornea/*/ | head -1)config.yml"
cat <<EOF

Training finished. Config: $CONFIG

Next:
  1. ns-viewer --load-config $CONFIG     -> build + export a camera path in the viewer
  2. ns-render camera-path --load-config $CONFIG --camera-path-filename <camera_path.json> \\
         --output-format images --output-path $DATA_ROOT/renders/$EXP
  3. On Windows (diffusion venv):
       python diffusion/refine_renders.py --input \\\\wsl\$\\Ubuntu-22.04$DATA_ROOT/renders/$EXP ...
EOF
