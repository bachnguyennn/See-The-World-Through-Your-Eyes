#!/usr/bin/env bash
# Download the EyeNeRF sample captures, train on one, and (if the capture ships
# camera paths) render it. Run inside WSL after setup_wsl_eyenerf.sh, with
# `conda activate eyenerf`.
#
# Usage:
#   CAPTURE=capture_3 bash scripts/train_sample.sh [DATA_ROOT] [EXPERIMENT_NAME]
#   METHOD=cornea-diffusion PROMPT="two plush toys on a table" bash scripts/train_sample.sh
#
# METHOD     cornea (default, EyeNeRF as published) or cornea-diffusion (adds
#            diffusion-refined virtual views as extra supervision; needs the
#            diffusion env from the README). PROMPT describes the scene for it.
#
# CAPTURE    one of capture_1..capture_6 (default capture_3: the only one with
#            the authors' camera paths, so it renders without manual setup).
# DATA_ROOT  defaults to ~/eyenerf-data (inside WSL = fast disk).
# Outputs:   DATA_ROOT/outputs/<EXP>/..., renders in DATA_ROOT/renders/<EXP>/<path>/.
# Watch training at http://localhost:7007 in a Windows browser.

set -euo pipefail

DATA_ROOT="${1:-$HOME/eyenerf-data}"
CAPTURE="${CAPTURE:-capture_3}"
METHOD="${METHOD:-cornea}"
EXP="${2:-$CAPTURE$([[ "${METHOD:-cornea}" == cornea ]] || echo "-$METHOD")}"
RENDER_DOWNSCALE="${RENDER_DOWNSCALE:-2}"   # authors' paths are 1920x1080; 2 -> 960x540
SAMPLE_GDRIVE_ID="1YMHZUSifun5gA2sKHj8Jiu32dnqFDccr"   # from eyenerf/README.md

[[ "${CONDA_DEFAULT_ENV:-}" == "eyenerf" ]] || { echo "Run 'conda activate eyenerf' first." >&2; exit 1; }
mkdir -p "$DATA_ROOT"

# --- download + unpack (skipped if already there) -----------------------------
if ! find "$DATA_ROOT/sample" -name transforms_train.json -print -quit 2>/dev/null | grep -q .; then
  echo "==> Downloading sample captures from Google Drive (~6 GB)"
  gdown --fuzzy "https://drive.google.com/file/d/$SAMPLE_GDRIVE_ID/view" -O "$DATA_ROOT/sample_download"
  mkdir -p "$DATA_ROOT/sample"
  case "$(file -b --mime-type "$DATA_ROOT/sample_download")" in
    application/zip)          unzip -q "$DATA_ROOT/sample_download" -d "$DATA_ROOT/sample" ;;
    application/x-tar|application/gzip|application/x-gzip) tar -xf "$DATA_ROOT/sample_download" -C "$DATA_ROOT/sample" ;;
    *) echo "Unknown archive type: $(file -b "$DATA_ROOT/sample_download")" >&2; exit 1 ;;
  esac
  rm "$DATA_ROOT/sample_download"
fi

DATA_DIR="$(dirname "$(find "$DATA_ROOT/sample" -path "*/$CAPTURE/transforms_train.json" -print -quit)")"
[[ -f "$DATA_DIR/transforms_test.json" ]] || {
  echo "Capture '$CAPTURE' not found. Available:" >&2
  find "$DATA_ROOT/sample" -name transforms_train.json -exec dirname {} \; | xargs -n1 basename >&2
  exit 1
}
echo "==> Dataset: $DATA_DIR"

# --- train (command from eyenerf/README.md) -----------------------------------
# EyeNeRF opens ./config_hash.json (tiny-cuda-nn network config) relative to the
# working directory, so training must run from inside eyenerf/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/eyenerf"
echo "==> Training '$EXP' - open http://localhost:7007 to watch"
extra=()
if [[ "$METHOD" == cornea-diffusion && -n "${PROMPT:-}" ]]; then
  extra+=(--pipeline.guidance-prompt "$PROMPT")
fi
ns-train "$METHOD" \
  --data "$DATA_DIR" \
  --output_dir "$DATA_ROOT/outputs" \
  --experiment_name "$EXP" \
  --pipeline.model.rot-loss-mult 0.1 \
  --pipeline.model.use-texture-field True \
  --viewer.quit-on-train-completion True \
  "${extra[@]}"

CONFIG="$(ls -td "$DATA_ROOT/outputs/$EXP/$METHOD"/*/ | head -1)config.yml"
echo "==> Training finished. Config: $CONFIG"

# --- render the capture's own camera paths, if it has any -----------------------
shopt -s nullglob
paths=("$DATA_DIR"/camera_paths/*.json)
for p in "${paths[@]}"; do
  out="$DATA_ROOT/renders/$EXP/$(basename "$p" .json)"
  echo "==> Rendering $(basename "$p") -> $out"
  ns-render camera-path --load-config "$CONFIG" --camera-path-filename "$p" \
    --downscale-factor "$RENDER_DOWNSCALE" --output-format images --output-path "$out"
done

cat <<EOF

Done. Config: $CONFIG
$( ((${#paths[@]})) && echo "Renders:  $DATA_ROOT/renders/$EXP/" || echo "No camera paths in this capture: make one with 'ns-viewer --load-config $CONFIG' and render it with ns-render camera-path.")
Explore:  ns-viewer --load-config $CONFIG    (http://localhost:7007)
From Windows the renders are at  \\\\wsl.localhost\\Ubuntu-22.04$DATA_ROOT/renders/$EXP
EOF
