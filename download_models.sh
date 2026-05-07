#!/bin/bash
set -e

MODEL_ROOT="${MODEL_ROOT:-./models}"
mkdir -p "$MODEL_ROOT"

pip install -q huggingface_hub

download_model () {
  local repo_id="$1"
  local local_dir="$2"

  if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
    echo "Model already exists at $local_dir. Skipping."
  else
    echo "Downloading $repo_id to $local_dir ..."
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$repo_id",
    local_dir="$local_dir",
    local_dir_use_symlinks=False
)
PY
  fi
}

case "${1:-sd15}" in
  sd15)
    download_model "ledun-ai/stable-diffusion-v1-5" "$MODEL_ROOT/stable-diffusion-v1-5"
    ;;
  sd21)
    download_model "ledun-ai/stable-diffusion-2-1-base" "$MODEL_ROOT/stable-diffusion-2-1-base"
    ;;
  sdxl)
    download_model "ledun-ai/stable-diffusion-xl-base-1.0" "$MODEL_ROOT/stable-diffusion-xl-base-1.0"
    ;;
  all)
    download_model "ledun-ai/stable-diffusion-v1-5" "$MODEL_ROOT/stable-diffusion-v1-5"
    download_model "ledun-ai/stable-diffusion-2-1-base" "$MODEL_ROOT/stable-diffusion-2-1-base"
    download_model "ledun-ai/stable-diffusion-xl-base-1.0" "$MODEL_ROOT/stable-diffusion-xl-base-1.0"
    ;;
  *)
    echo "Usage: bash download_models.sh [sd15|sd21|sdxl|all]"
    exit 1
    ;;
esac

echo "Done."