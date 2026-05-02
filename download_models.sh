#!/bin/bash
# download_models.sh

set -e  

MODEL_DIR="./models/stable-diffusion-2-1-base"
if [ -d "$MODEL_DIR" ]; then
    echo "Model already exists at $MODEL_DIR. Skipping download."
    exit 0
fi

echo "Downloading Stable Diffusion 2.1 base model from Hugging Face..."

pip install huggingface_hub -q


python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='sd-research/stable-diffusion-2-1-base', local_dir='$MODEL_DIR', local_dir_use_symlinks=False)
"

echo "Model downloaded to $MODEL_DIR"