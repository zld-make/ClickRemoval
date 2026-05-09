<h1 align="center">ClickRemoval: An Interactive Open-Source Tool for Object Removal in Diffusion Models</h1>

ClickRemoval is a **fully open‑source, training‑free** object removal tool built on pretrained latent diffusion models (Stable Diffusion).

## Qualitative Comparison
The figure below compares ClickRemoval with several baseline methods (e.g., LaMa, SD-Inpaint, etc.) on object removal tasks.

<div align="center">
  <img src="assets/images/comparison.png" alt="Comparison of various models" width="95%">
  <br>
  <em>Figure: Visual comparison of different models. ClickRemoval removes target objects more thoroughly and restores backgrounds more naturally.</em>
</div>

## Key Features

- **Plug‑and‑play** – Works with any Stable Diffusion model that contains self‑attention layers (SD1.5, SD2.1, SDXL, and their fine‑tuned variants).
- **Click‑only interaction** – No masks, no text prompts, no training. Supports positive/negative clicks for higher precision.
- **Innovative attention modulation** – SGAR & SGAS unify localisation and inpainting in a single forward pass, avoiding error accumulation of multi‑stage systems.

## Interaction and Method Overview
ClickRemoval supports progressive click-based refinement while using self-attention modulation to guide object removal and background restoration.
<div align="center">
  <img src="assets/images/coord.png" alt="Progressive click interaction" width="360">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/images/framework.png" alt="ClickRemoval architecture" width="440">
  <br>
  <em>Left: Progressive click interaction across representative object removal scenarios. Right: Overall framework of ClickRemoval.</em>
</div>

## Quick Start

### Build the Docker image

```bash
docker build -f Dockerfile.cudnn -t clickremoval:cudnn .
```
### Download model weights inside Docker
By default, we recommend starting with SD1.5 for the fastest reviewer check.
```bash
mkdir -p models outputs

python -m pip install -U huggingface_hub
bash download_models.sh sd15
```
You can replace sd15 with sd21, sdxl, or all.
```bash
bash download_models.sh sd21
bash download_models.sh sdxl
bash download_models.sh all
```
## Model Paths

ClickRemoval first looks for model weights under `./models`. When using Docker, the local `./models` directory is mounted into the container as `/workspace/models`.

Host-side paths:
```text
models/
├── stable-diffusion-v1-5/
├── stable-diffusion-2-1-base/
└── stable-diffusion-xl-base-1.0/
```
Container-side paths:
```text
/workspace/models/stable-diffusion-v1-5
/workspace/models/stable-diffusion-2-1-base
/workspace/models/stable-diffusion-xl-base-1.0
```

### Run the Gradio Demo
```bash
docker run --gpus all \
  -p 7860:7860 \
  --name clickremoval \
  -v "$(pwd)/models:/workspace/models" \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  clickremoval:cudnn
```
Then open
```markdown
http://localhost:7860
```
### Low VRAM Option
```markdown
In the Gradio interface, the **Low VRAM** option is disabled by default.  
Keeping it disabled usually provides faster inference.
If you run the SDXL version and encounter CUDA out-of-memory errors, please enable **Low VRAM** in the interface. This option reduces GPU memory usage at the cost of slower inference.
```
If the container name already exists, remove it first
```bash
docker rm -f clickremoval
```
## Command-line Inference
Besides the interactive Gradio demo, ClickRemoval also provides command-line inference scripts under `inference/` for reproducible testing and batch-style usage.
```text
inference/
├── inference_sd15.py
├── inference_sd21.py
└── inference_sdxl.py
```

## Run Without Docker
### environment
```bash
conda create -n clickremoval python=3.10 -y
conda activate clickremoval
```
### dependencies
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Download models
```bash
chmod +x download_models.sh
bash download_models.sh sd15
```
### Run Gradio Demo
```bash
python app.py --model sd15 --device cuda --port 7860
```
Then open
```bash
http://localhost:7860
```
## Supported Backbones

| Model | Preset | Local Directory | Hugging Face Repository | Resolution | Recommended Use |
|-------|--------|-----------------|--------------------------|------------|-----------------|
| SD1.5 | `sd15` | `models/stable-diffusion-v1-5` | [⬇️ SD1.5](https://huggingface.co/ledun-ai/stable-diffusion-v1-5) | 512 | Lightweight, fast demo, resource-constrained devices |
| SD2.1 | `sd21` | `models/stable-diffusion-2-1-base` | [⬇️ SD2.1](https://huggingface.co/ledun-ai/stable-diffusion-2-1-base) | 512 | Balanced quality and speed |
| SDXL | `sdxl` | `models/stable-diffusion-xl-base-1.0` | [⬇️ SDXL](https://huggingface.co/ledun-ai/stable-diffusion-xl-base-1.0) | 1024 | High-quality removal and stronger visual restoration |

> ⚠️ Note: For reproducible deployment, ClickRemoval uses the author-maintained `ledun-ai` Hugging Face repositories as the default download sources. These repositories mirror compatible Diffusers-format Stable Diffusion backbones used by ClickRemoval and are maintained to provide stable access during review and future use. Users may also manually place their own compatible Diffusers-format weights under the corresponding directories in `./models`.

> ✅ For the fastest reviewer check, we recommend starting with `sd15`.  
> 🌟 For the best visual quality, we recommend using `sdxl`.

---

## What's New

**2026-04-03** – Full code refresh, updated requirements, GitHub sync → project fully runnable.

**2026-04-23** – Added Dockerfile with cuDNN support and model download shell script; fixed multiple bugs and vulnerabilities in inference code.

**2026-05-02** – Added interactive Gradio demo (`app.py`) and a complete Jupyter Notebook tutorial (`ClickRemoval_Test_Tutorial.ipynb`).

**2026-05-03** – Updated Docker deployment, model mounting strategy, Hugging Face model paths, and SD1.5/SD2.1/SDXL support.

*More updates coming soon...*

---

