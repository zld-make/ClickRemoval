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

## Run

### Build the Docker image
```bash
docker build -f Dockerfile.cudnn -t clickremoval:cudnn .
```

### Download the default SD1.5 model
You can replace sd15 with sd21, sdxl or all.
```bash
bash download_models.sh sd15
```

### Run the Gradio Demo
```bash
mkdir -p models hf_cache outputs

docker run --gpus all \
  -p 7860:7860 \
  --name clickremoval_test \
  -v "$(pwd)/models:/workspace/models" \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  clickremoval:cudnn
```

## Running Locally Without Docker
### environment
conda environment
```bash
conda create -n clickremoval python=3.12 -y
conda activate clickremoval
```
dependencies
```bash
pip install -r requirements.txt
```
Download models, You can replace sd15 with sd21, sdxl or all.
```bash
bash download_models.sh sd15
```

## Gradio Demo
```bash
python app.py --model sd15 --device cuda --port 7860
```

## Model Architecture

The figure below illustrates the overall architecture of ClickRemoval, including the Attention Activation and Suppression (AAS) module and the Self-Attention Redirection Guidance (SARG) mechanism.

<div align="center">
  <img src="assets/images/framework.png" alt="ClickRemoval Architecture" width="80%">
  <br>
  <em>Figure: Overview of ClickRemoval. M2N2 converts user clicks into semantic maps, SGAR and SGAS redirect self-attention during denoising, and ARG blends the original and modulated predictions to control removal strength.</em>
</div>

## Supported Backbones

| Model | Preset | Local Directory | Hugging Face Repository | Resolution | Recommended Use |
|-------|--------|-----------------|--------------------------|------------|-----------------|
| SD1.5 | `sd15` | `models/stable-diffusion-v1-5` | [SD1.5](https://huggingface.co/ledun-ai/stable-diffusion-v1-5) | 512 | Lightweight, fast demo, resource-constrained devices |
| SD2.1 | `sd21` | `models/stable-diffusion-2-1-base` | [SD2.1](https://huggingface.co/ledun-ai/stable-diffusion-2-1-base) | 512 | Balanced quality and speed |
| SDXL | `sdxl` | `models/stable-diffusion-xl-base-1.0` | [SDXL](https://huggingface.co/ledun-ai/stable-diffusion-xl-base-1.0) | 1024 | High-quality removal and stronger visual restoration |

> ⚠️ Note: The SD2.1 download uses `sd-research/stable-diffusion-2-1-base` as an alternative mirror because the original `stabilityai/stable-diffusion-2-1-base` repository may be unavailable or deprecated in some environments. For strict reproducibility, users may manually place compatible Diffusers-format SD2.1 weights under `models/stable-diffusion-2-1-base`.

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

