# ClickRemoval: An Interactive Open-Source Tool for Object Removal in Diffusion Models

ClickRemoval is a **fully open‑source, training‑free** object removal tool built on pretrained latent diffusion models (Stable Diffusion).

## docker

### docker build

docker build -f Dockerfile.cudnn -t clickremoval:cudnn .

## environment



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

### Option 1: Docker Installation (Recommended)


## Model Architecture

The figure below illustrates the overall architecture of ClickRemoval, including the Attention Activation and Suppression (AAS) module and the Self-Attention Redirection Guidance (SARG) mechanism.

<div align="center">
  <img src="assets/images/framework.png" alt="ClickRemoval Architecture" width="80%">
  <br>
  <em>Figure: Overview of ClickRemoval. M2N2 converts user clicks into semantic maps, SGAR and SGAS redirect self-attention during denoising, and ARG blends the original and modulated predictions to control removal strength.</em>
</div>

## Supported Backbones

| Model | Steps | Use Case | Download |
|-------|-------|----------|----------|
| SD1.5  | 25    | Lightweight, real‑time, resource‑constrained devices | [⬇️ SD1.5](https://huggingface.co/ledun-ai/stable-diffusion-v1-5) |
| SD2.1  | 50    | Balanced quality and speed | [⬇️ SD2.1](https://huggingface.co/ledun-ai/stable-diffusion-2-1-base) |
| SDXL   | 50    | High‑quality removal for production use | [⬇️ SDXL](https://huggingface.co/ledun-ai/stable-diffusion-xl-base-1.0) |

All variants are fully compatible with community fine‑tuned models (e.g. anime, photorealistic).

## What's New

**2026-04-03** – Full code refresh, updated requirements, GitHub sync → project fully runnable.  
*More updates coming soon...*

> 🚀 The tool is fully open‑source under the Apache‑2.0 license.  
> 🔗 Repository: [https://github.com/zld-make/ClickRemoval](https://github.com/zld-make/ClickRemoval)  
> 🐳 Docker image and live demo are also available.
