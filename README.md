<h1 align="center">ClickRemoval: An Interactive Open-Source Tool for Object Removal in Diffusion Models</h1>

ClickRemoval is a **fully open‑source, training‑free** object removal tool built on pretrained latent diffusion models (Stable Diffusion).

<p align="center">
  <a href="https://arxiv.org/abs/2605.14461"><img src="https://img.shields.io/badge/ClickRemoval-arXiv-B31B1B?logo=arxiv&logoColor=red&labelColor=666666" alt="arXiv"></a>&nbsp;
  <a href="https://huggingface.co/ledun-ai"><img src="https://img.shields.io/badge/Hugging Face-Models-FF9A00?logo=huggingface&logoColor=yellow" alt="Hugging Face Models"></a>&nbsp;
  <img src="https://img.shields.io/badge/ModelScope%20Demo-coming%20soon-lightgrey?logo=modelscope&logoColor=white" alt="ModelScope Demo (coming soon)">
  <a href="https://github.com/zld-make/ClickRemoval/"><img src="https://img.shields.io/github/stars/zld-make/ClickRemoval?style=social" alt="GitHub stars"></a>&nbsp;
</p>

<p align="center">
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene1_unified.gif?raw=true" width="30%" style="vertical-align: top;"> 
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene3_unified.gif?raw=true" width="30%" style="vertical-align: top;"> 
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene4_unified.gif?raw=true" width="30%" style="vertical-align: top;">
</p>

<p align="center">
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene2_unified.gif?raw=true" width="23%" style="vertical-align: top;"> 
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene5_unified.gif?raw=true" width="23%" style="vertical-align: top;"> 
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene6_unified.gif?raw=true" width="23%" style="vertical-align: top;">
  <img src="https://github.com/zld-make/ClickRemoval-Images/blob/main/scene7_unified.gif?raw=true" width="23%" style="vertical-align: top;">
</p>

## Qualitative Comparison
The figure below compares ClickRemoval with several baseline methods (e.g., LaMa, SD-Inpaint, etc.) on object removal tasks.

<div align="center">
  <img src="assets/images/comparison.png" alt="Comparison of various models" width="95%">
  <br>
  <em>Figure: Visual comparison of different models. ClickRemoval removes target objects more thoroughly and restores backgrounds more naturally.</em>
</div>

## Key Features

- **Easy deployment** – ClickRemoval supports SD1.5, SD2.1, SDXL, and compatible fine-tuned Stable Diffusion backbones without additional training.
- **Mask-free and prompt-free interaction** – Users remove objects by clicking on the image, without drawing masks or writing text descriptions.
- **Positive/negative click refinement** – Positive clicks specify the object or region to be removed. Negative clicks specify regions that should be kept unchanged. This is useful when the target object is close to other objects, partially occluded, or visually similar to surrounding regions.
- **Interactive Gradio demo** – The released demo allows users to upload an image, place clicks, choose a backbone, adjust inference options, and obtain the restored image directly.
- **Complete open-source package** – The repository includes source code, Docker configuration, model download scripts, example images, documentation, and evaluation utilities.

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
If the container name already exists, remove it first
```bash
docker rm -f clickremoval
```
### Low VRAM Option

In the Gradio interface, the **Low VRAM** option is disabled by default.  
Keeping it disabled usually provides faster inference.
If you run the SDXL version and encounter CUDA out-of-memory errors, please enable **Low VRAM** in the interface. This option reduces GPU memory usage at the cost of slower inference.

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

## Hardware Requirements
For GPU inference, we strongly recommend using an NVIDIA GPU with **at least 24GB VRAM**, such as an RTX 3090 or RTX 4090. This is especially important when running the SDXL backbone at 1024 resolution.

Lower-VRAM GPUs may work for SD1.5 or SD2.1, but SDXL may require enabling the **Low VRAM** option in the Gradio interface. If CUDA out-of-memory errors occur, please enable **Low VRAM** or use the SD1.5 preset for a faster and lighter test.

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

