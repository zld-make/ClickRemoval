#!/usr/bin/env python3
"""
ClickRemoval: Click-Driven Object Removal
Interactive demo for ACM MM Open Source Software track.

Usage:
    python app.py --model sd21 --device cuda --port 7860
"""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DDIMScheduler, DiffusionPipeline
import sys
sys.stdout.reconfigure(line_buffering=True)
# ----------------------------------------------------------------------
#  Configuration and Constants
# ----------------------------------------------------------------------
DEFAULT_MODEL = "sd15"  # 'sd15', 'sd21', 'sdxl'
DEFAULT_PIPELINE_STRENGTH = 0.8
MODEL_PATHS = {
    "sd15": "./models/stable-diffusion-v1-5",
    "sd21": "./models/stable-diffusion-2-1-base",
    "sdxl": "./models/stable-diffusion-xl-base-1.0",
}
HF_MODEL_IDS = {
    "sd15": "ledun-ai/stable-diffusion-v1-5",
    "sd21": "ledun-ai/stable-diffusion-2-1-base",
    "sdxl": "ledun-ai/stable-diffusion-xl-base-1.0",
}

MODEL_CONFIGS = {
    "sd15": {"start_layer": 7, "end_layer": 16, "sg_steps": 5, "resolution_default": 512},
    "sd21": {"start_layer": 7, "end_layer": 16, "sg_steps": 5,"resolution_default": 512},
    "sdxl": {"start_layer": 34, "end_layer": 70, "sg_steps": 9, "resolution_default": 1024},
}

# ----------------------------------------------------------------------
#  Global State for Cached Models
# ----------------------------------------------------------------------
_loaded_model_name = None
_loaded_pipeline = None
_loaded_aggregator = None
_loaded_device = None
_loaded_low_vram = False


# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
def get_font(size=20):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        font = ImageFont.load_default()
    return font


def draw_clicks_on_image(image: Image.Image,
                         positive_points: List[Tuple[int, int]],
                         negative_points: List[Tuple[int, int]]) -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    font = get_font(20)

    for idx, (x, y) in enumerate(positive_points):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#22c55e", fill="#22c55e", width=2)
        draw.text((x + 12, y - 12), f"P{idx+1}", fill="#22c55e", font=font, stroke_width=1, stroke_fill="black")

    for idx, (x, y) in enumerate(negative_points):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#ef4444", fill="#ef4444", width=2)
        draw.text((x + 12, y - 12), f"N{idx+1}", fill="#ef4444", font=font, stroke_width=1, stroke_fill="black")

    return img_copy


def save_json_safe(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_comparison(original: Image.Image, result: Image.Image) -> Image.Image:
    width, height = original.size
    comparison = Image.new("RGB", (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(result, (width, 0))
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Original", fill="white", stroke_width=1, stroke_fill="black")
    draw.text((width + 10, 10), "Result", fill="white", stroke_width=1, stroke_fill="black")
    return comparison


def zip_outputs(output_dir: Path) -> Path:
    zip_path = output_dir / "results.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in output_dir.iterdir():
            if file.name != "results.zip":
                zipf.write(file, arcname=file.name)
    return zip_path


# ----------------------------------------------------------------------
#  Model Loading (with caching)
# ----------------------------------------------------------------------
def load_model(model_name: str, device: str, low_vram: bool = False):
    global _loaded_model_name, _loaded_pipeline, _loaded_aggregator, _loaded_device, _loaded_low_vram

    if (_loaded_model_name == model_name and _loaded_device == device
            and _loaded_low_vram == low_vram and _loaded_pipeline is not None):
        return _loaded_pipeline, _loaded_aggregator

    if _loaded_pipeline is not None:
        _loaded_pipeline.to("cpu")
        del _loaded_pipeline
        if _loaded_aggregator:
            del _loaded_aggregator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    model_path = MODEL_PATHS.get(model_name)
    if model_path and not Path(model_path).exists():
        model_path = HF_MODEL_IDS.get(model_name, model_name)
        print(f"Local model not found, loading from HuggingFace: {model_path}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False)

    if model_name == "sd15":
        custom_pipe = "./pipelines/pipline1x.py"
        from src.stable_diffusion_1_attention_aggregator import StableDiffusion1AttentionAggregator
        AggregatorClass = StableDiffusion1AttentionAggregator
    elif model_name == "sd21":
        custom_pipe = "./pipelines/pipline2x.py"
        from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
        AggregatorClass = StableDiffusion2AttentionAggregator
    elif model_name == "sdxl":
        custom_pipe = "./pipelines/piplinexl.py"
        from src.stable_diffusion_xl_attention_aggregator import StableDiffusionxlAttentionAggregator
        AggregatorClass = StableDiffusionxlAttentionAggregator

    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline=custom_pipe,
        scheduler=scheduler,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    pipeline.enable_attention_slicing()
    if low_vram:
        pipeline.enable_model_cpu_offload()

    aggregator = AggregatorClass(device=device)

    _loaded_model_name = model_name
    _loaded_pipeline = pipeline
    _loaded_aggregator = aggregator
    _loaded_device = device
    _loaded_low_vram = low_vram

    return pipeline, aggregator


# ----------------------------------------------------------------------
#  Core ClickRemoval Inference (no intermediate visualizations)
# ----------------------------------------------------------------------
def run_clickremoval(
    image: Image.Image,
    positive_points: List[Tuple[int, int]],
    negative_points: List[Tuple[int, int]],
    model_name: str,
    seed: int,
    removal_guidance_scale: float,
    num_inference_steps: int,
    sg_steps: int,
    guidance_scale: float,
    resolution: int,
    low_vram: bool,
    device: str,
    progress_callback=None,
) -> Dict[str, Any]:
    if progress_callback:
        progress_callback(0, "Loading model...")

    pipeline, aggregator = load_model(model_name, device, low_vram)

    all_points = positive_points + negative_points
    points_in_segment = [True] * len(positive_points) + [False] * len(negative_points)

    if len(all_points) == 0:
        raise ValueError("At least one positive click is required.")

    if progress_callback:
        progress_callback(10, "Preprocessing image...")

    orig_w, orig_h = image.size
    target_size = resolution
    scale = target_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./outputs/demo") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path = output_dir / "temp_input.png"
    image.save(temp_image_path)

    generator = torch.Generator(device=device).manual_seed(seed)
    model_cfg = MODEL_CONFIGS[model_name]

    if progress_callback:
        progress_callback(30, "Redirecting self-attention and denoising...")

    def step_callback(step, timestep, latents):
        if progress_callback:
            progress = 30 + (step / num_inference_steps) * 50
            progress_callback(progress, f"Denoising step {step+1}/{num_inference_steps}")

    result_padded = pipeline(
        prompt="",
        image=str(temp_image_path),
        points=all_points,
        points_in_segment=points_in_segment,
        height=resolution,
        width=resolution,
        SGA=True,
        strength=DEFAULT_PIPELINE_STRENGTH,
        rm_guidance_scale=removal_guidance_scale,
        sg_steps=sg_steps,
        sg_scale=None,
        SGA_start_step=0,
        SGA_start_layer=model_cfg["start_layer"],
        SGA_end_layer=model_cfg["end_layer"],
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        attn_aggregator=aggregator,
        callback=step_callback,
        callback_steps=1,
    ).images[0]

    if progress_callback:
        progress_callback(90, "Post-processing result...")

    cropped_result = result_padded.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))
    final_result = cropped_result.resize((orig_w, orig_h), Image.LANCZOS)

    comparison = make_comparison(image, final_result)

    image.save(output_dir / "input.png")
    final_result.save(output_dir / "result.png")
    comparison.save(output_dir / "comparison.png")

    clicks_data = {
        "positive_points": positive_points,
        "negative_points": negative_points,
        "image_size": [orig_w, orig_h],
        "model": model_name,
        "seed": seed,
        "strength": DEFAULT_PIPELINE_STRENGTH,
        "removal_guidance_scale": removal_guidance_scale,
        "steps": num_inference_steps,
        "sg_steps": sg_steps,
        "resolution": resolution,
        "guidance_scale": guidance_scale,
    }
    save_json_safe(clicks_data, output_dir / "clicks.json")

    params = {
        "model_preset": model_name,
        "seed": seed,
        "strength": DEFAULT_PIPELINE_STRENGTH,
        "removal_guidance_scale": removal_guidance_scale,
        "num_inference_steps": num_inference_steps,
        "sg_steps": sg_steps,
        "guidance_scale": guidance_scale,
        "resolution": resolution,
        "low_vram_mode": low_vram,
        "device": device,
    }
    save_json_safe(params, output_dir / "params.json")

    if progress_callback:
        progress_callback(100, "Done!")

    return {
        "result": final_result,
        "comparison": comparison,
        "output_dir": output_dir,
    }


# ----------------------------------------------------------------------
#  Gradio UI Callbacks
# ----------------------------------------------------------------------
def add_click(image, evt: gr.SelectData, click_mode, positive_state, negative_state):
    if image is None:
        return image, positive_state, negative_state, "Please upload an image first."

    x, y = evt.index
    if click_mode == "Positive click":
        positive_state.append((x, y))
    else:
        negative_state.append((x, y))

    updated_img = draw_clicks_on_image(image, positive_state, negative_state)
    return updated_img, positive_state, negative_state, f"Added {click_mode} at ({x}, {y})"


def clear_last_click(positive_state, negative_state):
    if positive_state and (not negative_state or len(positive_state) >= len(negative_state)):
        removed = positive_state.pop()
        msg = f"Removed positive click at {removed}"
    elif negative_state:
        removed = negative_state.pop()
        msg = f"Removed negative click at {removed}"
    else:
        msg = "No clicks to remove."
    return positive_state, negative_state, msg


def clear_all_clicks(positive_state, negative_state):
    positive_state.clear()
    negative_state.clear()
    return positive_state, negative_state, "All clicks cleared."


def reset_interface(image, positive_state, negative_state):
    positive_state.clear()
    negative_state.clear()
    return None, None, positive_state, negative_state, "Interface reset."


def load_example(example_id):
    examples = [
        {"image": "./examples/test1.jpg", "positive": [(256, 256)], "negative": [], "model": "sd21", "removal_guidance": 6, "steps": 30, "sg_steps": 5, "resolution": 512},
        {"image": "./examples/test2.jpg", "positive": [(320, 240)], "negative": [(100, 100)], "model": "sd15", "removal_guidance": 5, "steps": 40, "sg_steps": 5, "resolution": 512},
        {"image": "./examples/test3.jpg", "positive": [(400, 300)], "negative": [], "model": "sdxl", "removal_guidance": 7, "steps": 50, "sg_steps": 9, "resolution": 1024},
    ]
    if example_id >= len(examples):
        return None, None, [], [], "sd21", 6, 30, 5, 512, "512 px (auto)", "Example not found."
    ex = examples[example_id]
    try:
        img = Image.open(ex["image"]).convert("RGB")
        img_with_clicks = draw_clicks_on_image(img, ex["positive"], ex["negative"])
        return (img, img_with_clicks, ex["positive"], ex["negative"], ex["model"],
                ex["removal_guidance"], ex["steps"], ex["sg_steps"], ex["resolution"],
                f"{ex['resolution']} px (auto)", f"Loaded example {example_id+1}.")
    except Exception as e:
        return None, None, [], [], "sd21", 6, 30, 5, 512, "512 px (auto)", f"Error: {e}"


def run_inference(image, positive_state, negative_state, model_name, seed, removal_guidance,
                  steps, sg_steps, guidance_scale, resolution, low_vram, device):
    seed = int(seed)
    steps = int(steps)
    resolution = int(resolution)
    sg_steps = int(sg_steps)
    params = make_params_json(model_name, seed, removal_guidance, steps, sg_steps, guidance_scale, resolution, low_vram, device)
    command = make_repro_command(positive_state, negative_state, model_name, seed, removal_guidance, steps, sg_steps, guidance_scale, resolution)

    if image is None:
        return None, None, None, "Upload an image before running ClickRemoval.", "No diagnostics yet.", command, params
    if len(positive_state) == 0:
        return None, None, None, "Add at least one positive click on the editing canvas.", "No diagnostics yet.", command, params

    pos = [tuple(p) for p in positive_state]
    neg = [tuple(p) for p in negative_state]

    status_text = "Starting..."
    def progress_callback(percent, msg):
        nonlocal status_text
        status_text = f"{percent}%: {msg}"

    try:
        result_dict = run_clickremoval(
            image=image, positive_points=pos, negative_points=neg,
            model_name=model_name, seed=seed, removal_guidance_scale=removal_guidance,
            num_inference_steps=steps, sg_steps=sg_steps, guidance_scale=guidance_scale,
            resolution=resolution, low_vram=low_vram, device=device,
            progress_callback=progress_callback,
        )
        zip_path = zip_outputs(result_dict["output_dir"])
        diagnostics = make_diagnostics(
            image=image,
            positive_points=pos,
            negative_points=neg,
            model_name=model_name,
            resolution=resolution,
            sg_steps=sg_steps,
            output_dir=result_dict["output_dir"],
            device=device,
        )
        return (result_dict["result"],
                result_dict["comparison"],
                str(zip_path),
                f"Completed. Outputs saved to {result_dict['output_dir']}",
                diagnostics,
                command,
                params)
    except Exception as e:
        return None, None, None, f"Error: {str(e)}", "Run failed before diagnostics were produced.", command, params


def update_canvas(image, pos, neg):
    if image is None:
        return None
    return draw_clicks_on_image(image, pos, neg)


def load_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None, None, [], [], "No image selected."

    file_path = getattr(uploaded_file, "name", uploaded_file)
    try:
        image = Image.open(file_path).convert("RGB")
        return image, image, [], [], "Image loaded. Add positive clicks to mark the object for removal."
    except Exception as e:
        return None, None, [], [], f"Failed to load image: {e}"


def reset_workspace(positive_state, negative_state, device):
    positive_state.clear()
    negative_state.clear()
    default_resolution = MODEL_CONFIGS[DEFAULT_MODEL]["resolution_default"]
    default_sg_steps = MODEL_CONFIGS[DEFAULT_MODEL]["sg_steps"]
    params = make_params_json(DEFAULT_MODEL, 42, 6, 30, default_sg_steps, 1.0, default_resolution, False, device)
    return (
        None, None, None, positive_state, negative_state,
        None, None, None,
        DEFAULT_MODEL, 42, 6, 30, default_sg_steps, 1.0, default_resolution, f"{default_resolution} px (auto)", False,
        "Workspace reset.",
        "No diagnostics yet.",
        make_repro_command([], [], DEFAULT_MODEL, 42, 6, 30, default_sg_steps, 1.0, default_resolution),
        params,
    )


def get_model_defaults(model_name: str):
    resolution = MODEL_CONFIGS[model_name]["resolution_default"]
    sg_steps = MODEL_CONFIGS[model_name]["sg_steps"]
    return resolution, f"{resolution} px (auto)", sg_steps


def make_repro_command(positive_points, negative_points, model_name, seed, removal_guidance,
                       steps, sg_steps, guidance_scale, resolution):
    def format_points(points):
        return ";".join(f"{int(x)},{int(y)}" for x, y in points) if points else "none"

    return (
        "python inference.py "
        "--image input.png "
        f"--model {model_name} "
        f"--positive-points \"{format_points(positive_points)}\" "
        f"--negative-points \"{format_points(negative_points)}\" "
        f"--seed {int(seed)} "
        f"--strength {DEFAULT_PIPELINE_STRENGTH} "
        f"--removal-guidance-scale {removal_guidance} "
        f"--steps {int(steps)} "
        f"--sg-steps {int(sg_steps)} "
        f"--guidance-scale {guidance_scale} "
        f"--resolution {int(resolution)}"
    )


def make_params_json(model_name, seed, removal_guidance, steps, sg_steps, guidance_scale, resolution, low_vram, device):
    return {
        "model_preset": model_name,
        "seed": int(seed),
        "strength": DEFAULT_PIPELINE_STRENGTH,
        "removal_guidance_scale": removal_guidance,
        "num_inference_steps": int(steps),
        "sg_steps": int(sg_steps),
        "guidance_scale": guidance_scale,
        "resolution": int(resolution),
        "low_vram_mode": low_vram,
        "device": device,
    }


def make_diagnostics(image, positive_points, negative_points, model_name, resolution, sg_steps, output_dir, device):
    width, height = image.size
    return (
        f"**Image**: {width} x {height}\n\n"
        f"**Clicks**: {len(positive_points)} positive, {len(negative_points)} negative\n\n"
        f"**Model**: {model_name} at {resolution} px\n\n"
        f"**SG steps**: {sg_steps}\n\n"
        f"**Strength**: {DEFAULT_PIPELINE_STRENGTH}\n\n"
        f"**Device**: {device}\n\n"
        f"**Output directory**: `{output_dir}`"
    )


APP_CSS = """
:root {
    --cr-blue: #2563eb;
    --cr-blue-dark: #1d4ed8;
    --cr-border: #e5e7eb;
    --cr-muted: #6b7280;
    --cr-surface: #ffffff;
    --cr-subtle: #f8fafc;
    --cr-text: #111827;
}

footer,
.footer,
#footer {
    display: none !important;
}

.gradio-container {
    max-width: none !important;
    background: #f6f7f9 !important;
    color: var(--cr-text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#clickremoval-app {
    max-width: 1480px;
    margin: 0 auto;
    padding: 24px;
}

#app-header {
    margin-bottom: 18px;
}

#app-header h1 {
    margin: 0;
    font-size: 30px;
    line-height: 1.1;
    font-weight: 720;
    letter-spacing: 0;
}

#app-header p {
    margin: 6px 0 0;
    color: var(--cr-muted);
}

#app-header .microcopy {
    margin-top: 8px;
    font-size: 13px;
    letter-spacing: 0;
    color: #8a94a3;
}

#layout-row {
    align-items: stretch;
    gap: 18px;
}

#control-panel {
    flex: 0 0 320px !important;
    max-width: 320px !important;
    min-width: 320px !important;
}

#main-workspace {
    min-width: 0;
}

.control-section {
    background: var(--cr-surface);
    border: 1px solid var(--cr-border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    padding: 14px;
    margin-bottom: 12px;
}

.control-section h3,
.panel-title h3 {
    margin: 0 0 12px;
    font-size: 12px;
    line-height: 1;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5563;
}

.main-card {
    background: var(--cr-surface);
    border: 1px solid var(--cr-border);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    padding: 16px;
}

.image-title {
    margin-bottom: 8px;
}

.image-title h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 650;
    color: #1f2937;
}

.work-image {
    border: 1px solid var(--cr-border) !important;
    border-radius: 8px !important;
    overflow: hidden;
    background: #fbfdff;
}

.work-image img {
    object-fit: contain !important;
}

.status-bar {
    margin-top: 12px;
    padding: 10px 12px;
    min-height: 40px;
    border: 1px solid var(--cr-border);
    border-radius: 8px;
    background: var(--cr-subtle);
    color: #374151;
    font-size: 13px;
}

.status-bar p {
    margin: 0;
}

.examples-panel {
    margin-top: 18px;
}

.examples-panel button {
    min-height: 44px !important;
}

button.primary,
.gr-button-primary {
    background: var(--cr-blue) !important;
    border-color: var(--cr-blue) !important;
    color: #ffffff !important;
}

button.primary:hover,
.gr-button-primary:hover {
    background: var(--cr-blue-dark) !important;
    border-color: var(--cr-blue-dark) !important;
}

button.secondary,
.gr-button-secondary {
    background: #ffffff !important;
    border-color: var(--cr-border) !important;
    color: #374151 !important;
}

.tabs button.selected,
.tab-nav button.selected {
    color: var(--cr-blue) !important;
    border-color: var(--cr-blue) !important;
}

label,
.label-wrap {
    color: #374151 !important;
    font-size: 12px !important;
}

.compact-upload {
    min-height: 42px;
}

@media (max-width: 920px) {
    #layout-row {
        flex-direction: column;
    }

    #control-panel {
        flex: 1 1 auto !important;
        max-width: none !important;
        min-width: 0 !important;
    }
}
"""


# ----------------------------------------------------------------------
#  Build Gradio Interface
# ----------------------------------------------------------------------
def build_ui(device: str):
    default_resolution = MODEL_CONFIGS[DEFAULT_MODEL]["resolution_default"]
    default_sg_steps = MODEL_CONFIGS[DEFAULT_MODEL]["sg_steps"]
    default_params = make_params_json(DEFAULT_MODEL, 42, 6, 30, default_sg_steps, 1.0, default_resolution, False, device)
    default_command = make_repro_command([], [], DEFAULT_MODEL, 42, 6, 30, default_sg_steps, 1.0, default_resolution)

    with gr.Blocks(
        title="ClickRemoval",
        css=APP_CSS,
        elem_id="clickremoval-app",
    ) as demo:
        gr.Markdown(
            """
            # ClickRemoval
            Click-driven object removal with clicks only
            <div class="microcopy">No masks &middot; No text prompts &middot; No additional training</div>
            """,
            elem_id="app-header",
        )

        with gr.Row(equal_height=False, elem_id="layout-row"):
            with gr.Column(elem_id="control-panel"):
                with gr.Group(elem_classes=["control-section"]):
                    gr.Markdown("### Input", elem_classes=["panel-title"])
                    upload_file = gr.File(
                        label="Upload image",
                        file_types=["image"],
                        elem_classes=["compact-upload"],
                    )
                    click_mode = gr.Radio(
                        ["Positive click", "Negative click"],
                        value="Positive click",
                        label="Click mode",
                        info="Positive removes. Negative protects.",
                    )
                    with gr.Row():
                        clear_last_btn = gr.Button("Clear Last", size="sm", variant="secondary")
                        clear_all_btn = gr.Button("Clear All", size="sm", variant="secondary")

                with gr.Group(elem_classes=["control-section"]):
                    gr.Markdown("### Model", elem_classes=["panel-title"])
                    model_choice = gr.Dropdown(
                        choices=[
                            ("SD 1.5 - Fast", "sd15"),
                            ("SD 2.1 - Balanced", "sd21"),
                            ("SDXL 1.0 - High Quality", "sdxl"),
                        ],
                        value=DEFAULT_MODEL,
                        label="Model preset",
                    )
                    resolution_display = gr.Textbox(
                        value=f"{default_resolution} px (auto)",
                        label="Resolution",
                        interactive=False,
                    )
                    low_vram_check = gr.Checkbox(value=False, label="Low VRAM")

                with gr.Group(elem_classes=["control-section"]):
                    gr.Markdown("### Generation", elem_classes=["panel-title"])
                    seed_slider = gr.Number(value=42, label="Seed", precision=0)
                    strength_slider = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=6,
                        step=0.5,
                        label="Removal guidance",
                    )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Repair steps",
                    )
                    sg_steps_slider = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=default_sg_steps,
                        step=1,
                        label="Strongly guided steps",
                    )
                    with gr.Accordion("Advanced Settings", open=False):
                        guidance_cfg = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=1.0,
                            step=0.1,
                            label="Guidance scale",
                        )

                with gr.Group(elem_classes=["control-section"]):
                    gr.Markdown("### Action", elem_classes=["panel-title"])
                    run_btn = gr.Button("Run ClickRemoval", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")

            with gr.Column(elem_id="main-workspace"):
                with gr.Group(elem_classes=["main-card"]):
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            gr.Markdown("### Original with Clicks / Editing Canvas", elem_classes=["image-title"])
                            canvas = gr.Image(
                                type="pil",
                                show_label=False,
                                interactive=True,
                                height=520,
                                elem_classes=["work-image"],
                            )
                        with gr.Column():
                            gr.Markdown("### Result", elem_classes=["image-title"])
                            result_img = gr.Image(
                                type="pil",
                                show_label=False,
                                interactive=False,
                                height=520,
                                elem_classes=["work-image"],
                            )

                    status_bar = gr.Markdown("Ready.", elem_classes=["status-bar"])

                    with gr.Tabs():
                        with gr.TabItem("Comparison"):
                            comparison_img = gr.Image(
                                label="Original vs Result",
                                type="pil",
                                height=420,
                                elem_classes=["work-image"],
                            )
                            download_zip = gr.File(label="Download outputs")
                        with gr.TabItem("Diagnostics"):
                            diagnostics_view = gr.Markdown("No diagnostics yet.")
                        with gr.TabItem("Reproducibility"):
                            cmd_example = gr.Code(language="shell", value=default_command, label="Command")
                            params_json = gr.JSON(value=default_params, label="Parameters")

                with gr.Accordion("Examples", open=False, elem_classes=["examples-panel"]):
                    gr.Markdown("Load a small preset example into the editing canvas.")
                    with gr.Row():
                        example_btns = [gr.Button(f"Example {i+1}", size="sm", variant="secondary") for i in range(3)]
                    example_status = gr.Markdown("Examples are optional.")

        image_state = gr.State(None)
        positive_state = gr.State([])
        negative_state = gr.State([])
        resolution_state = gr.Number(value=default_resolution, visible=False)
        device_state = gr.State(device)

        upload_file.change(
            load_uploaded_image,
            inputs=upload_file,
            outputs=[image_state, canvas, positive_state, negative_state, status_bar],
            queue=False,
        )

        canvas.select(
            add_click,
            inputs=[image_state, click_mode, positive_state, negative_state],
            outputs=[canvas, positive_state, negative_state, status_bar],
            queue=False,
        )

        clear_last_btn.click(
            clear_last_click,
            inputs=[positive_state, negative_state],
            outputs=[positive_state, negative_state, status_bar],
            queue=False,
        ).then(update_canvas, inputs=[image_state, positive_state, negative_state], outputs=canvas)

        clear_all_btn.click(
            clear_all_clicks,
            inputs=[positive_state, negative_state],
            outputs=[positive_state, negative_state, status_bar],
            queue=False,
        ).then(update_canvas, inputs=[image_state, positive_state, negative_state], outputs=canvas)

        reset_btn.click(
            reset_workspace,
            inputs=[positive_state, negative_state, device_state],
            outputs=[
                upload_file, image_state, canvas, positive_state, negative_state,
                result_img, comparison_img, download_zip,
                model_choice, seed_slider, strength_slider, steps_slider,
                sg_steps_slider, guidance_cfg, resolution_state, resolution_display, low_vram_check,
                status_bar, diagnostics_view, cmd_example, params_json,
            ],
            queue=False,
        )

        model_choice.change(
            get_model_defaults,
            inputs=model_choice,
            outputs=[resolution_state, resolution_display, sg_steps_slider],
            queue=False,
        )

        for idx, btn in enumerate(example_btns):
            btn.click(
                load_example,
                inputs=[gr.Number(value=idx, visible=False)],
                outputs=[
                    image_state, canvas, positive_state, negative_state,
                    model_choice, strength_slider, steps_slider,
                    sg_steps_slider, resolution_state, resolution_display, example_status,
                ],
                queue=False,
            )

        run_btn.click(
            run_inference,
            inputs=[
                image_state, positive_state, negative_state,
                model_choice, seed_slider, strength_slider,
                steps_slider, sg_steps_slider, guidance_cfg, resolution_state,
                low_vram_check, device_state,
            ],
            outputs=[
                result_img, comparison_img, download_zip,
                status_bar, diagnostics_view, cmd_example, params_json,
            ],
            queue=True,
        )

        def update_repro(model, seed, removal_guidance, steps, sg_steps, guidance, resolution, low_vram):
            return (
                make_repro_command([], [], model, seed, removal_guidance, steps, sg_steps, guidance, resolution),
                make_params_json(model, seed, removal_guidance, steps, sg_steps, guidance, resolution, low_vram, device),
            )

        for comp in [model_choice, seed_slider, strength_slider, steps_slider, sg_steps_slider, guidance_cfg, resolution_state, low_vram_check]:
            comp.change(
                update_repro,
                inputs=[model_choice, seed_slider, strength_slider, steps_slider, sg_steps_slider, guidance_cfg, resolution_state, low_vram_check],
                outputs=[cmd_example, params_json],
                queue=False,
            )

    return demo


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ClickRemoval Interactive Demo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["sd15","sd21","sdxl"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_ui(device=args.device)
    demo.launch(server_port=args.port, server_name="0.0.0.0", share=args.share, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
