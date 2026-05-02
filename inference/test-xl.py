import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
import json
from pathlib import Path
from src.stable_diffusion_xl_attention_aggregator import StableDiffusionxlAttentionAggregator
from PIL import Image as PILImage
import traceback

def load_coordinates(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        points = data.get("points", [])
        labels = data.get("labels", [])

        formatted_points = []
        for point in points:
            if len(point) == 2:
                formatted_points.append((point[0], point[1]))

        return formatted_points, labels
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return [], []
if __name__ == "__main__":
    # single image path
    image_path = "./examples/test1.jpg"
    json_path = "./examples/test1.json"
    output_dir = "./out"

    os.makedirs(output_dir, exist_ok=True)

    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)

    model_path = "./models/stable-diffusion-xl-base-1.0"
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="./pipelines/piplinexl.py",
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)
    pipeline.enable_attention_slicing()
    pipeline.enable_model_cpu_offload()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    attn_aggregator = StableDiffusionxlAttentionAggregator(device='cuda:0')
    points, labels = load_coordinates(json_path)
    seed = 123
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipeline(
        prompt="",
        image=image_path,
        points=points,
        points_in_segment=labels,
        height=1024,
        width=1024,
        SGA=True,
        strength=0.8,
        rm_guidance_scale=7,
        sg_steps=9,
        sg_scale=0.3,
        SGA_start_step=0,
        SGA_start_layer=34,
        SGA_end_layer=70,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=1,
        attn_aggregator=attn_aggregator,
    ).images[0]

    with PILImage.open(image_path) as orig_img:
        orig_w, orig_h = orig_img.size
    target_size = 1024
    scale = target_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    if pad_left > 0 or pad_top > 0:
        image = image.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))

    output_path = Path(output_dir) / f"{Path(image_path).stem}.png"
    image.save(output_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Processing completed.")

