import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
import json
from pathlib import Path
from src.stable_diffusion_xl_attention_aggregator import StableDiffusionxlAttentionAggregator
from PIL import Image

if __name__ == "__main__":
    # Set paths for single image processing
    image_path = "/home/zld/work/my/data/examples/img/0000b7e1500c94d7.jpg"   # Change to your image path
    json_path = "/home/zld/work/my/data/examples/coord/0000b7e1500c94d7.json" # Change to your JSON path
    output_dir = "./out"

    os.makedirs(output_dir, exist_ok=True)

    # Check if files exist
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)

    model_path = "/mnt/nvme0n1/zld/BrushNet_data/ckpt/stable-diffusion-xl-base-1.0"
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="./pipelines/piplinexl.py",
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)

    def load_coordinates(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        points = data.get("points", [])
        labels = data.get("labels", [])

        formatted_points = []
        for point in points:
            if len(point) == 2:
                formatted_points.append((point[0], point[1]))

        return formatted_points, labels

    points, labels = load_coordinates(json_path)

    seed = 123  # fixed seed, can be changed
    generator = torch.Generator(device=device).manual_seed(seed)

    attn_aggregator = StableDiffusionxlAttentionAggregator(device='cuda:0')

    image = pipeline(
        prompt="",
        image=str(image_path),
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
        attn=attn_aggregator,
    ).images[0]

    output_path = Path(output_dir) / f"{Path(image_path).stem}.png"
    image.save(output_path)
    print(f"Saved output to {output_path}")