import torch
from diffusers import DDIMScheduler, DiffusionPipeline
import json
from pathlib import Path
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
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
    except Exception:
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    image_path = "/root/work/my/examples/img2/002784_input.jpg"
    json_path = "/root/work/my/examples/coord2/002784_input.json"
    output_dir = "./out"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              clip_sample=False, set_alpha_to_one=False)

    model_path = r"/root/autodl-tmp/stable-diffusion-2-1-base"
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="./pipelines/pipline2x.py",
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)

    points, labels = load_coordinates(json_path)

    if points and len(points) == len(labels):
        generator = torch.Generator(device=device).manual_seed(123)
        attn_aggregator = StableDiffusion2AttentionAggregator(pipe=pipeline, device='cuda:0')

        image = pipeline(
            prompt="",
            image=image_path,
            points=points,
            points_in_segment=labels,
            height=512,
            width=512,
            SGA=True,
            strength=0.8,
            rm_guidance_scale=9,
            sg_steps=30,
            sg_scale=0.3,
            SGA_start_step=0,
            SGA_start_layer=12,
            SGA_end_layer=32,
            num_inference_steps=50,
            generator=generator,
            guidance_scale=1,
            attn_aggregator=attn_aggregator,
        ).images[0]

        output_path = Path(output_dir) / f"{Path(image_path).stem}.png"
        image.save(output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Processing completed.")