import torch
import gradio as gr
from PIL import Image, ImageDraw
from pathlib import Path
import sys
import traceback

sys.path.append(str(Path(__file__).parent))
from diffusers import DDIMScheduler, DiffusionPipeline
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator

dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)

model_path = "/root/autodl-tmp/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    custom_pipeline="./pipelines/pipline2x.py",
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)

pipeline.enable_attention_slicing()
if device.type == "cuda":
    pipeline.enable_model_cpu_offload()

attn_aggregator = StableDiffusion2AttentionAggregator(pipe=pipeline, device=device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Model loaded.")


def resize_to_512(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h == 512:
        return image
    scale = 512 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - 512) // 2
    top = (new_h - 512) // 2
    return image.crop((left, top, left + 512, top + 512))


def draw_points(image: Image.Image, remove_points, keep_points):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y) in remove_points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='green', outline='white')
    for (x, y) in keep_points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline='white')
    return img


# ---------- 全局变量存储状态 ----------
current_image = None
remove_points = []  # green (objects to remove)
keep_points = []  # red (areas to keep)
current_mode = "Remove Mode"  # "Remove Mode" or "Keep Mode"


def upload_image(img):
    global current_image, remove_points, keep_points, current_mode
    if img is None:
        current_image = None
        return None, "No image", None
    if img.size == (512, 512):
        current_image = img
        remove_points = []
        keep_points = []
        current_mode = "Remove Mode"
        return gr.update(), "Mode: Remove Mode", None
    img_512 = resize_to_512(img)
    current_image = img_512
    remove_points = []
    keep_points = []
    current_mode = "Remove Mode"
    # 显示原始图片（不带点）
    return img_512, "Mode: Remove Mode", None


def handle_click(evt: gr.SelectData):
    global current_image, remove_points, keep_points, current_mode
    if current_image is None:
        return current_image, "Please upload image first"
    x, y = evt.index[0], evt.index[1]
    if current_mode == "Remove Mode":
        remove_points.append((x, y))
        action = "Remove (green)"
    else:
        keep_points.append((x, y))
        action = "Keep (red)"
    img_with_pts = draw_points(current_image, remove_points, keep_points)
    print(f"Click: mode={current_mode}, remove={remove_points}, keep={keep_points}")
    return img_with_pts, f"{action} added at ({x},{y})"


def reset_points():
    global remove_points, keep_points
    remove_points = []
    keep_points = []
    if current_image is None:
        return None, "No image"
    img_with_pts = draw_points(current_image, remove_points, keep_points)
    return img_with_pts, "Points reset"


def switch_mode():
    global current_mode
    current_mode = "Keep Mode" if current_mode == "Remove Mode" else "Remove Mode"
    return f"Mode: {current_mode}"


def remove_objects():
    global remove_points, keep_points, current_image
    print("=== remove_objects called ===")
    print(f"remove_points: {remove_points}, keep_points: {keep_points}")
    if current_image is None:
        return None, "Please upload image first"
    if not remove_points:
        return None, "Please add at least one remove point (green). Switch to Remove mode and left-click on objects to remove."
    points = remove_points + keep_points
    labels = [1] * len(remove_points) + [0] * len(keep_points)
    try:
        print(f"Running removal: {len(remove_points)} remove points, {len(keep_points)} keep points")
        result = pipeline(
            prompt="",
            image=current_image,
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
            generator=torch.Generator(device=device).manual_seed(123),
            guidance_scale=1,
            attn_aggregator=attn_aggregator,
        ).images[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result, "Removal completed successfully"
    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# ---------- Gradio 界面 ----------
with gr.Blocks(title="ClickRemoval SD2.1 (Left-click with Mode Switch)") as demo:
    gr.Markdown("""
    # 🖱️ ClickRemoval for Stable Diffusion 2.1
    - **Left click** to add a point according to current mode.
    - **Mode button** switches between **Remove Mode** (green points) and **Keep Mode** (red points).
    - At least one **remove point (green)** is required before clicking **Start removal**.
    """)

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Click on image (Left-click only)", interactive=True, height=512,
                                 width=512)
            mode_btn = gr.Button("Mode: Remove Mode", variant="secondary")
            reset_btn = gr.Button("🗑️ Reset all points", variant="secondary")
            remove_btn = gr.Button("✨ Start removal", variant="primary")
            info = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            output_img = gr.Image(type="pil", label="Result", height=512, width=512)

    # 事件绑定
    input_img.change(upload_image, [input_img], [input_img, info, output_img])
    input_img.select(handle_click, None, [input_img, info])  # 注意：handle_click 只接收 evt，不接收输入组件
    mode_btn.click(switch_mode, None, [mode_btn])
    reset_btn.click(reset_points, None, [input_img, info])
    remove_btn.click(remove_objects, None, [output_img, info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)