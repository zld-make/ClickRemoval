import torch
import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import sys

# ================= 导入你的自定义模块 =================
# 确保路径正确，如果项目结构不同请修改
sys.path.append(str(Path(__file__).parent))
from pipelines.mypiplinexl import MyPipeline   # 替换为你的实际 pipeline 类名
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
from diffusers import DDIMScheduler, DiffusionPipeline

# ================= 全局加载模型（只加载一次） =================
print("正在加载模型...")
dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)

# 模型路径（请改为你的实际路径）
model_path = "/mnt/nvme0n1/zld/BrushNet_data/ckpt/stable-diffusion-xl-base-1.0"

# 加载 pipeline（注意 custom_pipeline 路径）
pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    custom_pipeline="./pipelines/mypiplinexl.py",
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)

# 内存优化
pipeline.enable_attention_slicing()
pipeline.enable_model_cpu_offload()

# 注意力聚合器
attn_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')

# 预清理显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("模型加载完成！")

# ================= 辅助函数 =================
def resize_to_1024(image: Image.Image) -> Image.Image:
    """将图片缩放并居中裁剪/填充到 1024x1024（保持比例，无黑边）"""
    w, h = image.size
    if w == h == 1024:
        return image
    # 缩放使短边为1024
    scale = 1024 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # 中心裁剪
    left = (new_w - 1024) // 2
    top = (new_h - 1024) // 2
    image = image.crop((left, top, left + 1024, top + 1024))
    return image

def draw_points_on_image(image: Image.Image, points, labels):
    """在图片上绘制标记点：红色（正点击=移除），蓝色（负点击=保留）"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y), label in zip(points, labels):
        color = 'red' if label == 1 else 'blue'
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline='white')
    return img

# ================= 处理函数（与 Gradio 交互） =================
def upload_image(img):
    """用户上传图片时：resize 到 1024x1024 并显示，同时清空之前的状态"""
    if img is None:
        return None, [], [], gr.update(visible=True), None
    img_1024 = resize_to_1024(img)
    # 返回：显示的图片、清空的正点列表、负点列表、模式重置、结果清空
    return img_1024, [], [], "当前模式：正点击（移除）", None

def handle_click(image, evt: gr.SelectData, positive_points, negative_points, current_mode):
    """
    用户点击图片时的回调
    image: 当前显示的图片（PIL，已 resize 到 1024x1024）
    evt: 点击事件，包含坐标
    positive_points, negative_points: 状态列表
    current_mode: 当前模式字符串（用于判断）
    """
    if image is None:
        return image, positive_points, negative_points, "请先上传图片", None
    x, y = evt.index[0], evt.index[1]
    # 根据模式添加到对应列表
    if current_mode == "正点击（移除）":
        positive_points.append((x, y))
    else:
        negative_points.append((x, y))
    # 合并所有点及其标签用于绘图
    all_points = positive_points + negative_points
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)
    img_with_points = draw_points_on_image(image, all_points, all_labels)
    # 返回更新后的图片、更新后的点列表、状态提示
    return img_with_points, positive_points, negative_points, f"已添加点 ({x},{y})"

def reset_points(image):
    """重置所有点击点"""
    if image is None:
        return image, [], [], "当前模式：正点击（移除）", None
    return image, [], [], "当前模式：正点击（移除）", None

def switch_mode(mode):
    """切换点击模式"""
    return mode  # 直接返回新模式字符串

def remove_objects(image, positive_points, negative_points):
    """
    执行移除操作
    """
    if image is None:
        return None, "请先上传图片"
    if not positive_points and not negative_points:
        return None, "请至少点击一个点"
    # 构造 points 和 labels
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)

    # 使用 pipeline 推理
    try:
        # 注意：你的 pipeline 可能接受 PIL Image 或路径，这里直接传 PIL
        result_image = pipeline(
            prompt="",
            image=image,                    # 直接传 PIL Image
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
            generator=torch.Generator(device=device).manual_seed(123),
            guidance_scale=1,
            attn=attn_aggregator,
        ).images[0]
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result_image, "移除完成！"
    except Exception as e:
        return None, f"出错：{str(e)}"

# ================= 构建 Gradio 界面 =================
with gr.Blocks(title="ClickRemoval - 点击移除对象", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖱️ ClickRemoval：点击即移除
    **上传图片**，在想要移除的对象上点击红色点（正点击），  
    如果某些区域需要保留，可切换到“负点击”添加蓝色点，然后点击“开始移除”。
    """)

    # 状态存储（每个会话独立）
    positive_state = gr.State([])   # 正点击坐标列表
    negative_state = gr.State([])   # 负点击坐标列表
    mode_state = gr.State("正点击（移除）")  # 当前模式

    with gr.Row():
        with gr.Column(scale=1):
            # 图片上传与显示
            input_img = gr.Image(
                type="pil",
                label="点击图片上的位置添加点",
                interactive=True,
                height=512,
                width=512
            )
            # 模式控制
            with gr.Row():
                positive_btn = gr.Button("➕ 正点击（移除）", variant="primary")
                negative_btn = gr.Button("➖ 负点击（保留）", variant="secondary")
            mode_text = gr.Textbox(label="当前模式", value="正点击（移除）", interactive=False)
            reset_btn = gr.Button("🗑️ 重置所有点", variant="secondary")
            remove_btn = gr.Button("✨ 开始移除", variant="primary")
            info_text = gr.Textbox(label="状态信息", interactive=False)

        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="移除结果", height=512, width=512)

    # ========== 事件绑定 ==========
    # 上传图片时重置所有状态
    input_img.change(
        upload_image,
        inputs=[input_img],
        outputs=[input_img, positive_state, negative_state, mode_text, output_img]
    )

    # 点击图片：添加点
    input_img.select(
        handle_click,
        inputs=[input_img, positive_state, negative_state, mode_state],
        outputs=[input_img, positive_state, negative_state, info_text]
    )

    # 切换模式
    positive_btn.click(
        lambda: "正点击（移除）",
        outputs=[mode_state]
    ).then(
        lambda mode: mode,
        inputs=[mode_state],
        outputs=[mode_text]
    )
    negative_btn.click(
        lambda: "负点击（保留）",
        outputs=[mode_state]
    ).then(
        lambda mode: mode,
        inputs=[mode_state],
        outputs=[mode_text]
    )

    # 重置点：清空状态并重新绘制图片（不带点）
    reset_btn.click(
        reset_points,
        inputs=[input_img],
        outputs=[input_img, positive_state, negative_state, mode_text, output_img]
    )

    # 开始移除
    remove_btn.click(
        remove_objects,
        inputs=[input_img, positive_state, negative_state],
        outputs=[output_img, info_text]
    )

# ================= 启动服务 =================
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)