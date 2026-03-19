# app.py
import gradio as gr
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
import json
from pathlib import Path
import traceback
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import shutil
import cv2

# 全局变量存储点击点
global_points = []
global_labels = []


# 根据模型类型动态导入对应的注意力聚合器
def load_model_pipeline(model_type, model_path):
    """加载指定的模型和管道"""
    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False
    )

    if model_type == "SDXL":
        from src.stable_diffusion_xl_attention_aggregator import StableDiffusionxlAttentionAggregator
        custom_pipeline = "./pipelines/mypiplinexl.py"
        attn_aggregator = StableDiffusionxlAttentionAggregator(device=device)
    elif model_type == "SD2.1":
        from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
        custom_pipeline = "./pipelines/mypipline2x.py"
        attn_aggregator = StableDiffusion2AttentionAggregator(device=device)
    elif model_type == "SD1.5":
        from src.stable_diffusion_1_attention_aggregator import StableDiffusion1AttentionAggregator
        custom_pipeline = "./pipelines/mypipline1x.py"
        attn_aggregator = StableDiffusion1AttentionAggregator(device=device)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    # 加载管道
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline=custom_pipeline,
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=dtype,
    ).to(device)

    # 启用内存优化
    pipeline.enable_attention_slicing()
    pipeline.enable_model_cpu_offload()

    # 预清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipeline, attn_aggregator, device


def parse_coordinates(coords_text):
    """解析坐标文本（支持JSON格式或纯文本格式）"""
    points = []
    labels = []

    try:
        # 尝试解析为JSON格式
        if coords_text.strip().startswith('{'):
            data = json.loads(coords_text)
            points = data.get("points", [])
            labels = data.get("labels", [])
        elif coords_text.strip():
            # 解析为纯文本格式：x1,y1,label1;x2,y2,label2;...
            lines = coords_text.strip().split(';')
            for line in lines:
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        label = int(parts[2].strip())
                        points.append((x, y))
                        labels.append(label)

    except Exception as e:
        print(f"解析坐标时出错: {e}")

    return points, labels


def draw_points_on_image(image, points, labels):
    """在图像上绘制坐标点"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # 定义不同标签的颜色
    colors = {
        0: ('red', 'R'),  # 要移除的区域
        1: ('green', 'K'),  # 要保留的区域
    }

    # 绘制点
    for idx, ((x, y), label) in enumerate(zip(points, labels)):
        color, text = colors.get(label, ('yellow', '?'))
        # 将坐标转换为图像上的实际位置
        img_x = int(x * image.width)
        img_y = int(y * image.height)

        # 绘制点
        radius = 8
        draw.ellipse(
            [(img_x - radius, img_y - radius), (img_x + radius, img_y + radius)],
            fill=color,
            outline='white',
            width=2
        )

        # 添加标签文本
        draw.text(
            (img_x + radius + 5, img_y - 10),
            f"{text}{idx + 1}",
            fill='white',
            stroke_width=2,
            stroke_fill='black'
        )

    # 添加图例
    legend_y = 20
    for label, (color, text) in colors.items():
        draw.ellipse([(10, legend_y), (25, legend_y + 15)], fill=color)
        draw.text(
            (30, legend_y - 2),
            f"左键: {text} ({'移除' if label == 0 else '保留'})",
            fill='white',
            stroke_width=1,
            stroke_fill='black'
        )
        legend_y += 25

    return img_copy


def handle_image_click(img, evt: gr.SelectData):
    """处理图像点击事件"""
    global global_points, global_labels

    if img is None:
        return None, "请先上传图像"

    # 获取点击坐标和按钮信息
    # evt.index 返回 (row, col) 格式，即 (y, x)
    y, x = evt.index

    # 转换为归一化坐标 (0-1)
    height, width = img.shape[:2] if len(img.shape) == 3 else (img.shape[0], img.shape[1])
    x_norm = x / width
    y_norm = y / height

    # 判断是左键还是右键点击
    # Gradio 的 SelectData 目前不直接提供按钮信息
    # 我们将通过一个简单的规则：左键添加移除点，右键添加保留点
    # 实际上需要前端配合，这里我们先假设都是左键，右键功能通过按钮切换模式实现

    # 默认使用移除点（标签0）
    label = 0

    # 添加坐标点
    global_points.append((x_norm, y_norm))
    global_labels.append(label)

    # 创建带标注的图像
    pil_img = Image.fromarray(img)
    annotated_img = draw_points_on_image(pil_img, global_points, global_labels)

    # 更新坐标显示
    coords_text = format_coordinates(global_points, global_labels)

    return annotated_img, coords_text


def handle_image_right_click(img, evt: gr.SelectData):
    """处理图像右键点击事件"""
    global global_points, global_labels

    if img is None:
        return None, "请先上传图像"

    # 获取点击坐标
    y, x = evt.index

    # 转换为归一化坐标
    height, width = img.shape[:2] if len(img.shape) == 3 else (img.shape[0], img.shape[1])
    x_norm = x / width
    y_norm = y / height

    # 右键添加保留点（标签1）
    label = 1

    # 添加坐标点
    global_points.append((x_norm, y_norm))
    global_labels.append(label)

    # 创建带标注的图像
    pil_img = Image.fromarray(img)
    annotated_img = draw_points_on_image(pil_img, global_points, global_labels)

    # 更新坐标显示
    coords_text = format_coordinates(global_points, global_labels)

    return annotated_img, coords_text


def clear_points():
    """清除所有坐标点"""
    global global_points, global_labels
    global_points = []
    global_labels = []
    return None, "坐标已清除", ""


def undo_last_point():
    """撤销最后一个坐标点"""
    global global_points, global_labels
    if global_points:
        global_points.pop()
        global_labels.pop()

    coords_text = format_coordinates(global_points, global_labels)
    return f"已撤销最后一个点，剩余 {len(global_points)} 个点", coords_text


def format_coordinates(points, labels):
    """格式化坐标显示"""
    if not points:
        return "尚未添加任何坐标点"

    lines = []
    for i, ((x, y), label) in enumerate(zip(points, labels)):
        label_text = "移除" if label == 0 else "保留"
        lines.append(f"点{i + 1}: ({x:.3f}, {y:.3f}) - {label_text}")

    return "\n".join(lines)


def export_coordinates():
    """导出坐标为JSON格式"""
    global global_points, global_labels

    if not global_points:
        return "{}"

    data = {
        "points": global_points,
        "labels": global_labels
    }

    return json.dumps(data, indent=2)


def import_coordinates(coords_text):
    """从文本导入坐标"""
    global global_points, global_labels

    try:
        if not coords_text.strip():
            global_points = []
            global_labels = []
            return None, "坐标已清除", ""

        points, labels = parse_coordinates(coords_text)

        if not points:
            return None, "未找到有效坐标", ""

        global_points = points
        global_labels = labels

        # 如果有当前图像，更新标注
        status = f"已导入 {len(points)} 个坐标点"
        coords_display = format_coordinates(points, labels)

        return status, coords_display

    except Exception as e:
        return None, f"导入坐标时出错: {str(e)}", ""


def update_annotation(img):
    """更新图像标注"""
    global global_points, global_labels

    if img is None or not global_points:
        return img, format_coordinates(global_points, global_labels)

    pil_img = Image.fromarray(img)
    annotated_img = draw_points_on_image(pil_img, global_points, global_labels)

    return annotated_img, format_coordinates(global_points, global_labels)


def process_image(
        input_image,
        model_type,
        coords_text,
        strength,
        num_inference_steps,
        guidance_scale,
        use_mouse_points
):
    """
    处理单张图像
    """
    # 模型路径映射
    model_paths = {
        "SDXL": "/home/zld/work/AttentiveEraser-master/hy-tmp/stable-diffusion-xl-base-1.0",
        "SD2.1": "/home/zld/work/BrushNet-main/BrushNet-main/data/ckpt/stable-diffusion-2-1-base",
        "SD1.5": "/home/zld/work/BrushNet-main/BrushNet-main/data/ckpt/stable-diffusion-v1-5"
    }

    # 保存输入图像到临时文件
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        input_image.save(tmp_file.name)
        input_path = tmp_file.name

    try:
        # 获取坐标点
        if use_mouse_points:
            # 使用鼠标点击的点
            points = global_points.copy()
            labels = global_labels.copy()
        else:
            # 使用文本输入的坐标
            points, labels = parse_coordinates(coords_text)

        if not points:
            # 如果没有点，尝试使用全局点
            if global_points:
                points = global_points.copy()
                labels = global_labels.copy()
            else:
                return None, None, "错误: 未找到有效的坐标点"

        print(f"使用 {len(points)} 个坐标点")

        # 加载模型
        print(f"加载 {model_type} 模型...")
        pipeline, attn_aggregator, device = load_model_pipeline(
            model_type,
            model_paths[model_type]
        )

        # 设置生成器
        generator = torch.Generator(device=device).manual_seed(42)

        # 根据模型类型设置不同的参数
        if model_type == "SDXL":
            # SDXL 参数
            result = pipeline(
                prompt="",
                image=input_path,
                points=points,
                points_in_segment=labels,
                height=1024,
                width=1024,
                AAS=True,
                strength=strength,
                rm_guidance_scale=5,
                ss_steps=9,
                ss_scale=0.3,
                AAS_start_step=0,
                AAS_start_layer=34,
                AAS_end_layer=70,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                attn=attn_aggregator,
            )
        else:
            # SD2.1 和 SD1.5 参数
            result = pipeline(
                prompt="",
                image=input_path,
                points=points,
                points_in_segment=labels,
                height=512,
                width=512,
                SGA=True,
                strength=strength,
                rm_guidance_scale=9,
                sg_steps=9,
                sg_scale=0.3,
                SGA_start_step=0,
                SGA_start_layer=12,
                SGA_end_layer=32,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                attn_aggregator=attn_aggregator,
            )

        # 获取结果图像
        output_image = result.images[0]

        # 在原图上绘制坐标点
        annotated_image = draw_points_on_image(input_image, points, labels)

        # 清理临时文件
        os.unlink(input_path)

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return annotated_image, output_image, "处理成功！"

    except Exception as e:
        # 清理临时文件
        if os.path.exists(input_path):
            os.unlink(input_path)

        error_msg = f"处理过程中出现错误:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def batch_process(
        input_dir,
        coords_dir,
        output_dir,
        model_type,
        progress=gr.Progress()
):
    """
    批量处理图像
    """
    # 模型路径映射
    model_paths = {
        "SDXL": "/home/zld/work/AttentiveEraser-master/hy-tmp/stable-diffusion-xl-base-1.0",
        "SD2.1": "/home/zld/work/BrushNet-main/BrushNet-main/data/ckpt/stable-diffusion-2-1-base",
        "SD1.5": "/home/zld/work/BrushNet-main/BrushNet-main/data/ckpt/stable-diffusion-v1-5"
    }

    try:
        # 检查输入目录
        input_path = Path(input_dir)
        if not input_path.exists():
            return "错误: 输入目录不存在"

        # 检查坐标目录
        coords_path = Path(coords_dir)
        if not coords_path.exists():
            return "错误: 坐标目录不存在"

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))

        if not image_files:
            return "错误: 输入目录中未找到图片文件"

        print(f"找到 {len(image_files)} 个图片文件")

        # 加载模型
        progress(0, desc="加载模型中...")
        pipeline, attn_aggregator, device = load_model_pipeline(
            model_type,
            model_paths[model_type]
        )

        results = []
        processed_count = 0

        for i, image_file in enumerate(image_files):
            try:
                progress(i / len(image_files), desc=f"处理图片 {i + 1}/{len(image_files)}")

                # 查找对应的坐标文件
                json_name = image_file.stem + ".json"
                json_file = coords_path / json_name

                if not json_file.exists():
                    results.append(f"警告: 未找到 {json_name}")
                    continue

                # 加载坐标文件
                with open(json_file, 'r') as f:
                    data = json.load(f)

                points = data.get("points", [])
                labels = data.get("labels", [])

                if not points or len(points) != len(labels):
                    results.append(f"警告: {json_name} 格式不正确")
                    continue

                # 设置生成器
                generator = torch.Generator(device=device).manual_seed(123 + i)

                # 根据模型类型处理
                if model_type == "SDXL":
                    result = pipeline(
                        prompt="",
                        image=str(image_file),
                        points=points,
                        points_in_segment=labels,
                        height=1024,
                        width=1024,
                        AAS=True,
                        strength=0.8,
                        rm_guidance_scale=5,
                        ss_steps=9,
                        ss_scale=0.3,
                        AAS_start_step=0,
                        AAS_start_layer=34,
                        AAS_end_layer=70,
                        num_inference_steps=50,
                        generator=generator,
                        guidance_scale=1,
                        attn=attn_aggregator,
                    )
                else:
                    result = pipeline(
                        prompt="",
                        image=str(image_file),
                        points=points,
                        points_in_segment=labels,
                        height=512,
                        width=512,
                        SGA=True,
                        strength=0.8,
                        rm_guidance_scale=9,
                        sg_steps=9,
                        sg_scale=0.3,
                        SGA_start_step=0,
                        SGA_start_layer=12,
                        SGA_end_layer=32,
                        num_inference_steps=50,
                        generator=generator,
                        guidance_scale=1,
                        attn_aggregator=attn_aggregator,
                    )

                # 保存结果
                output_file = output_path / f"{image_file.stem}_removed.png"
                result.images[0].save(output_file)
                processed_count += 1
                results.append(f"✓ {image_file.name} -> {output_file.name}")

                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                results.append(f"✗ {image_file.name}: {str(e)}")
                continue

        summary = f"批量处理完成！\n处理了 {processed_count}/{len(image_files)} 张图片\n输出目录: {output_dir}"
        return "\n".join([summary] + results)

    except Exception as e:
        return f"批量处理错误: {str(e)}\n{traceback.format_exc()}"


def create_web_app():
    """创建 Gradio Web 应用"""

    with gr.Blocks(title="Interactive Image Inpainting Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🖼️ 交互式图像修复演示")
        gr.Markdown("上传图像，通过鼠标点击标记要修复的区域，然后选择模型进行修复")

        with gr.Tabs():
            # 单张图像处理标签页
            with gr.Tab("交互式图像修复"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 模型选择
                        model_selector = gr.Dropdown(
                            choices=["SDXL", "SD2.1", "SD1.5"],
                            value="SD1.5",
                            label="选择修复模型"
                        )

                        # 图像上传和鼠标交互
                        gr.Markdown("### 1. 上传并标记图像")
                        input_image = gr.Image(
                            type="numpy",
                            label="点击图像标记坐标",
                            interactive=True
                        )

                        # 鼠标交互控制
                        with gr.Row():
                            clear_btn = gr.Button("清除所有标记", variant="secondary")
                            undo_btn = gr.Button("撤销上一个点", variant="secondary")
                            export_btn = gr.Button("导出坐标", variant="secondary")

                        # 坐标显示和编辑
                        coords_display = gr.Textbox(
                            label="当前坐标点",
                            lines=6,
                            interactive=False
                        )

                        coords_input = gr.Textbox(
                            label="手动输入/编辑坐标 (JSON格式)",
                            placeholder='{"points": [[0.3,0.5], [0.7,0.5]], "labels": [0, 1]}',
                            lines=4
                        )

                        with gr.Row():
                            import_btn = gr.Button("导入坐标", variant="secondary")
                            use_text_coords = gr.Checkbox(
                                label="使用手动输入的坐标（不勾选则使用鼠标标记的坐标）",
                                value=False
                            )

                        # 处理参数
                        gr.Markdown("### 2. 设置修复参数")
                        with gr.Row():
                            strength = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="修复强度"
                            )

                            num_inference_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                label="推理步数"
                            )

                        guidance_scale = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=7.5,
                            step=0.5,
                            label="引导比例"
                        )

                        # 处理按钮
                        process_btn = gr.Button("开始修复", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        # 标注图像显示
                        annotated_image = gr.Image(
                            type="pil",
                            label="标注后的图像"
                        )

                        # 修复结果显示
                        output_image = gr.Image(
                            type="pil",
                            label="修复结果"
                        )

                        # 状态显示
                        status_text = gr.Textbox(
                            label="处理状态",
                            interactive=False,
                            lines=3
                        )

                # 使用说明
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""
                    ### 操作指南:
                    1. **上传图像**: 点击上传按钮选择要修复的图像
                    2. **标记坐标**:
                       - **左键点击**: 标记要移除的区域（红色点，标记为"R"）
                       - **右键点击**: 标记要保留的区域（绿色点，标记为"K"）
                       - **撤销**: 点击"撤销上一个点"按钮移除最后一个标记
                       - **清除**: 点击"清除所有标记"按钮移除所有标记
                    3. **导入/导出坐标**:
                       - **导出**: 点击"导出坐标"将当前标记导出为JSON格式
                       - **导入**: 在文本框中输入JSON坐标，点击"导入坐标"按钮
                    4. **选择模型**: 从下拉菜单中选择修复模型
                    5. **调整参数**: 根据需要调整修复参数
                    6. **开始修复**: 点击"开始修复"按钮进行处理

                    ### 坐标格式:
                    ```json
                    {
                      "points": [[x1, y1], [x2, y2], ...],
                      "labels": [0, 1, ...]
                    }
                    ```
                    - **points**: 归一化坐标，范围 0-1
                    - **labels**: 0 表示移除，1 表示保留

                    ### 模型说明:
                    - **SDXL**: 高质量修复，推荐用于高分辨率图像
                    - **SD2.1**: 平衡质量和速度
                    - **SD1.5**: 经典模型，速度快
                    """)

                # 示例坐标按钮
                with gr.Row():
                    gr.Markdown("### 示例坐标:")
                    example_single = gr.Button("单个移除点")
                    example_pair = gr.Button("点对 (移除+保留)")
                    example_complex = gr.Button("复杂示例")

                # 示例坐标数据
                single_point = '{"points": [[0.5, 0.5]], "labels": [0]}'
                pair_points = '{"points": [[0.3, 0.5], [0.7, 0.5]], "labels": [0, 1]}'
                complex_example = '''{
                    "points": [
                        [0.3, 0.3], 
                        [0.7, 0.3], 
                        [0.5, 0.7]
                    ],
                    "labels": [0, 1, 0]
                }'''

                # 绑定事件
                # 图像点击事件
                input_image.select(
                    fn=handle_image_click,
                    inputs=[input_image],
                    outputs=[annotated_image, coords_display]
                )

                # 图像右键点击事件（通过选择事件模拟）
                input_image.select(
                    fn=handle_image_right_click,
                    inputs=[input_image],
                    outputs=[annotated_image, coords_display]
                )

                # 按钮事件
                clear_btn.click(
                    fn=clear_points,
                    outputs=[input_image, coords_display, coords_input]
                )

                undo_btn.click(
                    fn=undo_last_point,
                    outputs=[status_text, coords_display]
                )

                export_btn.click(
                    fn=export_coordinates,
                    outputs=coords_input
                )

                import_btn.click(
                    fn=import_coordinates,
                    inputs=coords_input,
                    outputs=[status_text, coords_display, coords_input]
                )

                # 示例坐标按钮
                example_single.click(
                    fn=lambda: single_point,
                    outputs=coords_input
                ).then(
                    fn=lambda: ("已加载单个点示例", single_point),
                    outputs=[status_text, coords_display]
                )

                example_pair.click(
                    fn=lambda: pair_points,
                    outputs=coords_input
                ).then(
                    fn=lambda: ("已加载点对示例", pair_points),
                    outputs=[status_text, coords_display]
                )

                example_complex.click(
                    fn=lambda: complex_example,
                    outputs=coords_input
                ).then(
                    fn=lambda: ("已加载复杂示例", complex_example),
                    outputs=[status_text, coords_display]
                )

                # 处理按钮事件
                process_btn.click(
                    fn=process_image,
                    inputs=[
                        input_image,
                        model_selector,
                        coords_input,
                        strength,
                        num_inference_steps,
                        guidance_scale,
                        use_text_coords
                    ],
                    outputs=[
                        annotated_image,
                        output_image,
                        status_text
                    ]
                )

            # 批量处理标签页
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column():
                        batch_model_selector = gr.Dropdown(
                            choices=["SDXL", "SD2.1", "SD1.5"],
                            value="SD1.5",
                            label="选择模型"
                        )

                        input_dir = gr.Textbox(
                            label="图像文件夹路径",
                            value="./data/examples/img",
                            placeholder="包含图片的文件夹路径"
                        )

                        coords_dir = gr.Textbox(
                            label="坐标文件夹路径",
                            value="./data/examples/coord",
                            placeholder="包含JSON坐标文件的文件夹路径"
                        )

                        output_dir = gr.Textbox(
                            label="输出文件夹路径",
                            value="./output/batch_results",
                            placeholder="保存结果的文件夹路径"
                        )

                        batch_process_btn = gr.Button("开始批量处理", variant="primary")

                    with gr.Column():
                        batch_status = gr.Textbox(
                            label="批量处理状态",
                            lines=20,
                            max_lines=50
                        )

                batch_process_btn.click(
                    fn=batch_process,
                    inputs=[
                        input_dir,
                        coords_dir,
                        output_dir,
                        batch_model_selector
                    ],
                    outputs=batch_status
                )

        # 页面底部信息
        with gr.Row():
            gr.Markdown("""
            ### 注意事项:
            1. **首次使用**: 首次运行需要下载模型，请耐心等待
            2. **硬件要求**: 建议使用GPU以获得最佳性能
            3. **图像大小**: SDXL支持1024x1024，SD1.5/2.1支持512x512
            4. **坐标精度**: 坐标是归一化的，范围0-1
            5. **点标记**: 左键标记移除区域，右键标记保留区域

            ### 开源信息:
            - 项目地址: [GitHub Repository](https://github.com/yourusername/image-inpainting-demo)
            - 基于: Attentive Eraser 和 BrushNet 技术
            - 许可证: MIT License
            """)

    return app


# 创建必要的目录
def setup_directories():
    """创建必要的目录结构"""
    dirs = [
        "./data/examples/img",
        "./data/examples/coord",
        "./output",
        "./output/batch_results",
        "./pipelines",
        "./src"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"已创建/确认目录: {dir_path}")


# 前端JavaScript代码，用于更好的鼠标交互
javascript_code = """
<script>
// 自定义鼠标交互
function setupImageInteraction() {
    const imageElements = document.querySelectorAll('[data-testid="image"]');

    imageElements.forEach(img => {
        // 添加右键点击支持
        img.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            // 这里可以触发一个自定义事件
            // 由于Gradio的限制，我们通过其他方式处理
        });
    });
}

// 页面加载完成后执行
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupImageInteraction);
} else {
    setupImageInteraction();
}
</script>
"""

if __name__ == "__main__":
    # 设置目录
    setup_directories()

    # 创建并启动应用
    app = create_web_app()

    # 添加自定义JavaScript
    app.head = javascript_code

    # 启动 Gradio 应用
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 设置为 True 可以创建公共链接
        debug=True
    )