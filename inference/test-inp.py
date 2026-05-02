import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
import json
from pathlib import Path
from src.stable_diffusion_2_attention_aggregator import StableDiffusion2AttentionAggregator
import traceback

if __name__ == "__main__":
    image_dir = "/home/zld/work/my/data/examples/img"
    coord_dir = "/home/zld/work/my/data/examples/coord"
    output_dir = "./out"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取图片文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))

    if not image_files:
        print("未找到图片文件！")
        exit()

    print(f"找到 {len(image_files)} 个图片文件")

    # 初始化模型（只加载一次）
    dtype = torch.float16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)

    # 修改为你的模型路径
    model_path = r"/home/zld/work/BrushNet-main/BrushNet-main/data/ckpt/stable-diffusion-inpainting"
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline="./pipelines/mypiplineinp.py",
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


    def preprocess_image(image_path, device):
        """预处理图片"""
        image = to_tensor((load_image(image_path)))
        image = image.unsqueeze_(0).float() * 2 - 1  # [0,1] --> [-1,1]
        if image.shape[1] != 3:
            image = image.expand(-1, 3, -1, -1)
        image = F.interpolate(image, (1024, 1024))
        image = image.to(dtype).to(device)
        return image


    def preprocess_mask(mask_path, device):
        """预处理掩码"""
        mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
        mask = mask.unsqueeze_(0).float()  # 0 or 1
        mask = F.interpolate(mask, (1024, 1024))
        mask = gaussian_blur(mask, kernel_size=(77, 77))
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(dtype).to(device)
        return mask


    def load_coordinates(json_path):
        """加载坐标文件"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            points = data.get("points", [])
            labels = data.get("labels", [])

            # 确保坐标格式正确
            formatted_points = []
            for point in points:
                if len(point) == 2:
                    formatted_points.append((point[0], point[1]))

            return formatted_points, labels
        except Exception as e:
            traceback.print_exc()
            return [], []

    attn_aggregator = StableDiffusion2AttentionAggregator(device='cuda:0')
    # 处理每张图片
    for i, image_path in enumerate(image_files):
        try:
            print(f"\n处理第 {i + 1}/{len(image_files)} 张图片: {image_path.name}")

            # 构建对应的坐标文件路径
            json_name = image_path.stem + ".json"
            json_path = Path(coord_dir) / json_name

            if not json_path.exists():
                print(f"警告: 未找到对应的坐标文件 {json_name}")
                continue

            # 加载坐标
            points, labels = load_coordinates(json_path)

            if not points or len(points) != len(labels):
                print(f"警告: 坐标文件 {json_name} 格式不正确或为空")
                continue

            print(f"加载了 {len(points)} 个坐标点")

            # 设置随机种子（可选：每张图片使用不同的种子）
            seed = 123 + i  # 可以根据需要修改种子策略
            generator = torch.Generator(device=device).manual_seed(seed)

            # 执行修复
            image = pipeline(
                prompt="",
                image=str(image_path),
                points=points,
                points_in_segment=labels,
                height=512,  # SD1.5在第26层时潜在表示为128，而SDXL最大为64，在自注意力计算时（平方），128会显著增加（128是64的16倍）显存利用（即便SDXL深度大于SD1.5）
                width=512,
                AAS=True,
                strength=0.8,
                rm_guidance_scale=9,
                ss_steps=9,
                ss_scale=0.3,
                AAS_start_step=0,
                AAS_start_layer=2,
                AAS_end_layer=32,
                num_inference_steps=50,
                generator=generator,
                guidance_scale=1,
                attn_aggregator=attn_aggregator,
            ).images[0]

            # 保存结果
            output_path = Path(output_dir) / f"{image_path.stem}_removed.png"
            image.save(output_path)
            print(f"保存结果到: {output_path}")

            # 清理显存（避免累积）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            traceback.print_exc()
            # 继续处理下一张图片
            continue

    print("\n批量处理完成！")