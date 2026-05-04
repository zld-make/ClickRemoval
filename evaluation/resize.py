import os
from PIL import Image
from tqdm import tqdm
import argparse


def resize_images(input_dir, output_dir, size=(299, 299), resample=Image.Resampling.LANCZOS):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    for fname in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        try:
            img = Image.open(in_path).convert('RGB')
            img_resized = img.resize(size, resample)
            img_resized.save(out_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images to a fixed square size")
    parser.add_argument("--input_dir", default="./input",
                        help="Directory containing input images (default: ./input)")
    parser.add_argument("--output_dir", default="./output",
                        help="Directory to save resized images (default: ./output)")
    parser.add_argument("--size", type=int, default=299,
                        help="Target square size (width = height, default: 299)")
    parser.add_argument("--resample", type=str, default="lanczos",
                        choices=["lanczos", "bicubic", "bilinear"],
                        help="Resampling method: lanczos, bicubic, or bilinear (default: lanczos)")

    args = parser.parse_args()

    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
    }
    resample = resample_map[args.resample]

    resize_images(args.input_dir, args.output_dir, (args.size, args.size), resample)