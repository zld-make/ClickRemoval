# prepare_local_fid.py
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def get_safe_square_region(mask_np, img_shape, expand_ratio=0.25, min_size=299):
    """
    Extract a safe square region around the mask area.
    """
    h_img, w_img = img_shape
    ys, xs = np.where(mask_np > 128)
    if len(ys) == 0:
        return None

    y1, y2 = np.min(ys), np.max(ys)
    x1, x2 = np.min(xs), np.max(xs)
    bbox_h = y2 - y1
    bbox_w = x2 - x1

    pad_h = int(bbox_h * expand_ratio)
    pad_w = int(bbox_w * expand_ratio)
    y1_exp = max(0, y1 - pad_h)
    y2_exp = min(h_img, y2 + pad_h)
    x1_exp = max(0, x1 - pad_w)
    x2_exp = min(w_img, x2 + pad_w)

    cur_h = y2_exp - y1_exp
    cur_w = x2_exp - x1_exp

    target_size = max(cur_h, cur_w, min_size)

    center_y = (y1_exp + y2_exp) // 2
    center_x = (x1_exp + x2_exp) // 2

    half = target_size // 2
    y1_final = max(0, center_y - half)
    y2_final = min(h_img, center_y + half)
    x1_final = max(0, center_x - half)
    x2_final = min(w_img, center_x + half)

    final_h = y2_final - y1_final
    final_w = x2_final - x1_final

    if final_h < target_size:
        if y1_final > 0:
            y1_final = max(0, y1_final - (target_size - final_h))
        else:
            y2_final = min(h_img, y2_final + (target_size - final_h))
    if final_w < target_size:
        if x1_final > 0:
            x1_final = max(0, x1_final - (target_size - final_w))
        else:
            x2_final = min(w_img, x2_final + (target_size - final_w))

    return int(x1_final), int(y1_final), int(x2_final), int(y2_final)


def find_mask_path(image_stem, mask_dir):
    """
    Find the mask file corresponding to an image.
    """
    candidates = []

    candidates.append(f"{image_stem}.png")

    if image_stem.endswith("_output"):
        base = image_stem[:-7]
        candidates.append(f"{base}_input.png")
    else:
        candidates.append(f"{image_stem}_input.png")

    # Handle _removed suffix
    if image_stem.endswith("_removed"):
        base = image_stem[:-8]  # remove "_removed"
        candidates.append(f"{base}.png")
        candidates.append(f"{base}_input.png")

    # Handle _output suffix again (fallback)
    if image_stem.endswith("_output"):
        base = image_stem[:-7]
        candidates.append(f"{base}.png")

    # Deduplicate and check existence
    for cand in set(candidates):
        mask_path = Path(mask_dir) / cand
        if mask_path.exists():
            return mask_path
    return None


def process_folder(image_dir, mask_dir, output_dir, expand_ratio=0.25, min_size=299):
    """
    Batch process a folder: crop local regions for each image based on its mask.
    """
    os.makedirs(output_dir, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]

    for img_path in image_paths:
        # Find corresponding mask
        mask_path = find_mask_path(img_path.stem, mask_dir)

        if mask_path is None:
            print(f"Warning: No mask found for {img_path.name}, skipping")
            continue

        # Load image and mask
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Skipping corrupted file {img_path.name}: {e}")
            continue
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        img_np = np.array(img)

        region = get_safe_square_region(mask_np, img_np.shape[:2], expand_ratio, min_size)
        if region is None:
            print(f"Warning: No valid region in mask {mask_path.name}, skipping {img_path.name}")
            continue

        x1, y1, x2, y2 = region
        cropped = img.crop((x1, y1, x2, y2))

        if cropped.size[0] < min_size or cropped.size[1] < min_size:
            cropped = cropped.resize((min_size, min_size), Image.Resampling.LANCZOS)

        out_path = Path(output_dir) / img_path.name
        cropped.save(out_path)
        print(f"Processed: {img_path.name} (mask: {mask_path.name}) -> {out_path} (size {cropped.size})")

    print(f"Done! Processed folder: {image_dir} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch crop local regions for Local-FID calculation")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing original or generated images")
    parser.add_argument("--mask_dir", required=True,
                        help="Directory containing mask images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save cropped patches")
    parser.add_argument("--expand_ratio", type=float, default=0.25,
                        help="Expansion ratio around mask bounding box (default: 0.25)")
    parser.add_argument("--min_size", type=int, default=299,
                        help="Minimum side length of the cropped square (default: 299)")
    args = parser.parse_args()

    process_folder(args.image_dir, args.mask_dir, args.output_dir,
                   args.expand_ratio, args.min_size)


if __name__ == "__main__":
    main()