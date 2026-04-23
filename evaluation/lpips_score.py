import argparse
import numpy as np
import torch
from tqdm import tqdm
from lpips import PerceptualLoss
import glob
import os

import cv2
import PIL.Image as Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')

def load_image(fname, mode='RGB', eval_resolution=256,return_orig=False):
    img = np.array(Image.open(fname).resize((eval_resolution,eval_resolution), Image.Resampling.BILINEAR).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class InpaintingDataset(Dataset):
    def __init__(self, datadir, predictdir, img_suffix='.jpg', eval_resolution=256,inpainted_suffix='_inpainted.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        if not datadir.endswith('/'):
            datadir += '/'
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.img_filenames]
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]
        self.test_filenames = [os.path.join("/hy-tmp/DATA/sample/", id + img_suffix) for id in self.ids]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor
        self.eval_resolution = eval_resolution

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.test_filenames[i], mode='RGB',eval_resolution=self.eval_resolution)
        mask = load_image(self.mask_filenames[i], mode='L',eval_resolution=self.eval_resolution)
        result = dict(image=image, mask=mask[None, ...])
        result['inpainted'] = load_image(self.pred_filenames[i], mode='RGB', eval_resolution=self.eval_resolution)
        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            result['inpainted'] = scale_image(result['inpainted'], self.scale_factor)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)

        return result
    
class LPIPSScore(nn.Module):
    def __init__(self, model='net-lin', net='vgg', model_path="/hy-tmp/lama/models/lpips_models/vgg.pth", use_gpu=True):
        super().__init__()
        self.score = PerceptualLoss(model=model, net=net, model_path=model_path,
                                    use_gpu=use_gpu, spatial=False).eval()

    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch).flatten()
        return batch_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        default=".DATA/original/",
        help="Directory of the original images and masks",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/fid",
        help="Directory of evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Bath size of Inception v3 forward pass",
    )
    parser.add_argument(
        "--inpainted_suffix",
        type=str,
        default='_removed.png',
        help="inference_dir's suffix",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = LPIPSScore()
    score.to(device)
    args = parser.parse_args()
    dataset = InpaintingDataset(args.datadir, args.inference_dir, img_suffix='.jpg', eval_resolution=512, inpainted_suffix=args.inpainted_suffix)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    total_score = 0
    num_batches = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = move_to_device(batch, device)
        image_batch, mask_batch, inpainted_batch = batch['image'], batch['mask'], batch['inpainted']

        batch_score=score(inpainted_batch, image_batch, mask_batch)
        total_score += batch_score.mean().item()
        num_batches += 1
    average_score = total_score / num_batches

    print(f'LPIPS score: {average_score:.4f}')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/lpips.txt", 'w') as f:
        f.write(str(average_score))
    print(f"output to {args.output_dir}.txt ")