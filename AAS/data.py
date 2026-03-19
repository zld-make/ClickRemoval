import csv
import glob
import os

import cv2
import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import gaussian_blur

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

def load_image(image_path):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (512, 512), mode="bicubic")
    image = image.to(torch.float16)
    return image

def load_mask(mask_path):
    mask = read_image(mask_path,mode=ImageReadMode.GRAY)
    mask = mask.unsqueeze_(0).float() / 255.  # 0 or 1
    mask = F.interpolate(mask, (512, 512), mode="bicubic")
    mask = gaussian_blur(mask, kernel_size=(13, 13))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(torch.float16)
    return mask

def load_image_xl(image_path):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(torch.float32)
    return image

def load_mask_xl(mask_path):
    mask = read_image(mask_path,mode=ImageReadMode.GRAY)
    mask = mask.unsqueeze_(0).float() / 255.  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(13, 13))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(torch.float32)
    return mask
    
class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i])
        mask= load_mask(self.mask_filenames[i])
        result = dict(image=image, mask=mask) # mask[None, ...] is equivalent to mask[np.newaxis, ...]

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result
    
class InpaintingDataset_with_text(InpaintingDataset):
    def __init__(self, datadir, test_scene, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        super().__init__(datadir, img_suffix, pad_out_to_modulo, scale_factor)
        self.texts = [os.path.basename(os.path.dirname(fname)) for fname in self.img_filenames]
        self.test_scene  = self.read_csv_to_dict(test_scene)
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]

    def read_csv_to_dict(self,file_path):
        data_dict = {} 
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader)  # Skip header if there is one
            for row in reader:
                id = row[0].rsplit('.', 1)[0]
                LabelName = row[1]
                BoxXMin = float(row[2])
                BoxXMax = float(row[3])
                BoxYMin = float(row[4])
                BoxYMax = float(row[5])    
                data_dict[id] = {
                    'LabelName': LabelName,
                    'BoxXMin': BoxXMin,
                    'BoxXMax': BoxXMax,
                    'BoxYMin': BoxYMin,
                    'BoxYMax': BoxYMax
                }      
        return data_dict

    def __getitem__(self, i):
        result = super().__getitem__(i)
        scene_id = self.ids[i]
        result['object_name'] = self.test_scene[scene_id]["LabelName"]
        return result
    
class XLInpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image_xl(self.img_filenames[i])
        mask= load_mask_xl(self.mask_filenames[i])
        result = dict(image=image, mask=mask) # mask[None, ...] is equivalent to mask[np.newaxis, ...]

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result