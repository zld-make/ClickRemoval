import os
from argparse import ArgumentParser
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from AAS.AAS_utils import regiter_attention_editor_diffusers
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from AAS.AAS import AAS
#from evaluation.data import InpaintingDataset, move_to_device
from AAS.data import move_to_device
import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageFilter
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
torch.cuda.set_device(0)  # set the GPU device
torch.set_grad_enabled(False)

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
    mask = gaussian_blur(mask, kernel_size=(7, 7))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(torch.float16)
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


def main(args):

    config = OmegaConf.load(args.config)
    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_path = config.model.path 
    custom_pipeline = config.model.pipeline_path
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        custom_pipeline=custom_pipeline,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    #freeU
    from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
    register_free_upblock2d(pipe, b1=1.4, b2=1.6, s1=0.9, s2=0.2)
    register_free_crossattn_upblock2d(pipe, b1=1.4, b2=1.6, s1=0.9, s2=0.2)
    
    out_ext = config.get('out_ext', '.png')
    out_suffix = config.out_suffix
    seed = config.seed
    
    if not config.dataset.datadir.endswith('/'):
        config.dataset.datadir += '/'
    dataset = InpaintingDataset(**config.dataset)

    strength = 1
    num_inference_steps = 50
    START_STEP = 0
    END_STEP = int(strength*num_inference_steps)
    LAYER = 7 #0~5down,6mid,7~15up
    END_LAYER = 16
    #removelist=[6]
    layer_idx=list(range(LAYER, END_LAYER))
    #for i in removelist:
    #    layer_idx.remove(i)
        
    for img_i in tqdm.trange(len(dataset)):
        seed_everything(seed)
        generator=torch.Generator("cuda").manual_seed(seed)
        img_fname = dataset.img_filenames[img_i]
        mask_fname = dataset.mask_filenames[img_i]
        cur_out_fname = os.path.join(
                config.outdir, 
                os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + out_suffix + str(seed) + out_ext                                                       
            )
        
        cur_out_fname_ori = os.path.join(
                config.outdir, 
                os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + "_ori_" + str(seed) + out_ext                                                       
            )
        
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        cur_img_fname = os.path.join(
            config.outdir, 
            os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + out_ext                                                                     
        )
        cur_mask_fname = os.path.join(
            config.outdir, 
            os.path.splitext(img_fname[len(config.dataset.datadir):])[0] + "_mask"+ out_ext                                                                     
        )
        #batch = default_collate([dataset[img_i]])
        batch = dataset[img_i]
        batch = move_to_device(batch, device)
        prompt = ""
        start_code, latents_list = pipe.invert(
        #start_code, x0 = pipe.invert(
                                    batch['image'],
                                    prompt,
                                    generator=generator,
                                    guidance_scale=1,
                                    num_inference_steps=50,
                                    return_intermediates=True)

        # inference the synthesized image with MyREMOVAL
        # hijack the attention module
        attentionstore = None
        editor = AAS(attentionstore, START_STEP, END_STEP, layer_idx= layer_idx, mask=batch['mask'], ss_steps=9, ss_scale=0.3)
        regiter_attention_editor_diffusers(pipe, editor)

        #image_s = Image.open(img_fname).convert('RGB')
        #mask = Image.open(mask_fname)
        #image, pred_x0_list_denoise, latents_list_denoise = pipe(
        image = pipe(
                    prompt,
                    width=512,
                    height=512,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=1.0,
                    latents=start_code,
                    generator=generator,
                    x0_latents=latents_list[0],
                    record_list = list(reversed(latents_list)),
                    mask = batch['mask'],
                    rm_guidance_scale=9,
                    return_intermediates = False)

        if config.save.tile == True:
            pil_mask = to_pil_image(batch['mask'].squeeze(0))
            pil_mask_blurred = pil_mask.filter(ImageFilter.GaussianBlur(radius=15))
            mask_blurred = to_tensor(pil_mask_blurred).unsqueeze_(0).to(batch['mask'].device)
            msak_f = 1-(1-batch['mask'])*(1-mask_blurred)
            out_tile = msak_f * image[-1:] + (1 - msak_f) * (batch['image']* 0.5 + 0.5)
            save_image(out_tile, cur_out_fname)

        if config.save.result == True:
            save_image(image[-1], cur_out_fname_ori)
        if config.save.mask == True:
            save_image(batch['mask'], cur_mask_fname)
        if config.save.resize_img == True:
            save_image(batch['image']* 0.5 + 0.5, cur_img_fname)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to inference config')
    main(parser.parse_args())

    