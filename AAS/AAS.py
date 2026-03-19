import os
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .AAS_utils import AttentionBase
from torchvision.utils import save_image

class AAS(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self, attnstore=None,start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD",ss_steps=9,ss_scale=1.0):
        """
        Args:
            start_step: the step to start AAS
            start_layer: the layer to start AAS
            layer_idx: list of the layers to apply AAS
            step_idx: list the steps to apply AAS
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.attnstore = attnstore
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        self.ss_steps = ss_steps
        self.ss_scale = ss_scale
        print("AAS at denoising steps: ", self.step_idx)
        print("AAS at U-Net layers: ", self.layer_idx)
        print("start AAS")
        self.mask_8 = F.max_pool2d(mask,(512//8,512//8)).round().squeeze().squeeze()
        self.mask_16 = F.max_pool2d(mask,(512//16,512//16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(mask,(512//32,512//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(512//64,512//64)).round().squeeze().squeeze()
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, **kwargs):
        B = q.shape[0] // num_heads
        if is_mask_attn:
            mask_flatten = mask.flatten(0)                
            if self.cur_step <= self.ss_steps:                                                                                                                                                                                                                                                               
                # background
                sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) 
                #object
                sim_fg = self.ss_scale*sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

                sim = torch.cat([sim_fg, sim_bg], dim=0)

            else:
                sim += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

        attn = sim.softmax(-1)
        ## attn store
        if self.attnstore is not None:
            self.attnstore(attn,is_cross,place_in_unet,self.cur_step)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if H == 16:
            mask = self.mask_16.to(sim.device)
        elif H == 32:
            mask = self.mask_32.to(sim.device)
        elif H == 8:
            mask = self.mask_8.to(sim.device)
        else:
            mask = self.mask_64.to(sim.device)

        
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask, **kwargs)

        if self.mask is not None:
            if out_target.shape[0] == 2:
                out_target_fg, out_target_bg = out_target.chunk(2, 0)
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_target = out_target_fg * mask + out_target_bg * (1 - mask)
            else:
                out_target = out_target
        
        out = torch.cat([out_source, out_target], dim=0)
        return out


class AAS_768(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD",ss_steps=9,ss_scale=1.0):
        """
        Args:
            start_step: the step to start AAS
            start_layer: the layer to start AAS
            layer_idx: list of the layers to apply AAS
            step_idx: list the steps to apply AAS
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        self.ss_scale = ss_scale
        self.ss_steps = ss_steps
        print("AAS at denoising steps: ", self.step_idx)
        print("AAS at U-Net layers: ", self.layer_idx)
        print("start AAS")
        self.mask_12 = F.max_pool2d(mask,(768//12,768//12)).round().squeeze().squeeze()
        self.mask_24 = F.max_pool2d(mask,(768//24,768//24)).round().squeeze().squeeze()
        self.mask_48 = F.max_pool2d(mask,(768//48,768//48)).round().squeeze().squeeze()
        self.mask_96 = F.max_pool2d(mask,(768//96,768//96)).round().squeeze().squeeze()
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, **kwargs):
        B = q.shape[0] // num_heads
        if is_mask_attn:
            mask_flatten = mask.flatten(0)                 
            if self.cur_step <= self.ss_steps:
                                                                                                                                                                                                                                                                                 
                # background
                sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) 
                #object
                sim_fg = self.ss_scale*sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
                sim = torch.cat([sim_fg, sim_bg], dim=0)

            else:
                sim += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        
        if H == 24:
            mask = self.mask_24.to(sim.device)
        elif H == 48:
            mask = self.mask_48.to(sim.device)
        elif H == 12:
            mask = self.mask_12.to(sim.device)
        else:
            mask = self.mask_96.to(sim.device)

        
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask, **kwargs)

        if self.mask is not None:
            if out_target.shape[0] == 2:
                out_target_fg, out_target_bg = out_target.chunk(2, 0)
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_target = out_target_fg * mask + out_target_bg * (1 - mask)
            else:
                out_target = out_target
        
        out = torch.cat([out_source, out_target], dim=0)
        return out
        


class AAS_XL(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self,  start_step=4, end_step= 50, start_layer=10, end_layer=16,layer_idx=None, step_idx=None, total_steps=50,  mask=None, mask_save_dir=None, model_type="SD",ss_steps=9,ss_scale=1.0):
        """
        Args:
            start_step: the step to start AAS
            start_layer: the layer to start AAS
            layer_idx: list of the layers to apply AAS
            step_idx: list the steps to apply AAS
            total_steps: the total number of steps
            mask: source mask with shape (h, w)
            mask_save_dir: the path to save the mask image
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.end_step = end_step
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.mask = mask  # mask with shape (1, 1 ,h, w)
        self.ss_steps = ss_steps
        self.ss_scale = ss_scale
        print("AAS at denoising steps: ", self.step_idx)
        print("AAS at U-Net layers: ", self.layer_idx)
        print("start AAS")
        self.mask_16 = F.max_pool2d(mask,(1024//16,1024//16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(mask,(1024//32,1024//32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(mask,(1024//64,1024//64)).round().squeeze().squeeze()
        self.mask_128 = F.max_pool2d(mask,(1024//128,1024//128)).round().squeeze().squeeze()

        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask.png"))
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,is_mask_attn, mask, **kwargs):
        B = q.shape[0] // num_heads
        if is_mask_attn:
            mask_flatten = mask.flatten(0)
            if self.cur_step <= self.ss_steps:                                                                                                                                                                                                                                           
                # background
                sim_bg = sim + mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min) 

                #object
                sim_fg = self.ss_scale*sim
                sim_fg += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)
                sim = torch.cat([sim_fg, sim_bg], dim=0)
            else:
                sim += mask_flatten.masked_fill(mask_flatten == 1, torch.finfo(sim.dtype).min)

        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        if H == 16:
            mask = self.mask_16.to(sim.device)
        elif H == 32:
            mask = self.mask_32.to(sim.device)
        elif H == 64:
            mask = self.mask_64.to(sim.device)
        else:
            mask = self.mask_128.to(sim.device)


        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(q_wo, k_wo, v_wo, sim_wo, attn_wo, is_cross, place_in_unet, num_heads,is_mask_attn=False,mask=None,**kwargs)
        out_target = self.attn_batch(q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads, is_mask_attn=True, mask = mask,**kwargs)

        if self.mask is not None:
            if out_target.shape[0] == 2:
                out_target_fg, out_target_bg = out_target.chunk(2, 0)
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_target = out_target_fg * mask + out_target_bg * (1 - mask)
            else:
                out_target = out_target
        
        out = torch.cat([out_source, out_target], dim=0)
        return out

