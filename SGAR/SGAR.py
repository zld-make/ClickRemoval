import os
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .AAS_utils import AttentionBase
from torchvision.utils import save_image

class SGA_SD(AttentionBase):
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    def __init__(
            self,
            start_step=4,
            end_step=50,
            start_layer=10,
            end_layer=16,
            layer_idx=None,
            step_idx=None,
            total_steps=50,
            M=None,
            markov_map=None,
            model_type="SD",
            sg_steps=9,
            sg_scale=1.0,
            tfg_schedule_type="increase",  # "decrease", "increase", "constant", "parabolic", "step"
            tfg_base_scale=1.0,
            use_tfg_schedule=True,
    ):
        """
        Args:
            start_step: the step to start SGA
            start_layer: the layer to start SGA
            layer_idx: list of the layers to apply SGA
            step_idx: list the steps to apply SGA
            total_steps: the total number of steps
            M: source M with shape (h, w)
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
        self.M = M  # M with shape (1, 1 ,h, w)
        self.sg_steps = sg_steps
        self.sg_scale = sg_scale
        self.use_tfg_schedule = use_tfg_schedule
        if use_tfg_schedule:
            self.tfg_schedule = TDGSchedule(
                sg_steps=sg_steps,
                schedule_type=tfg_schedule_type,
                base_scale=tfg_base_scale
            )
        self.mask_8 = F.max_pool2d(M, (512 // 8, 512 // 8)).round().squeeze().squeeze()
        self.mask_16 = F.max_pool2d(M, (512 // 16, 512 // 16)).round().squeeze().squeeze()
        self.mask_32 = F.max_pool2d(M, (512 // 32, 512 // 32)).round().squeeze().squeeze()
        self.mask_64 = F.max_pool2d(M, (512 // 64, 512 // 64)).round().squeeze().squeeze()
        self.mask_128 = F.max_pool2d(M, (512 // 128, 512 // 128)).round().squeeze().squeeze()
        self.markov_map_8 = F.max_pool2d(markov_map, (512 // 8, 512 // 8),
                                          ceil_mode=True).round().squeeze().squeeze()
        self.markov_map_16 = F.max_pool2d(markov_map, (512 // 16, 512 // 16),
                                          ceil_mode=True).round().squeeze().squeeze()
        self.markov_map_32 = F.max_pool2d(markov_map, (512 // 32, 512 // 32),
                                          ceil_mode=True).round().squeeze().squeeze()
        self.markov_map_64 = F.max_pool2d(markov_map, (512 // 64, 512 // 64),
                                          ceil_mode=True).round().squeeze().squeeze()
        self.markov_map_128 = F.max_pool2d(markov_map, (512 // 128, 512 // 128),
                                           ceil_mode=True).round().squeeze().squeeze()

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, is_mask_attn, M, markov_flatten,
                   **kwargs):
        B = q.shape[0] // num_heads
        dtype = sim.dtype
        if is_mask_attn:
            mask_flatten = M.flatten(0)
            if self.cur_step <= self.sg_steps:
                if self.use_tfg_schedule:
                    schedule_value = self.tfg_schedule.get_schedule(self.cur_step)
                else:
                    schedule_value = self.sg_scale

                sim_bg = sim + mask_flatten.masked_fill(mask_flatten >= 1, -20)
                markov_reshaped = markov_flatten.view(1, 1, -1)  # (1, 1, N)
                bg_key_mask = (mask_flatten == 0).float().view(1, 1, -1)  # (1, 1, N)
                suppression_factors = 1 - markov_reshaped * (1 - schedule_value)
                suppression_factors = bg_key_mask * suppression_factors + (1 - bg_key_mask) * 1.0
                sim_fg = sim * suppression_factors
                sim_fg += mask_flatten.masked_fill(mask_flatten >= 1, -20.0)
                sim = torch.cat([sim_fg, sim_bg], dim=0)
            else:
                sim += mask_flatten.masked_fill(mask_flatten >= 1, -20.0)

        attn = sim.softmax(-1).to(dtype=dtype)
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
        # B = q.shape[0] // num_heads // 2
        H = int(np.sqrt(q.shape[1]))
        # H = W = int(np.sqrt(q.shape[1]))
        # print(f"H={H}, q.shape={q.shape}")
        if H == 8:
            M = self.mask_8.to(sim.device)
            markov_map = self.markov_map_8.to(sim.device) if hasattr(self, 'markov_map_8') else None
        elif H == 16:
            M = self.mask_16.to(sim.device)
            markov_map = self.markov_map_16.to(sim.device) if hasattr(self, 'markov_map_16') else None
        elif H == 32:
            M = self.mask_32.to(sim.device)
            markov_map = self.markov_map_32.to(sim.device) if hasattr(self, 'markov_map_32') else None
        elif H == 64:
            M = self.mask_64.to(sim.device)
            markov_map = self.markov_map_64.to(sim.device) if hasattr(self, 'markov_map_64') else None
        elif H == 128:
            M = self.mask_128.to(sim.device)
            markov_map = self.markov_map_128.to(sim.device) if hasattr(self, 'markov_map_128') else None


        # print("SGA begin")
        q_wo, q_w = q.chunk(2)
        k_wo, k_w = k.chunk(2)
        v_wo, v_w = v.chunk(2)
        sim_wo, sim_w = sim.chunk(2)
        attn_wo, attn_w = attn.chunk(2)

        out_source = self.attn_batch(
            q_wo,
            k_wo,
            v_wo,
            sim_wo,
            attn_wo,
            is_cross,
            place_in_unet,
            num_heads,
            is_mask_attn=False,
            M=None,
            markov_flatten=None,
            **kwargs,
        )

        markov_flatten = None
        if markov_map is not None:
            markov_flatten = markov_map.flatten(0).to(sim.device)
        out_target = self.attn_batch(
            q_w, k_w, v_w, sim_w, attn_w, is_cross, place_in_unet, num_heads,
            is_mask_attn=True, M=M, markov_flatten=markov_flatten, **kwargs
        )

        if self.M is not None:
            if out_target.shape[0] == 2:
                out_target_fg, out_target_bg = out_target.chunk(2, 0)
                M = M.reshape(-1, 1)  # (hw, 1)
                out_target = out_target_fg * M + out_target_bg * (1 - M)
            else:
                out_target = out_target

        out = torch.cat([out_source, out_target], dim=0)
        return out

class TDGSchedule:
    def __init__(self, sg_steps, schedule_type="increase", base_scale=1.0):
        self.sg_steps = sg_steps
        self.schedule_type = schedule_type
        self.base_scale = base_scale

        self.schedule_cache = self._precompute_schedule()

    def _precompute_schedule(self):
        schedule = {}
        for t in range(self.sg_steps):
            normalized_t = t / self.sg_steps
            schedule[t] = self._compute_schedule_value(normalized_t)
        return schedule

    def _compute_schedule_value(self, normalized_t):
        if self.schedule_type == "increase":
            return normalized_t * self.base_scale
        elif self.schedule_type == "decrease":
            return (1 - normalized_t) * self.base_scale
        elif self.schedule_type == "constant":
            return self.base_scale
        elif self.schedule_type == "parabolic":
            return 4 * normalized_t * (1 - normalized_t) * self.base_scale
        elif self.schedule_type == "step":
            if normalized_t < 0.3:
                return 0.3 * self.base_scale
            elif normalized_t < 0.7:
                return 0.7 * self.base_scale
            else:
                return self.base_scale
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def get_schedule(self, t):
        return self.schedule_cache.get(t, self.base_scale)