from math import sqrt
import numpy as np
import math

import cv2
from diffusers import UNet2DConditionModel
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import deprecate
from typing import Optional
from diffusers.models.attention_processor import AttnProcessor2_0

class AttnProcessor2_0Wrapper(AttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, other, path=None, callback_func=None):
        super().__init__()

        # copy all members of the class we want to wrap
        self.__dict__ = other.__dict__.copy()

        # Adding our own members for tracking
        self.path = path
        self.wrapper_callback_func = callback_func

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Calling callback (we don't want dropout, so we do this before dropout is applied)
        if self.wrapper_callback_func is not None:
            attn_weight = self.wrapper_callback_func(self.path, attn_weight)

        # Continue as normal
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def sd2_inject_attention_wrappers(module, callback_func=None, path='', collect_wrappers=None) -> dict:
    """
    module: stable diffusion pipe.unet as input
    callback_func: Will be called whenever the self attention is used. Parameters are (path: str, x: torch.Tensor) and expects to return a
        torch.Tensor in the same shape, device and dtype as the given x. (to replace the current attention. Easiest is just to return original x to not modify)
    """
    if collect_wrappers is None:
        collect_wrappers = dict()

    if isinstance(module, Attention):
        if not isinstance(module.processor, AttnProcessor2_0):
            print('Can not track this type of attention processor', module.processor.__class__.__name__)
            print('Expected AttnProcessor2_0')
            exit()
        module.set_processor(AttnProcessor2_0Wrapper(module.processor, path=path, callback_func=callback_func))
        collect_wrappers[path] = module.processor
        return collect_wrappers
    elif hasattr(module, 'children'):
        for k, v in list(module.named_children()):
            sd2_inject_attention_wrappers(v, callback_func, path + '.' + k, collect_wrappers)
    return collect_wrappers


def sd2_perform_single_image_diffusion_step(pipe, img: np.ndarray, timestep, device, torch_dtype):
    """
    img: (H, W, 3) dtype is np.uint8. H and W should be divisible by 64
    callback_func: expects 2 parameters (path, x: torch.Tensor). It should return an attention tensor in the same shape, dtype and on the same device as x.
        if it is not None, None of the tensors will be tracked
    """

    # prompt_embeds shape [1, 77, 1024], (batch_size, ?, resolution_h, resolution_w)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt('', device, 1, False)
    preprocessed_image = pipe.image_processor.preprocess(img / 255).to(torch_dtype).to(device)

    # init_latents shape [1, 4, 96, 96], (batch_size, ?, resolution_h, resolution_w)
    init_latents = pipe.vae.config.scaling_factor * pipe.vae.encode(preprocessed_image).latent_dist.mode()

    with torch.no_grad():
        # Do one iteration
        _ = pipe.unet(
            init_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]


class StableDiffusionxlAttentionAggregator(object):
    def __init__(self,
                 timestep=100,
                 attention_resolution=128,
                 weight_down_block_0=0,
                 weight_down_block_1=0,
                 weight_up_block_0=0.5,
                 weight_up_block_1=0.5,
                 weight_up_block_2=0,
                 #hugging_face_model_id="stabilityai/stable-diffusion-2",
                 device='cuda:0',
                 torch_dtype=torch.float16):
        self.stable_diffusion_img_size = (8 * attention_resolution, 8 * attention_resolution)
        self.attn_target_resolution = (attention_resolution, attention_resolution)
        self.current_merged_tensor = None
        self.timestep = timestep
        self.device = device
        self.torch_dtype = torch_dtype

        # Load pipe and setup callbacks
        local_model_path = "/root/autodl-tmp/stable-diffusion-xl-base-1.0/unet"
        self.pipe = UNet2DConditionModel.from_pretrained(
            local_model_path,
            torch_dtype="fp16",
            local_files_only=True,
            use_safetensors=True,
        ).to(device)

        self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        self.attention_wrappers = sd2_inject_attention_wrappers(self.pipe.unet, callback_func=self.collect_attention_tensors_callback)
        for path in self.attention_wrappers.keys():
            print(path)

        self.path_to_weight_dict = dict()
        for attn_key in self.attention_wrappers.keys():
            if attn_key == '.down_blocks.0.attentions.0.transformer_blocks.0.attn1':
                weight = weight_down_block_0
            elif attn_key == '.down_blocks.0.attentions.1.transformer_blocks.0.attn1':
                weight = weight_down_block_1
            elif attn_key == '.up_blocks.3.attentions.0.transformer_blocks.0.attn1':
                weight = weight_up_block_0
            elif attn_key == '.up_blocks.3.attentions.1.transformer_blocks.0.attn1':
                weight = weight_up_block_1
            elif attn_key == '.up_blocks.3.attentions.2.transformer_blocks.0.attn1':
                weight = weight_up_block_2
            else:
                weight = 0
            self.path_to_weight_dict[attn_key] = weight

    def collect_attention_tensors_callback(self, path, x: torch.Tensor):

        # Get the weight of this tensor
        weight = self.path_to_weight_dict[path] if path in self.path_to_weight_dict else 0
        if weight == 0:
            return x

        # Average heads and batch and then reshape
        attn = torch.mean(torch.mean(x.float(), dim=1), dim=0)
        width = int(round(sqrt(attn.shape[-1])))
        attn = attn.reshape(width, width, width, width)

        # We sum up directly to use less memory
        if self.current_merged_tensor is None:
            self.current_merged_tensor = attn * weight
        else:
            self.current_merged_tensor = self.current_merged_tensor + attn * weight

        # Return original unchanged x
        return x

    def extract_attention(self, image: np.ndarray) -> torch.Tensor:
        """
        :param image: Image of arbitrary size in the shape of (H, W, 3) and dtype np.uint8
        :return: Attention Tensor in the shape of (h, w, h, w) with all elements >= 0 and the last 2 axis summing up to 1
        """
        self.current_merged_tensor = None
        sd2_perform_single_image_diffusion_step(
            pipe=self.pipe,
            img=cv2.resize(image, self.stable_diffusion_img_size, interpolation=cv2.INTER_AREA),
            timestep=self.timestep,
            device=self.device,
            torch_dtype=self.torch_dtype
        )
        out = self.current_merged_tensor / torch.sum(self.current_merged_tensor.reshape(self.attn_target_resolution[1], self.attn_target_resolution[0], -1), dim=2)[:, :, None, None]
        return out


def main():
    image_rgb = cv2.imread('../images/image.jpg')[:, :, ::-1]
    aggr = StableDiffusionxlAttentionAggregator()
    print(aggr.extract_attention(image_rgb).shape)


if __name__ == '__main__':
    main()
