import math
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import ChannelDimension
from transformers.models.dinov2.modeling_dinov2 import Dinov2SdpaSelfAttention, Dinov2SdpaAttention


class Dinov2SdpaSelfAttention_Wrapper(nn.Module):
    def __init__(self, other: Dinov2SdpaSelfAttention, path=None, callback_func=None) -> None:
        super().__init__()
        self.other = other

        # Adding our own members for tracking
        self.path = path
        self.wrapper_callback_func = callback_func

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        """
        Even though this is the exact implementation given by torch.nn.functional.scaled_dot_product_attention
        the results using DinoV2 still changes about a MAE of about 1e-5. So there are is some small numerical difference as
        PyTorch uses a highly optimized routine which does the same as their provided code.
        :return:
        """
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

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)

        # Calling callback (we don't want dropout, so we do this before dropout is applied)
        if self.wrapper_callback_func is not None:
            attn_weight = self.wrapper_callback_func(self.path, attn_weight)

        # Continue as normal
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.other.query(hidden_states)

        key_layer = self.other.transpose_for_scores(self.other.key(hidden_states))
        value_layer = self.other.transpose_for_scores(self.other.value(hidden_states))
        query_layer = self.other.transpose_for_scores(mixed_query_layer)

        context_layer = self.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.other.attention_probs_dropout_prob if self.other.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.other.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer, None


def dinov2_inject_attention_wrappers(module, callback_func=None, path='', collect_wrappers=None) -> dict:
    """
    module: stable diffusion pipe.unet as input
    callback_func: Will be called whenever the self attention is used. Parameters are (path: str, x: torch.Tensor) and expects to return a
        torch.Tensor in the same shape, device and dtype as the given x. (to replace the current attention. Easiest is just to return original x to not modify)
    """
    if collect_wrappers is None:
        collect_wrappers = dict()

    if isinstance(module, Dinov2SdpaAttention):
        module.attention = Dinov2SdpaSelfAttention_Wrapper(module.attention, path=path, callback_func=callback_func)
        collect_wrappers[path] = module.attention
        return collect_wrappers
    elif hasattr(module, 'children'):
        for k, v in list(module.named_children()):
            dinov2_inject_attention_wrappers(v, callback_func, path + '.' + k, collect_wrappers)
    return collect_wrappers


class DinoV2AttentionAggregator(object):
    def __init__(self,
                 hugging_face_model_id='facebook/dinov2-base',
                 attention_resolution=80,
                 device='cuda:0',
                 weight_layer_0=0,
                 weight_layer_1=0,
                 weight_layer_2=0,
                 weight_layer_3=0,
                 weight_layer_4=0,
                 weight_layer_5=0,
                 weight_layer_6=0,
                 weight_layer_7=0,
                 weight_layer_8=0,
                 weight_layer_9=0,
                 weight_layer_10=0,
                 weight_layer_11=1,
                 torch_dtype=torch.float16):
        self.attention_resolution = attention_resolution
        self.weight_layer_0 = weight_layer_0
        self.weight_layer_1 = weight_layer_1
        self.weight_layer_2 = weight_layer_2
        self.weight_layer_3 = weight_layer_3
        self.weight_layer_4 = weight_layer_4
        self.weight_layer_5 = weight_layer_5
        self.weight_layer_6 = weight_layer_6
        self.weight_layer_7 = weight_layer_7
        self.weight_layer_8 = weight_layer_8
        self.weight_layer_9 = weight_layer_9
        self.weight_layer_10 = weight_layer_10
        self.weight_layer_11 = weight_layer_11
        self.device = device
        self.dtype = torch_dtype
        self.processor = AutoImageProcessor.from_pretrained(hugging_face_model_id)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base', torch_dtype=torch_dtype).to(device)
        self.attention_wrappers = dinov2_inject_attention_wrappers(self.model, callback_func=self.collect_attention_tensors_callback)
        self.path_to_weight_dict = dict()
        for attn_key in self.attention_wrappers.keys():
            layer_idx = int(attn_key.split('.')[3])
            weight = getattr(self, f'weight_layer_{layer_idx}')
            self.path_to_weight_dict[attn_key] = weight

    def collect_attention_tensors_callback(self, path, x: torch.Tensor):
        # Get the weight of this tensor
        weight = self.path_to_weight_dict[path] if path in self.path_to_weight_dict else 0
        if weight == 0:
            return x

        # Removing 'extra learnable "classification token"', see "AN IMAGE IS WORTH 16X16 WORDS" paper
        attn = x[:, :, 1:, 1:]

        # Average heads and batch and then reshape
        attn = torch.mean(torch.mean(attn.float(), dim=1), dim=0)
        width = int(round(math.sqrt(attn.shape[-1])))
        attn = attn.reshape(width, width, width, width)

        # We sum up directly to use less memory
        if self.current_merged_tensor is None:
            self.current_merged_tensor = attn * weight
        else:
            self.current_merged_tensor = self.current_merged_tensor + attn * weight

        # Return original unchanged x
        return x

    def preprocess_image(self, img: np.ndarray):
        img = cv2.resize(img, (self.attention_resolution * 14, self.attention_resolution * 14), interpolation=cv2.INTER_AREA)
        img = self.processor.rescale(image=img, scale=self.processor.rescale_factor, input_data_format=ChannelDimension.LAST)
        img = self.processor.normalize(image=img, mean=self.processor.image_mean, std=self.processor.image_std, input_data_format=ChannelDimension.LAST)
        return torch.tensor(img, dtype=self.dtype, device=self.device).permute((2, 0, 1))[None]

    def extract_attention(self, image: np.ndarray) -> torch.Tensor:
        """
        :param image: Image of arbitrary size in the shape of (H, W, 3) and dtype np.uint8
        :return: Attention Tensor in the shape of (h, w, h, w) with all elements >= 0 and the last 2 axis summing up to 1
        """
        self.current_merged_tensor = None
        with torch.no_grad():
            self.model(self.preprocess_image(image))
        out = self.current_merged_tensor / torch.sum(self.current_merged_tensor.reshape(self.attention_resolution, self.attention_resolution, -1), dim=2)[:, :, None, None]
        return out


def main():
    image_rgb = cv2.imread('../images/image.jpg')[:, :, ::-1]
    aggr = DinoV2AttentionAggregator()
    print(aggr.extract_attention(image_rgb).shape)


if __name__ == '__main__':
    main()
