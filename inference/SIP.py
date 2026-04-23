import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AAS.AAS_utils import regiter_attention_editor_diffusers
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from AAS.AAS import AAS
from AAS.data import InpaintingDataset, move_to_device
import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageFilter
torch.cuda.set_device(0)  # set the GPU device
torch.set_grad_enabled(False)

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

    strength = 0.8
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
        attentionstore = None
        # inference the synthesized image with MyREMOVAL
        # hijack the attention module
        editor = AAS(attentionstore,START_STEP, END_STEP, layer_idx= layer_idx, mask=batch['mask'], ss_steps=9, ss_scale=0.3)
        regiter_attention_editor_diffusers(pipe, editor)

        #image_s = Image.open(img_fname).convert('RGB')
        #mask = Image.open(mask_fname)
        #image, pred_x0_list_denoise, latents_list_denoise = pipe(
        image = pipe(
                    prompt=prompt, 
                    image=batch['image'], 
                    mask_image=batch['mask'],
                    num_inference_steps = num_inference_steps,
                    strength=strength,
                    generator=generator, 
                    rm_guidance_scale=9,
                    guidance_scale = 1,
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

    