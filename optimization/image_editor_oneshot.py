import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_blended.metrics_accumulator import MetricsAccumulator
from utils_blended.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from model_vit.loss_histo import RGBuvHistBlock
from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_blended.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path
from id_loss import IDLoss
mean_sig = lambda x:sum(x)/len(x)
class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        if self.args.use_ffhq:
            self.model_config.update(
            {
                "attention_resolutions": "16",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_head_channels": 64,
                "num_res_blocks": 1,
                "resblock_updown": True,
                "use_fp16": False,
                "use_scale_shift_norm": True,
            }
        )
        else:
            self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
        
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        if self.args.use_ffhq:
            self.model.load_state_dict(
                torch.load(
                    "checkpoints/ffhq_10m.pt"
                    map_location="cpu",
                )
            )
            self.idloss = IDLoss().to(self.device)
        else:
            self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config
        self.VIT_LOSS = Loss_vit(cfg,lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg= self.args.lambda_trg).eval()#.requires_grad_(False)
        names = ['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16']
        # init networks
        self.clip_net = CLIPS(names=names, device=self.device, erasing=False)#.requires_grad_(False)
        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].requires_grad_(False)
        )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        
    def clip_loss(self, x_in, text_embed):
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss
    def get_image_prior_losses(self,inputs_jit):
    # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        return loss_var_l2
    def unaugmented_clip_distance(self, x, text_embed):
        x = TF.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    def edit_image_by_prompt(self):
        text_embed = self.clip_model.encode_text(
            clip.tokenize(self.args.prompt).to(self.device)
        ).float()

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        
        self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
        self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.target_image = (
            TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        
        self.prev = self.init_image.detach()
        txt2 = self.args.prompt
        txt1 = self.args.source
        with torch.no_grad():
            self.E_I0 = E_I0 = self.clip_net.encode_image(0.5*self.init_image+0.5, ncuts=0)
            self.E_IT = E_IT = self.clip_net.encode_image(0.5*self.target_image+0.5, ncuts=0)
            
            self.E_S, self.E_T = E_S, E_T =  self.clip_net.encode_text([txt1, txt2])

            self.tgt = (1 * E_IT  ).normalize() #+ *E_I0

            self.imgt = None
            
        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path)
        
        self.mask = torch.ones_like(self.init_image, device=self.device)
        self.mask_pil = None
        if self.args.mask is not None:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            if self.args.invert_mask:
                image_mask_pil_binarized = 255 - image_mask_pil_binarized
                self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)

            if self.args.export_assets:
                mask_path = self.assets_path / Path(
                    self.args.output_file.replace(".png", "_mask.png")
                )
                self.mask_pil.save(mask_path)
        pred = self.clip_net.encode_image(0.5*self.prev+0.5, ncuts=0)
        clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
        # self.loss_prev= clip_loss.clone()
        self.loss_init= clip_loss.detach().clone()
        self.loss_prev = clip_loss.detach().clone()
        self.loss_diff = -1
        self.flag_resample=False
        # self.loss_prev
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        
        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)
            self.flag_resample=False
            with torch.enable_grad():
                lambda_cont = 1.0
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                
                target_stage_t = self.diffusion.q_sample(self.target_image, t[0])
                # background_stage_t = torch.tile(
                #     background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                # )
                # out_bg = self.diffusion.p_mean_variance(
                #     self.model, background_stage_t, t, clip_denoised=False, model_kwargs={"y": y}
                # )
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                # x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in = out["pred_xstart"]
                # print(x_in.min())
                loss = torch.tensor(0)
                if self.args.clip_guidance_lambda != 0:
                    pred = self.clip_net.encode_image(0.5*x_in+0.5, ncuts=8)
                    # out_clip = pred
                    clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
                    # clip_loss  = - (out_clip @ self.tgt.T).flatten().reduce(mean_sig)
                    # clip_loss = self.clip_loss(x_in, text_embed) * self.args.clip_guidance_lambda
                    if t[0] != total_steps:
                        
                        loss = loss + clip_loss*self.args.clip_guidance_lambda*lambda_cont
                        # *(t[0]/total_steps+1)
                    else:
                        loss = loss + clip_loss*self.args.clip_guidance_lambda
                    # self.loss_diff = clip_loss.detach().clone()-self.loss_init
                    # self.loss_diff = clip_loss.detach().clone()
                    # self.loss_diff = clip_loss.detach().clone()-self.loss_prev
                    # print(self.args.iterations_num)
                        self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                # if t[0] == total_steps:
                #     print('iter 0')
                #     print(clip_loss)
                # else:
                #     print('iter X')
                #     print(clip_loss)
                # if t[0] != total_steps:
                if self.args.vit_lambda != 0:
                    if t[0]>self.args.diff_iter:
                        vit_loss,vit_loss_val = self.VIT_LOSS(0.5*x_in+0.5,0.5*self.init_image+0.5,0.5*self.prev+0.5,use_dir=True,target = 0.5*self.target_image+0.5,aug = self.image_augmentations)
                        # vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,target = self.target_image,aug = self.image_augmentations)
                        
                    else:
                        vit_loss,vit_loss_val = self.VIT_LOSS(0.5*x_in+0.5,0.5*self.init_image+0.5,0.5*self.prev+0.5,use_dir=False,target = 0.5*self.target_image+0.5,aug = self.image_augmentations)
                        # vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,target = self.target_image,aug = self.image_augmentations)
                    # clip_loss = self.clip_loss(x_in, text_embed) * self.args.clip_guidance_lambda
                    loss = loss + vit_loss
                    # self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                if self.args.lpips_sim_lambda:
                    # loss = loss+ self.lpips_model(x_in, self.target_image).sum() * self.args.lpips_sim_lambda
                    loss = loss + self.histo_loss(x_in,self.target_image) *500
                if self.args.l2_sim_lambda:
                    loss = loss + mse_loss( out["pred_xstart"], self.target_image) * self.args.l2_sim_lambda
                    # loss = loss + self.smooth_l1(out["pred_xstart"], self.target_image) * self.args.l2_sim_lambda
                    # loss = loss + 1e-4*self.get_image_prior_losses(x_in)
                    # loss = loss + self.histo_loss(out["pred_xstart"],self.target_image) *500

                if self.args.l2_inter:
                    loss = loss + mse_loss(out["pred_xstart"], out_bg["pred_xstart"]) * self.args.l2_inter

                # self.loss_prev = clip_loss.detach().clone()

                self.prev = x_in.detach().clone()
                if self.args.use_ffhq:
                    loss =  loss + self.idloss(x_in,self.init_image) * self.args.id_lambda
                # print(r_loss)
                # if self.loss_diff > -0.004 and r_loss > 0.001 :
                # print(self.loss_diff)
                if self.args.ddim:
                    crit = -0.005
                else:
                    # crit = -0.005
                    crit = -0.005
                    
                # if t[0].item() < total_steps-1:
                #     if r_loss>0.01:
                #         self.flag_resample =True
                if self.args.lambda_contra_ssim >0:
                    if t[0].item() == total_steps:
                        if vit_loss_val["loss_contra_ssim"] >4:
                            self.flag_resample =True    
                
                # print(
                # crit = self.args.criterion_resample
                # if self.loss_diff > -crit or r_loss > 0.001  :
                # print(vit_loss_val["loss_contra_ssim"])
                # if vit_loss_val["loss_contra_ssim"] <3.5:
                #     lambda_cont = 2
                # if clip_loss<-0:
                
                if t[0].item() < total_steps-1:
                    if r_loss>0.01:
                        self.flag_resample =True
                
            # or 
                    
                    # print(t[0].item())
                    # if t[0].item() == total_steps-2:
                    # if t[0].item() == total_steps:
                    #     # print("resample")
                    #     self.flag_resample =True
                # self.flag_resample=False
            return -torch.autograd.grad(loss, x)[0], self.flag_resample

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")
    
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            # print(self.args.ddim)
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None,
                randomize_class=True,
            )
            # print(self.flag_resample)
            if self.flag_resample:
                continue
                
            
            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_stem(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}"
                        )

                        # if (
                        #     self.mask is not None
                        #     and self.args.enforce_background
                        #     and j == total_steps
                        #     and not self.args.local_clip_guided_diffusion
                        # ):
                        #     pred_image = (
                        #         self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                        #     )
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)
                        masked_pred_image = self.mask * pred_image.unsqueeze(0)
                        final_distance = self.unaugmented_clip_distance(
                            masked_pred_image, text_embed
                        )
                        formatted_distance = f"{final_distance:.4f}"

#                         
#                     if self.args.animals:
#                         img_name = self.args.source + '2' +self.args.prompt +'.jpg'
#                         path_animal = self.args.output_path_ani+ self.args.prompt
#                         Path(path_animal).mkdir(parents=True, exist_ok=True)
                        
#                         final_path = os.path.join(path_animal, img_name)
#                         pred_image_pil.save(final_path)
#                     else:
#                         if self.args.export_assets:
#                             pred_path = self.assets_path / visualization_path.name
#                             pred_image_pil.save(pred_path)

#                         if j == total_steps:
            path_friendly_distance = formatted_distance.replace(".", "")
            ranked_pred_path = self.ranked_results_path / (
                path_friendly_distance + "_" + visualization_path.name
            )
            pred_image_pil.save(ranked_pred_path)

                        # intermediate_samples[b].append(pred_image_pil)
                        # if should_save_image:
                        #     show_editied_masked_image(
                        #         title=self.args.prompt,
                        #         source_image=self.init_image_pil,
                        #         edited_image=pred_image_pil,
                        #         mask=self.mask_pil,
                        #         path=visualization_path,
                        #         # distance=formatted_distance,
                        #     )
            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.avi"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)

    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)

        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256
            else {"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)
