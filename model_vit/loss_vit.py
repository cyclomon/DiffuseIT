from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
from model_vit.extractor import VitExtractor
from model_vit.contra_loss import PatchLoss,ConstLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(0, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class Loss_vit(torch.nn.Module):

    def __init__(self, cfg,  lambda_ssim=1.0,lambda_dir_cls=1.0,lambda_contra_ssim=1.0,lambda_trg=0.0):
        super().__init__()

        self.cfg = cfg
        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=device)
        
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])
        self.normalize = Normalize()
        self.lambdas = dict(
            lambda_global_ssim=lambda_ssim,
            lambda_dir_cls=lambda_dir_cls,
            lambda_contra_ssim=lambda_contra_ssim,
            lambda_trg=lambda_trg,
        )
        self.cossim = torch.nn.CosineSimilarity(dim=0)
        self.patch_loss = PatchLoss()
        self.const_loss = ConstLoss()

    def forward(self, outputs, source,out_prev=None,use_dir=True,target=None,frac_cont=1.0):
        losses = {}
        losses_val = {}
        loss_G = 0
        outputs = 0.5*outputs+0.5
        source = 0.5*source+0.5
        if out_prev is not None:
            out_prev = 0.5*out_prev+0.5
        if target is not None:
            target = 0.5*target+0.5
            
        if self.lambdas['lambda_global_ssim'] > 0:
            losses['loss_global_ssim'] = self.calculate_global_ssim_loss(outputs, source)
            loss_G += losses['loss_global_ssim'] * self.lambdas['lambda_global_ssim']
            losses_val['loss_global_ssim'] = losses['loss_global_ssim'].item()
        if self.lambdas['lambda_contra_ssim'] > 0:
            losses['loss_contra_ssim'] = self.calculate_contra_ssim_loss(outputs, source)
            loss_G += losses['loss_contra_ssim'] * self.lambdas['lambda_contra_ssim']*frac_cont
            losses_val['loss_contra_ssim'] = losses['loss_contra_ssim'].item()
        if use_dir:
            if self.lambdas['lambda_dir_cls'] > 0:
                losses['loss_dir_cls'] = self.calculate_dir_cls_loss(outputs, out_prev)
                loss_G += losses['loss_dir_cls'] * self.lambdas['lambda_dir_cls']
                losses_val['loss_dir_cls'] = losses['loss_dir_cls'].item()
        if target is not None:
            if self.lambdas['lambda_trg'] > 0:
                losses['loss_trg'] = self.calculate_target_loss(outputs, target)
                loss_G += losses['loss_trg'] * self.lambdas['lambda_trg']
        
        return loss_G, losses_val

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss
    
    def calculate_dir_cls_loss(self, outputs,out_prev):
        loss = 0.0
        for a, b  in zip(outputs, out_prev):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                prev_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss -= F.mse_loss(prev_cls_token , cls_token)
        return loss
    
    def calculate_target_loss(self, outputs,target):
        loss = 0.0
        for a, b  in zip(outputs, target):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                targ_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :] 
            loss += F.mse_loss(targ_cls_token , cls_token)
        return loss
    def calculate_contra_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
                h, t, d = target_keys.shape
                concatenated_target = target_keys.transpose(0, 1).reshape(t, h * d)
            keys = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            h, t, d = keys.shape
            concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
            
            loss += self.patch_loss(concatenated_keys,concatenated_target).mean()
            
        loss /= len(inputs)
            
        return loss