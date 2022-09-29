# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import yaml
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import clip
# sys.path.append('taming-transformers')
# from taming.models.vqgan import VQModel

from utils_flexit.torch_utils import TensorDict
from pathlib import Path
from .template import imagenet_templates

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def soft_clamp(x, beta=5):
    x = 2*x - 1
    x = torch.tanh(x.pow(beta))
    x = x.relu().pow(1/beta) - (-x).relu().pow(1/beta)
    return (x+1)/2

# class VQGAN(VQModel):
#     def __init__(self, device='cpu', load_loss=False, root='./vq_model'):
#         with open(f'{root}/model.yaml') as f:
#             cfg = yaml.safe_load(f)['model']['params']
#         super().__init__(**cfg)
#         self.load_state_dict(torch.load(f'{root}/last.ckpt', map_location=device)['state_dict'], strict=False)
#         if not load_loss:
#             self.loss = None
        
    
#     def forward(self, x):
#         x = 2*x - 1
#         z, _, [_, _, indices] = super().encode(x)
#         grid = int(math.sqrt(indices.shape[0]/x.shape[0]))
#         return z, indices.flatten().reshape(x.shape[0], grid, grid)
    
#     def encode(self, x):
#         # avoid saturation
#         if len(x.shape) < 4:
#             x = x[None, :]
#         x = 2*x - 1
#         x = x.to(self.quant_conv.weight.device)
#         x = self.encoder(x)
#         x = self.quant_conv(x)
#         return x
    
#     def decode(self, x, clamp=True):
#         x = x.to(self.quant_conv.weight.device)
#         x = VQModel.decode(self, x)
#         x = (x+1)/2
#         if clamp:
#             x = soft_clamp(x)
#         return x

class CLIP:
    norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
    
    def __init__(self, name, device='cpu', erasing=False):
        self.name = name
        self.device = device
        self.model, prep = clip.load(name, jit=False, device=device)
        self.model.eval().requires_grad_(False)
        self.input_size = prep.transforms[0].size
        self.train_prep = nn.Sequential(
                                        *([T.RandomErasing(p=0.9, scale=(0.02, 0.1)) for _ in range(4)] if erasing else []),
                                        T.RandomRotation(10), 
                                        T.RandomResizedCrop(self.input_size, 
                                                            scale=(0.8, 1), ratio=(0.9, 1.1)),
                                        T.RandomHorizontalFlip(),
                                        self.norm)
        self.test_prep = nn.Sequential(T.Resize(self.input_size), 
                                       T.CenterCrop(self.input_size),
                                      self.norm)
        
    def encode_image(self, x, ncuts=1):
        x = x.squeeze(0)
        x = torch.stack([self.train_prep(x) for _ in range(ncuts)]) if ncuts else self.test_prep(x)[None]
        x = self.model.encode_image(x.to(self.device)).normalize()
        return x.mean(0)
    
    def encode_text(self, x):
        with torch.no_grad():
            output = []
            for x_text in x:
                text_template = compose_text_with_templates(x_text,imagenet_templates)

                tokens = clip.tokenize(text_template).to(self.device)
        # text_features1 = perceptor.encode_text(tokens).detach()
        # text_features1 = text_features1.mean(axis=0, keepdim=True)
        # text_features1 /= text_features1.norm(dim=-1, keepdim=True)
        #         tokens = clip.tokenize(x).to(self.device)
                text_features = self.model.encode_text(tokens).detach()
                text_features = text_features.mean(axis=0, keepdim=True)
                # text_features.normalize()
                output.append(text_features.normalize())
                
            return output 
        
        
class CLIPS:
    def __init__(self, names=['RN50x4', 'ViT-B/32'], device='cpu', **kwargs):
        self.networks = {n:CLIP(n, device=device, **kwargs) for n in names}
        
    def encode_image(self, x, ncuts=0):
        return TensorDict({name:model.encode_image(x, ncuts=ncuts) for name, model in self.networks.items()})
    
    def encode_text(self, x):
        return TensorDict({name:model.encode_text(x) for name, model in self.networks.items()})
    
    

class LPIPS:
    def __init__(self, device, grey_ratio=0.):
        import lpips
        self.device = device
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        self.grey_ratio = grey_ratio
        
    def norm(self, x):
        if len(x.shape)<4:
            x = x[None]
        r = self.grey_ratio
        x = (1-r)*x + r*x.mean(1, keepdim=True).expand(x.shape[0], 3, *x.shape[2:])
        x = x.to(self.device)
        return x
        
    def __call__(self, x1, x2):
        return self.lpips_vgg(self.norm(x1), self.norm(x2), normalize=True).mean()
        