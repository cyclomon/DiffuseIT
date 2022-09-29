# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch

def monkey_typing(module):
    def deco(f):
        setattr(module, f.__name__, f)
    return deco
    
    
@monkey_typing(torch.nn.Module)
def batch_forward(self, x, batch_size=16):
    module_device = next(self.parameters()).device
    
    out = torch.cat([self(batch.to(module_device)).to(x.device).detach() for batch in x.split(batch_size)])
    return out

@monkey_typing(torch.Tensor)
def normalize(self):
    return self/self.norm(dim=-1, keepdim=True)

def chain(x, *fl):
    for f in fl:
        x = f(x)
    return x

class TensorDict:
    def __init__(self, dct={}, **kwargs):
        self.dct = dct
        self.dct.update(**kwargs)
        
    def __getitem__(self, i):
        if isinstance(i, str) and i in self.dct:
            return self.dct[i]
        elif isinstance(i, TensorDict):
            return TensorDict({k: self.dct[k] * i[k] for k in self.dct})
            
        else:
            return TensorDict({k:v.__getitem__(i) for k, v in self.dct.items()})

    
    def __getattr__(self, attr):
        return TensorDict({k:getattr(v, attr) for k, v in self.dct.items()})
    
    def __call__(self, *args, **kwargs):
        return TensorDict({k:v(*args, **kwargs) for k, v in self.dct.items()})
        
    def apply(self, f):
        return TensorDict({k:f(v) for k, v in self.dct.items()})
    
    def __matmul__(self, t2):
        t1 = self.dct
        return TensorDict({k: t1[k] @ t2[k] for k in self.dct})
    
    def __sub__(self, t2):
        t1 = self.dct
        return TensorDict({k: t1[k] - t2[k] for k in self.dct})
        
    def __add__(self, t2):
        t1 = self.dct
        return TensorDict({k: t1[k] + t2[k] for k in self.dct})
    
    def __mul__(self, t2):
        t1 = self.dct
        return TensorDict({k: t1[k] * t2[k] for k in self.dct})
    
    def __truediv__(self, t2):
        t1 = self.dct
        return TensorDict({k: t1[k] / t2[k] for k in self.dct})
    
        
    def __rmul__(self, l):
        t1 = self.dct
        
        if isinstance(l, int) or isinstance(l, float):
            return TensorDict({k: l * t1[k] for k in self.dct})
        
        else:
            return TensorDict({k: t1[k] * t2[k] for k in self.dct})
        
    def __gt__(self, l):
        t1 = self.dct
        if isinstance(l, int) or isinstance(l, float):
            return TensorDict({k: (t1[k] > l) for k in self.dct})
        else:
            return TensorDict({k: (t1[k] > l[k]) for k in self.dct})
            
        
    def __lt__(self, l):
        t1 = self.dct
        if isinstance(l, int) or isinstance(l, float):
            return TensorDict({k: (t1[k] < l) for k in self.dct})
    
    def reduce(self, f=lambda x:x):
        return f(list(self.dct.values()))
    
    def __repr__(self):
        rep = ''
        for k, v in self.dct.items():
            rep += f'{k}: tensor{tuple(v.shape)}\n'
        return rep