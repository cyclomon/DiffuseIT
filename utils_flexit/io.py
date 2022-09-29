# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import numpy as np
import json
import pandas as pd
import json
from PIL import Image
import yaml
    
def load(fname, **kwargs):
    ext = str(fname).split('.')[-1]
    if ext == 'pt':
        return torch.load(fname, **kwargs)
    elif ext == 'npy':
        return np.load(fname, **kwargs)
    elif ext == 'txt':
        with open(fname) as f:
            lines = [x[:-1] for x in f]
        return lines
    elif ext == 'json':
        with open(fname) as f:
            obj = json.load(f)
        return obj
    elif ext == 'csv':
        return pd.read_csv(fname, **kwargs)
    elif ext == 'tsv':
        return pd.read_csv(fname, sep='\t', **kwargs)
    elif ext in ('yaml', 'yml'):
        with open(fname) as f:
            obj = yaml.safe_load(f)
        return obj
    elif ext.lower() in ('png', 'jpg', 'jpeg'):
        return Image.open(fname).convert('RGB')
    
    print(f'unrecognized extension {fname}')

def dump(obj, fname, **kwargs):
    if isinstance(obj, np.ndarray):
        np.save(fname, obj, **kwargs)
    elif isinstance(obj, torch.Tensor):
        torch.save(obj, fname, **kwargs)
    elif isinstance(obj, Image.Image):
        obj.save(fname)
    elif isinstance(obj, dict):
        with open(fname, 'w') as f:
            yaml.dump(obj, f)
    else:
        print('Unrecognized object type', type(obj))