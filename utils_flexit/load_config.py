# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import yaml
import importlib
from types import SimpleNamespace as nspace
from functools import partial


def _get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


chars = 'abcdefghijklmnopqrstuvwxyz_'
def load_config(fname):
    #1: load yaml
    with open(fname) as f:
        str_cfg = yaml.safe_load(f)
    cfg = nspace(**str_cfg)  
    
    #2: find str that are objects
    for k, v in cfg.__dict__.items():
        if isinstance(v, str) and '.' in v and v[0].lower() in chars:
            setattr(cfg, k, _get_obj_from_str(v))
        elif isinstance(v, str) and v[0] == '$':
            setattr(cfg, k, getattr(cfg, v[1:]))
        elif isinstance(v, str) and any(x.lower() not in chars for x in v):
            try:
                setattr(cfg, k, eval(v))
            except:
                pass
            
    #3: replace functions with partials
    for key in str_cfg:
        if key.endswith('.__class__'):
            key_class = getattr(cfg, key)
            key2 = key[:-len('.__class__')]
            kwargs = {k[len(key2)+1:]:v for k, v in cfg.__dict__.items() if k.startswith(key2+'.') and k != key}
            setattr(cfg, key2, partial(key_class, **kwargs))
        
    #4: return object and str config
    return cfg, str_cfg