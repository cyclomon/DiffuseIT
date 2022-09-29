# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import submitit
import os
import numpy as np
from pathlib import Path

def submit(f, main_arg, *args, 
           folder='default', 
           slurm_partition='devlab',
           ngpus=2,
           **kwargs):
    #function to parrallelize f
    Path(f'sblogs/{folder}').mkdir(exist_ok=True)
    i = 0
    while os.path.exists(f'sblogs/{folder}/run_{i}/'):
        i += 1
    aex = submitit.AutoExecutor(folder=f'sblogs/{folder}/run_{i}/')

    aex.update_parameters(timeout_min=60*24, 
                          slurm_partition=slurm_partition, 
                          nodes=1, 
                          gpus_per_node=1, 
                          slurm_array_parallelism=ngpus,
                          slurm_constraint='volta32gb')

    jobs = [aex.submit(f, batch, *args, **kwargs) for batch in np.array_split(main_arg, ngpus)]
    print(f'{ngpus} jobs successfully scheduled !')
    