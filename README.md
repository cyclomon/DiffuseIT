# DiffuseIT
## Official repository of "Diffusion-based Image Translation using Disentangled Style and Content Representation"

### Environment
Pytorch 1.9.1, Python 3.8

```
$ conda create --name DiffuseIT python=3.9
$ conda activate DiffuseIT
$ pip3 install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Text-guided Image translation

```
python main.py -p "Black Leopard" -s "Lion" -i "input_example/lion1.jpg" --output_path "./outputs/output_leopard" 
--use_range_restart --use_noise_aug_all --regularize_content
```

To to further regularize content when CLIP loss is extremely low, activate ```--regularize_content```

To use noise augmented images for our ViT losses, activate ```--use_noise_aug_all```

To use progressively increasing our contrastive loss, activate ```--use_prog_contrast```

To restart the whole process with high rgb regularize loss, activate ```--use_range_restart```

### Image-guided Image translation
```
python main.py -i "input_example/reptile1.jpg"  --output_path "./outputs/output_reptile" -tg "input_example/reptile2.jpg" --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --use_colormatch --use_noise_aug_all
```

To remove the color matching, deactivate ```--use_colormatch```



Our source code rely on Blended-diffusion, guided-diffusion, flexit, splicing vit
