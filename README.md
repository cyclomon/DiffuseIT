# DiffuseIT
## Official repository of "Diffusion-based Image Translation using Disentangled Style and Content Representation" (ICLR 2023)
### [Gihyun Kwon](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html)
LINK : https://arxiv.org/abs/2209.15264

### Environment
Pytorch 1.9.0, Python 3.9

```
$ conda create --name DiffuseIT python=3.9
$ conda activate DiffuseIT
$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install color-matcher
$ pip install git+https://github.com/openai/CLIP.git
```

### Model download
To generate images, please download the pre-trained diffusion model

imagenet 256x256 [LINK](https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view?usp=sharing)

FFHQ 256x256 [LINK](https://drive.google.com/file/d/1-oY7JjRtET4QP3PIWg3ilxAo4VfjCa3J/view?usp=sharing)

download the model into ```./checkpoints``` folder

For face identity loss when using FFHQ pre-trained model, download pre-trained ArcFace model [LINK](https://drive.google.com/file/d/1SJa5qVNM6jGZdmsnUsGNhjtrssGYuJfT/view?usp=sharing)

save the model into ```./id_model```

### Text-guided Image translation

We provide Colab Demo for Text-guided Image translation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OlN7LBT-cH8D0cY0arfhNoGyxUwMXz56?usp=sharing)

```
python main.py -p "Black Leopard" -s "Lion" -i "input_example/lion1.jpg" --output_path "./outputs/output_leopard" 
--use_range_restart --use_noise_aug_all --regularize_content
```

To to further regularize content when CLIP loss is extremely low, activate ```--regularize_content```

To use noise augmented images for our ViT losses, activate ```--use_noise_aug_all```

To use progressively increasing our contrastive loss, activate ```--use_prog_contrast```

To restart the whole process with high rgb regularize loss, activate ```--use_range_restart```

To use FFHQ pre-trained model, activate ```--use_ffhq```

For memory saving, we can use single CLIP model with ```--clip_models 'ViT-B/32'```

### Image-guided Image translation

We provide Colab Demo for Image-guided Image translation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nDAJ-rxftY-_1sSX48P-on26WBRlSIAw?usp=sharing)
```
python main.py -i "input_example/reptile1.jpg"  --output_path "./outputs/output_reptile" 
-tg "input_example/reptile2.jpg" --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 
--use_colormatch --use_noise_aug_all
```

To remove the color matching, deactivate ```--use_colormatch```



Our source code rely on Blended-diffusion, guided-diffusion, flexit, splicing vit
