python main.py -p "Black Leopard" -s "Lion" -i "input_example/lion1.jpg" --output_path "./outputs/output_leopard" --use_range_restart --use_noise_aug_all --regularize_content



python main.py -i "input_example/reptile1.jpg"  --output_path "./outputs/output_reptile" -tg "input_example/reptile2.jpg" --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --use_colormatch --use_noise_aug_all
