import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.system("\
        python3 main.py \
        --img_root {your directory} \
        --edge_root {your directory} \
        --mask_root {your directory} \
        --img_val_root {your directory} \
        --edge_val_root {your directory} \
        --mask_val_root {your directory} \
        --ckpt {if you want to resume training} \
        --flag {your training flag} \
        -n 100000 \
        ")
