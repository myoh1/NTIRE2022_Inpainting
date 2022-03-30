import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.system("\
        python3 test.py \
        --img_root {your directory} \
        --edge_root {your directory} \
        --mask_root {your directory} \
        --ckpt ckpt/{Dataset}.pkl \
          ")
