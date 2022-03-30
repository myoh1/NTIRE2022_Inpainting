This is NTIRE2022 Inpainting Track1(Unsupervised) Repository(https://codalab.lisn.upsaclay.fr/competitions/1607).
We trained each model for 4 datasets(Places, WikiArt, FFHQ, ImageNet)




# Data Preparation

We used edge to emphasize structure information by using Dexined(https://github.com/xavysp/DexiNed) for more detail edge information. So Each Image needs its corresponding edge result.



# Test

Edit your data and checkpoint directory of `run_test.py` and run it. You can download my checkpoints of each dataset here(https://drive.google.com/file/d/18RT-shl0FIqFmNVXfHTdNYj7byjRbaNf/view?usp=sharing)

I assume the following structure of data.
```
Image, Mask, Edge
- <dataset name>
|   +-- Completion
|		+- 000000.png
|		+- 000001.png
|			...
|   +-- ThinStrokes
			...
```
# Train

Edit your data directory of `run_train.py` and run it. 


```
Image, Mask, Edge
- train
|  - 000000.png
|  - 000001.png
|  - 000002.png
|  - 000003.png
|		...

