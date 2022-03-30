import os
from cv2 import threshold
import numpy as np
import random

from PIL import Image, ImageOps
from glob import glob
#import rawpy
import data_transforms as transforms
from torchvision import transforms as transforms
import torch

class RestList(torch.utils.data.Dataset):
    def __init__(self, phase, img_dir, edge_dir, mask_dir, t_pair, t_unpair, n=1, batch=1, out_name=False):
        self.phase = phase
        self.img_dir = img_dir
        self.edge_dir = edge_dir
        self.mask_dir = mask_dir
        self.batch = batch

        self.t_pair = t_pair
        self.t_unpair = t_unpair
        self.out_name = out_name

        self.image_list = None
        self.edge_list = None
        self.mask_list = None
        
        self.test_image_list = None
        self.test_edge_list = None
        self.test_mask_list = None
        
        self.DATA_NUM = n
        
        self._make_list()

    #! EDGE INPUT
    def __getitem__(self, index):
        np.random.seed()
        random.seed()
        if self.phase == 'train':
            assert self.image_list[index].split('/')[-1] == self.edge_list[index].split('/')[-1], \
                f"Different source, [{self.image_list[index].split('/')[-1]},{self.edge_list[index].split('/')[-1]}]"
            
            img = Image.open(self.image_list[index]).convert('RGB')
            img = ImageOps.exif_transpose(img)
            
            edge = Image.open(self.edge_list[index]).convert('L')
            edge = ImageOps.exif_transpose(edge)
            
            assert img.size == edge.size,\
                f"Different size, {self.image_list[index], self.edge_list[index]}"
            
            index_m = np.random.randint(len(self.mask_list))
            Mask = Image.open(self.mask_list[index_m]).convert('L')
            
            data = list(self.t_pair(*(img, edge)))
            data.append(self.t_unpair(*[Mask]))
            data.append(self.mask_list[index_m])

        elif self.phase in 'test':
            img  = Image.open(self.test_image_list[index]).convert('RGB')
            edge= Image.open(self.test_edge_list[index]).convert('L')
            Mask = Image.open(self.test_mask_list[index]).convert('L')
            name = (self.test_image_list[index]).split('/')
            name = name[-2] + '/' +name[-1]
            
            data = list(self.t_pair(*(img, edge)))
            data.append(self.t_unpair(*[Mask]))
            data.append(name)

        return tuple(data)


    def __len__(self):
        if self.phase in 'train':
            if self.DATA_NUM is None : return len(self.image_list)
            else: return self.DATA_NUM

        elif self.phase in 'test':
            return len(self.test_image_list)

    def _make_list(self):
        #! Train
        if self.phase == 'train':
            # '/root/workplace/NTIRE_inpainting/01_Data/Train_ImageNet/train/*.png'
            # '/root/workplace/NTIRE_inpainting/01_Data/Train_ImageNet/mask/*.png'
            
            img_source = os.path.join(self.img_dir, '*.png')
            edge_source = os.path.join(self.edge_dir, '*.png')
            mask_source = os.path.join(self.mask_dir, '*.png')
            
            image_list = sorted(glob(img_source))
            print("Img Loaded")
            edge_list = sorted(glob(edge_source))
            print("Edge Loaded")
            
            rand_idx = random.sample(range(0, len(image_list)), self.DATA_NUM)
            self.image_list = [image_list[x] for x in rand_idx]
            self.edge_list = [edge_list[x] for x in rand_idx]
            self.mask_list = sorted(glob(mask_source))
            print()
            print('Num of training images : ' + str(len(self.image_list)))
            print('Num of training masks : ' + str(len(self.mask_list)))
            
        
        #! TEST
        elif self.phase == 'test' :
            img_source = os.path.join(self.img_dir, '**/*.png')
            mask_source = os.path.join(self.mask_dir, '**/*.png')
            edge_source = os.path.join(self.edge_dir, '**/*.png')
            
            self.test_image_list = sorted(glob(img_source))
            print('Num of Test imgs : ' + str(len(self.test_image_list)))
            self.test_edge_list = sorted(glob(edge_source))
            
            self.test_mask_list = sorted(glob(mask_source))
            print('Num of Test masks : ' + str(len(self.test_mask_list)))
        
        

