import os
import sys

from network import EdgeModel, InpaintingModel
from dataset import RestList

import torch
from torch import device, nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import time

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--ckpt', default=None, type=str) # weight path
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--img_root', default = None, type = str)
    parser.add_argument('--edge_root', default = None, type = str)
    parser.add_argument('--mask_root', default = None, type = str)
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    phase = args.phase
    weight = args.ckpt
    batch_size = args.batch_size
    Edge_G = torch.nn.DataParallel(EdgeModel().cuda())
    Inpainting_G = torch.nn.DataParallel(InpaintingModel().cuda())
    
    img_dir = args.img_root
    edge_dir = args.edge_root
    mask_dir = args.mask_root
    
    device = torch.device("cuda")
    ckpt = torch.load(weight, map_location= device)
    dataset = img_dir.split('/')[-1]
    try:
        Edge_G.load_state_dict(ckpt['Edge_G'])
        Inpainting_G.load_state_dict(ckpt['Inpainting_G'])
        Edge_G = Edge_G.module
        Inpainting_G = Inpainting_G.module
    except:
        Edge_G = Edge_G.module
        Inpainting_G = Inpainting_G.module
        Edge_G.load_state_dict(ckpt['Edge_G'])
        Inpainting_G.load_state_dict(ckpt['Inpainting_G'])
        
    print("Weight Loaded")
    Edge_G.eval()
    Inpainting_G.eval()
    
    transform1 = [transforms.ToTensor_3()]
    transform2 = [transforms.ToTensor_One()]
    
    Test_loader = torch.utils.data.DataLoader(
            RestList(phase, img_dir, edge_dir, mask_dir, transforms.Compose(transform1), transforms.Compose(transform2), batch=batch_size),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
            
    output_dir = os.getcwd()
    
    save_dir_t = os.path.join(output_dir, 'result', 'submit', f'{dataset}', 'test')
    
    #! Test
    print("TEST")
    start = time.time()
    for i, (img, edge, gray, mask, name) in enumerate(Test_loader): #mask = 1
        img_var = img.float().cuda() # 3c
        edge_var = edge.float().cuda() # 1c
        gray_var = gray.float().cuda() # 1c
        mask_var = mask[0].cuda() # 3c
        name = name[0]
        
        stroke_type = name.split('/')[0]
        os.makedirs(save_dir_t+'/'+stroke_type, exist_ok=True)

        _, _, h, w = img_var.size()
        img_var = F.interpolate(img_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        edge_var = F.interpolate(edge_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        gray_var = F.interpolate(gray_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        mask_var = F.interpolate(mask_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        
        edge_var = (edge_var * -1) + 1 
        gray_var = (gray_var * -1) + 1

        masked_img = img_var * (1 - mask_var)
        masked_edge = edge_var * (1 - mask_var)
        masked_gray = gray_var * (1 - mask_var)
        
        edge = torch.cat((masked_edge, masked_gray, mask_var), dim = 1)

        with torch.no_grad():
            try:
                out_edge = Edge_G(edge)
                edge_comp = out_edge * mask_var + masked_edge
                out_rgb = Inpainting_G(edge_comp, masked_img, mask_var)
                output_comp = out_rgb * mask_var + masked_img
                out_comp = F.interpolate(output_comp, size=(h, w), mode='bilinear')
                
            except:
                print("Large Size Data")
                print(name)
                print("========================")
                edge = F.interpolate(edge, size=(512,512), mode='bilinear')
                mask_var_s = F.interpolate(mask_var, size=(512,512), mode='bilinear')
                masked_edge = F.interpolate(masked_edge, size=(512,512), mode='bilinear')
                masked_img_s = F.interpolate(masked_img, size=(512,512), mode='bilinear')
                
                out_edge = Edge_G(edge)
                edge_comp = out_edge * mask_var_s + masked_edge
                out_rgb_s = Inpainting_G(edge_comp, masked_img_s, mask_var_s) # 512
                
                out_rgb = F.interpolate(out_rgb_s, size=(h, w), mode='bilinear') # ori
                mask_var = F.interpolate(mask_var, size=(h, w), mode='bilinear') # ori
                masked_img = F.interpolate(masked_img, size=(h, w), mode='bilinear') # ori
                
                out_comp = out_rgb * mask_var + masked_img                
                
            resultname = os.path.join(save_dir_t, name[:-4] + '.png')
            print(resultname)
            # save_image(out_comp[0], resultname)
            
            
    print(f"runtime per frame: {(time.time() - start)/len(Test_loader):.4f}s\n")
    
    
if __name__ == '__main__':
    main()
