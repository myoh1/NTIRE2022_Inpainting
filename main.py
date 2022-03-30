import argparse
from email.policy import default
import logging
import os
import threading
import time
import numpy as np
import shutil
from os.path import join, exists, split
from math import log10
from train import train_rest

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from dataset import RestList
import data_transforms as transforms


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--crop-size', default=256, type=int) #
    parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N') #
    parser.add_argument('--epochs', type=int, default=150, metavar='N') #
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--flag', default='', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('-r', '--reverse', default=True, action='store_false')
    parser.add_argument('-ea', '--edge_adv', type=float, default=1)
    parser.add_argument('-ef', '--edge_fm', type=float, default=10)
    parser.add_argument('-rl', '--rgb_l1', type=float, default=1)
    parser.add_argument('-rp', '--rgb_perc', type=float, default=0.1)
    parser.add_argument('-rs', '--rgb_style', type=float, default=250)
    parser.add_argument('-ra', '--rgb_adv', type=float, default=0.0001)
    
    parser.add_argument('-n', '--data_num', default=3000000, type=int)
    parser.add_argument('--img_root', default = None, type = str)
    parser.add_argument('--edge_root', default = None, type = str)
    parser.add_argument('--mask_root', default = None, type = str)
    parser.add_argument('--img_val_root', default = None, type = str)
    parser.add_argument('--edge_val_root', default = None, type = str)
    parser.add_argument('--mask_val_root', default = None, type = str)
    
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    
    return args

def main():
    args = parse_args()
    
    dt_now = datetime.now()
    timeName = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(dt_now.year, dt_now.month, \
    dt_now.day, dt_now.hour, dt_now.minute)
    saveDirName = './runs/train/' + timeName + '_' + args.flag
    os.makedirs(saveDirName, exist_ok=True)

    # logging configuration
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(saveDirName + '/log_training.log')
    logger.addHandler(file_handler)

    train_rest(args, saveDirName=saveDirName, logger=logger)

if __name__ == '__main__':
    main()
