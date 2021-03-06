from re import X
import time
import datetime
from turtle import xcor
import torch
import math
from torch import nn, optim
from torch.nn import functional as F
from modules import ConvLayer, ResBlk, ASPP, spectral_norm, ResnetBlock, GatedConv2d, TransposeGatedConv2d
import numpy as np
from torchvision import transforms as t

class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.ec1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0)),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
            )
        self.ec2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
            )
        self.ec3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
            )

        blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dc1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
            )
        self.dc2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            )
        self.dc3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
            )
        
        self.init_weights()
        
    def init_weights(self, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, edge):
        
        x1 = self.ec1(edge)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.middle(x3) + x3
        x5 = self.dc1(x4)
        x6 = self.dc2(x5)
        x7 = self.dc3(x6)
        x8 = (torch.tanh(x7) + 1) / 2
        return x8

class InpaintingModel(nn.Module):
    def __init__(self):
        super(InpaintingModel, self).__init__()
        #! Downsample 2times
        self.ec1 = nn.Sequential(
            ASPP(5, 64),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
            )
        self.ec2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
            )
        self.ec3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
            )
        
        blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dc1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
            )
        self.dc2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
            )
        self.dc3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
            )

        self.init_weights()
        
    def init_weights(self, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
    def forward(self, edge, img, mask):
        x = torch.cat((img, edge, mask), dim = 1)
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.middle(x3) + x3
        x5 = self.dc1(x4)
        x6 = self.dc2(x5)
        x7 = self.dc3(x6)
        x8 = (torch.tanh(x7) + 1) / 2
        return x8

class Discriminator(nn.Module):
    
    def __init__(self, in_channels, use_sigmoid=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)),
        )

        self.init_weights()

    def init_weights(self, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]
    
