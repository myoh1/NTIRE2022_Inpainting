from inspect import ArgSpec
import os
import time
import shutil
import sys
import logging
from datetime import datetime
from network import EdgeModel, Discriminator, InpaintingModel
from dataset import RestList
from utils import save_checkpoint, psnr, AverageMeter, FullLoss, GANLoss, Folder_Create, Evaluation, EdgeLoss, BCELoss
from torch.utils.data.distributed import DistributedSampler

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np

torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING=1

def Train(loaders, models, optims, criterions, epoch, best_score, args, output_dir, eval_score=None, print_freq=10, logger=None):
    # param
    LAMBDA_EDGE_ADV = args.edge_adv
    LAMBDA_EDGE_FM = args.edge_fm
    
    LAMBDA_RGB_L1 = args.rgb_l1
    LAMBDA_RGB_PERC = args.rgb_perc
    LAMBDA_RGB_STYLE = args.rgb_style
    LAMBDA_RGB_ADV = args.rgb_adv
    
    reverse_value = args.reverse

    # edge loss
    losses_Edge_D = AverageMeter()
    losses_Edge_Adv = AverageMeter()
    losses_Edge_feat = AverageMeter()
    # inpainting loss
    losses_l1 = AverageMeter()
    losses_style = AverageMeter()
    losses_perc = AverageMeter()
    losses_RGB_Adv = AverageMeter()
    losses_RGB_D = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    score_psnr = AverageMeter()
    
    # Loaders, criterions, models, optimizers
    Train_loader, Val_loader = loaders
    criterion_GAN, criterion_eval, criterion_Pix, criterion_L1, criterion_BCE = criterions
        
    Edge_G, Inpainting_G, Edge_D, Inpainting_D = models
    optim_Edge, optim_Inpainting, optim_Edge_D, optim_Inapinting_D = optims

    Edge_G.train()
    Inpainting_G.train()
    Edge_D.train()
    Inpainting_D.train()

    end = time.time()
    
    os.makedirs('train', exist_ok= True)
    #! ################ Train
    for i, (img, edge, gray, mask, name) in enumerate(Train_loader):
        data_time.update(time.time() - end)

        img_var = img.float().cuda() # 3c
        edge_var = edge.float().cuda() # 1c
        gray_var = gray.float().cuda() # 1c
        mask_var = mask[0].cuda() # 3c

        # reverse value #!(do or not?)
        if reverse_value:
            edge_var = (edge_var * -1) + 1 
            gray_var = (gray_var * -1) + 1
        
        masked_img = img_var * (1 - mask_var)
        masked_edge = edge_var * (1 - mask_var)
        masked_gray = gray_var * (1 - mask_var)
        
        edge = torch.cat((masked_edge, masked_gray, mask_var), dim = 1)
        
        #! ##########################  edge model ##########################################
        out_edge = Edge_G(edge)
        edge_comp = out_edge * mask_var + masked_edge
            
        ## edge_D ##
        # loss D
        loss_Edge_D = criterion_GAN(Edge_D(torch.cat((masked_gray, out_edge.detach()), dim = 1))[0], False) + criterion_GAN(Edge_D(torch.cat((masked_gray, edge_var), dim = 1))[0], True)
        # update
        optim_Edge_D.zero_grad()
        loss_Edge_D.backward()
        optim_Edge_D.step()
        losses_Edge_D.update(loss_Edge_D.data, img_var.size(0))
        
        ## edge_G ##
        # adv loss
        loss_edge_Adv = criterion_GAN(Edge_D(torch.cat((masked_gray, out_edge),dim=1))[0], True)
        # feature matching loss
        _, feat_fake = Edge_D(torch.cat((masked_gray,out_edge), dim = 1))
        _, feat_real = Edge_D(torch.cat((masked_gray, edge_var), dim = 1))
        loss_edge_feat = 0
        for feat_f, feat_r in zip(feat_fake, feat_real):
            loss_edge_feat += criterion_L1(feat_f, feat_r.detach())
        loss_Edge_G = LAMBDA_EDGE_ADV * loss_edge_Adv + LAMBDA_EDGE_FM * loss_edge_feat
        # update
        optim_Edge.zero_grad()
        loss_Edge_G.backward()
        optim_Edge.step()
        
        losses_Edge_feat.update(loss_edge_feat.data, img_var.size(0))
        losses_Edge_Adv.update(loss_edge_Adv.data, img_var.size(0))
        
        
        #! ###########################  mask detector ########################################     
        #! ##########################  inpainting model ########################################
        out_rgb  = Inpainting_G(edge_comp.detach(), masked_img, mask_var)
        masked_output = out_rgb * mask_var + masked_img
        
        ## inpainting_D ##
        # loss D
        loss_RGB_D = criterion_GAN(Inpainting_D(out_rgb.detach())[0], False) + criterion_GAN(Inpainting_D(img_var)[0], True)
        # update
        optim_Inapinting_D.zero_grad()
        loss_RGB_D.backward()
        optim_Inapinting_D.step()
        
        losses_RGB_D.update(loss_RGB_D.data, img_var.size(0))
        
        ## inpainting_G ##
        # l1 loss, style loss, preceptual loss
        l1_loss, perc_loss, style_loss  = criterion_Pix(out_rgb, img_var, 'pixel')
        # adv loss
        loss_RGB_Adv = criterion_GAN(Inpainting_D(out_rgb)[0], True)
        loss_rgb = LAMBDA_RGB_L1 * l1_loss + LAMBDA_RGB_STYLE * style_loss + LAMBDA_RGB_PERC * perc_loss + LAMBDA_RGB_ADV * loss_RGB_Adv 
        # update
        optim_Inpainting.zero_grad()
        loss_rgb.backward()
        optim_Inpainting.step()
        
        losses_l1.update(l1_loss.data, img_var.size(0))
        losses_style.update(style_loss.data, img_var.size(0))
        losses_perc.update(perc_loss.data, img_var.size(0))
        losses_RGB_Adv.update(loss_RGB_Adv.data, img_var.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % print_freq == 0:
            logger.info('E : [{0}][{1}/{2}]\t'
                        'T {batch_time.val:.3f}\n'
                        'Edge_D {Edge_D.val:.4f} ({Edge_D.avg:.4f})  '
                        'Edge_Adv {Edge_Adv.val:.4f} ({Edge_Adv.avg:.4f})  '
                        'Edge_feat {Edge_feat.val:.4f} ({Edge_feat.avg:.4f})\n'
                        
                        'RGB_D {RGB_D.val:.4f} ({RGB_D.avg:.4f})  '
                        'RGB_Adv {RGB_Adv.val:.4f} ({RGB_Adv.avg:.4f})  '
                        'L1 {l1.val:.4f} ({l1.avg:.4f})  '
                        'Style {style.val:.4f} ({style.avg:.4f})  '
                        'Perc {perc.val:.4f} ({perc.avg:.4f})  '.format(
                epoch, i, len(Train_loader), batch_time=batch_time,
                Edge_D = losses_Edge_D, 
                Edge_Adv = losses_Edge_Adv, 
                Edge_feat = losses_Edge_feat,
                RGB_D = losses_RGB_D,
                RGB_Adv = losses_RGB_Adv, 
                style = losses_style,
                perc = losses_perc,
                l1 = losses_l1))
                
    Edge_G.eval()
    Inpainting_G.eval()
    Edge_D.eval()
    Inpainting_D.eval()

    #! Validation
    for i, (img, edge, gray, mask, name) in enumerate(Val_loader):
        img_var = img.float().cuda() # 3c
        edge_var = edge.float().cuda() # 1c
        gray_var = gray.float().cuda() # 1c
        mask_var = mask[0].cuda() # 1c
        # reverse value
        if reverse_value:
            edge_var = (edge_var * -1) + 1 
            gray_var = (gray_var * -1) + 1
    
        _, _, h, w = img_var.size()
        img_var = F.interpolate(img_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        edge_var = F.interpolate(edge_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        gray_var = F.interpolate(gray_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        mask_var = F.interpolate(mask_var, size=(h + 16 - h % 16 , w + 16 - w % 16), mode='bilinear')
        
        masked_img = img_var * (1 - mask_var)
        masked_edge = edge_var * (1 - mask_var)
        masked_gray = gray_var * (1 - mask_var)
        edge = torch.cat((masked_edge, masked_gray, mask_var), dim = 1)
        
        with torch.no_grad():
            try:
                out_edge = Edge_G(edge)
                edge_comp = out_edge * mask_var + masked_edge
                out_rgb = Inpainting_G(edge_comp, masked_img, mask_var)
                out_comp = out_rgb * mask_var + masked_img
                
                score_psnr_  = criterion_eval(out_comp, img_var, 'PSNR')
                score_psnr.update(score_psnr_, img_var.size(0))
            except: pass

    if logger is not None:
        logger.info(' * PSNR  Score is {s.avg:.3f}'.format(s=score_psnr))

    return score_psnr.avg


def train_rest(args, saveDirName='.', logger=None):
    # Print the systems settings
    logger.info(' '.join(sys.argv))
    for k, v in args.__dict__.items():
        logger.info('{0}:\t{1}'.format(k, v))
    
    device = torch.device("cuda")
    # Hyper-parameters
    batch_size = args.batch_size
    crop_size = args.crop_size
    lr = args.lr
    weight = args.ckpt
    
    img_dir = args.img_root
    edge_dir = args.edge_root
    mask_dir = args.mask_root
    img_val_dir = args.img_val_root
    edge_val_dir = args.edge_val_root
    mask_val_dir = args.mask_val_root
    
    best_score = 0
    t_triplet = [transforms.RandomCrop(crop_size),
                transforms.Resize(crop_size),
                transforms.RandomFlip(),
                transforms.ToTensor_3()] # transform function for paired training data
    # Define transform functions
    t_pair = [transforms.RandomCrop(crop_size),
                transforms.Resize(crop_size),
                transforms.RandomFlip(),
                transforms.ToTensor()] # transform function for paired training data
    t_unpair = [#transforms.RandomCrop_One(crop_size),
                transforms.Resize_One(crop_size),
                transforms.RandomFlip_One(),
                transforms.ToTensor_One()] # transform function for unpaired training data
    v_triplet = [transforms.Resize_pair(), transforms.ToTensor_3()]
    v_triplet1 = [transforms.Resize(256), transforms.ToTensor_3()]
    v_pair = [transforms.Resize_pair(), transforms.ToTensor()] # transform function for paired validation data
    v_unpair = [transforms.ToTensor_One()] # transform function for unpaired validation training data
    v_unpair1 = [transforms.Resize(256), transforms.ToTensor_One()]
    v_tensor3 = [transforms.ToTensor_3()]
    v_tensor31 = [transforms.Resize(256),transforms.ToTensor_3()]
    
    # Define dataloaders
    
    Train_loader = torch.utils.data.DataLoader(
        RestList('train', img_dir, edge_dir, mask_dir, transforms.Compose(t_triplet), transforms.Compose(t_unpair), batch=batch_size, n = args.data_num),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    Val_loader = torch.utils.data.DataLoader(
        RestList('train', img_val_dir, edge_val_dir, mask_val_dir, transforms.Compose(v_triplet), transforms.Compose(v_unpair), out_name=True, n = args.data_num),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    loaders = Train_loader, Val_loader
        
    cudnn.benchmark = True

    # Define networks
    Edge_G = torch.nn.DataParallel(EdgeModel()).cuda()
    Inpainting_G = torch.nn.DataParallel(InpaintingModel()).cuda()
    
    Edge_D = torch.nn.DataParallel(Discriminator(in_channels =2)).cuda()
    Inpainting_D = torch.nn.DataParallel(Discriminator(in_channels = 3)).cuda()
    
    # Define optimizers
    optim_Edge = torch.optim.Adam(Edge_G.parameters(), lr=lr, betas=(0, 0.99))
    optim_Inpainting = torch.optim.Adam(Inpainting_G.parameters(), lr=lr, betas=(0, 0.99))
    optim_Edge_D = torch.optim.Adam(Edge_D.parameters(), lr=lr, betas=(0, 0.99))
    optim_Inapinting_D = torch.optim.Adam(Inpainting_D.parameters(), lr=lr, betas=(0, 0.99))
    
    
    if weight is not None:
        ckpt = torch.load(weight, map_location= device)
        
        
        Edge_G.load_state_dict(ckpt['Edge_G'])
        Inpainting_G.load_state_dict(ckpt['Inpainting_G'])
        Edge_D.load_state_dict(ckpt['Edge_D'])
        Inpainting_D.load_state_dict(ckpt['Inpainting_D'])
        
        Edge_G = Edge_G.module
        Inpainting_G = Inpainting_G.module
        Edge_D = Edge_D.module
        Inpainting_D = Inpainting_D.module
        
        optim_Edge.load_state_dict(ckpt['optim_Edge'])
        optim_Inpainting.load_state_dict(ckpt['optim_Inpainting'])
        optim_Edge_D.load_state_dict(ckpt['optim_Edge_D'])
        optim_Inapinting_D.load_state_dict(ckpt['optim_Inapinting_D'])
        
        n = ckpt['epoch']
        print(f"Started Epoch : {n}")
        
        
    optims = optim_Edge, optim_Inpainting, optim_Edge_D, optim_Inapinting_D
    models = Edge_G, Inpainting_G, Edge_D, Inpainting_D
    

    # Define loss functions
    criterion_eval = Evaluation().cuda() # calculate psnr, ssim, lpips
    criterion_GAN  = GANLoss().cuda() # adv
    criterion_Pix  = FullLoss().cuda() # l1, perc, style ( or ssim )
    criterion_L1 = EdgeLoss().cuda() # l1
    criterion_BCE = BCELoss().cuda() # bce

    criterions = criterion_GAN, criterion_eval, criterion_Pix, criterion_L1, criterion_BCE

    
    # print(args)
    for epoch in range(args.epochs): # train and validation
        if weight is not None: epoch = epoch + n
        
        logger.info('Epoch : [{0}]'.format(epoch))
        
        val_psnr = Train(loaders, models, optims, criterions, epoch, best_score, args, output_dir=saveDirName+'/val', eval_score=psnr, logger=logger)
        
        ## save the neural network
        if best_score < val_psnr :
            best_score = val_psnr
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6] + '_Better.pkl'
        else : 
            history_path = saveDirName + '/' + 'ckpt{:03d}_'.format(epoch + 1) + 'p_' + str(val_psnr)[:6]+ '.pkl'

        save_checkpoint({
            'epoch': epoch + 1,
            
            'Edge_G': Edge_G.state_dict(),
            'Inpainting_G': Inpainting_G.state_dict(),
            'Edge_D': Edge_D.state_dict(),
            'Inpainting_D': Inpainting_D.state_dict(),
            
            'optim_Edge': optim_Edge.state_dict(),
            'optim_Inpainting': optim_Inpainting.state_dict(),
            'optim_Edge_D': optim_Edge_D.state_dict(),
            'optim_Inapinting_D': optim_Inapinting_D.state_dict(),
            
        }, True, filename=history_path)
