import os
import threading
# from wsgiref.types import InputStream
import numpy as np
import shutil
from math import log10, exp
from PIL import Image
from datetime import datetime
import logging
from scipy import linalg

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models import vgg19, inception_v3


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

@torch.no_grad()
def calculate_fi_given_path(path, img_size=256, batch_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def psnr(output, target):
    """
    Computes the PSNR
    """
    psnr = 0
    
    with torch.no_grad():
        mse = F.mse_loss(output, target)
        psnr = 10 * log10( 1/ (mse.item()+1e-8))
        
    return psnr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(args, epoch, optimizer, lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    # """
    if args.lr_mode == 'step':
        lr = max(args.lr * (0.5 ** (epoch // args.step)), 1e-10)
    # elif args.lr_mode == 'poly':
    #     lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    # else:
    #     raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, pre, names, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # pdb.set_trace()
    for ind in range(len(names)):
        #print(predictions[ind].shape)
        im = Image.fromarray(np.transpose(predictions[ind], (1, 2, 0)).astype(np.uint8))
        # fn = os.path.join(output_dir, names[ind][:-4] + pre+'.jpg')
        fn = os.path.join(output_dir, names[ind][:-4] + pre+'.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn, quality=100)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILasdfINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, sigma = 1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def make_one_hot(labels, class_num=8):
    one_hot = torch.cuda.FloatTensor(labels.size(0), class_num, labels.size(2), labels.size(3))
    target = one_hot.scatter_(1, labels.data, 1)
    return Variable(target)

class ProjectedLoss(torch.nn.Module):
    def __init__(self, n_vec=2):
        super(ProjectedLoss, self).__init__()
        self.n_vec = n_vec

    def forward(self, out_rep, gt_rep):
        mse_loss = 0
        for i in range(self.n_vec):
            mse_loss += F.mse_loss(out_rep[3*i:3*(i+1)], gt_rep[3*i:3*(i+1)])

        return mse_loss

class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)

# --- Perceptual loss network  --- #
class VGG(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGG, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "reul2_2",
            '17': "relu3_4",
            '26': "relu4_4",
            '35': "relu5_4"}
        # self.layer_name_mapping = {
        #     '3': "relu1_2",
        #     '8': "relu2_2",
        #     '17': "relu3_4",
        #     '26': "relu4_4",
        #     '35': "relu5_4"}
        
    def extract_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    def forward(self, x):
        return self.extract_features(x)

class Evaluation(torch.nn.Module):
    def __init__(self):
        super(Evaluation, self).__init__()
        self.LPIPS_Net = vgg19(pretrained=True).features
        self.ssim_module = SSIM()

    def forward(self, output, gt, metric='PSNR'):
        if metric == 'PSNR' :
            return psnr(output, gt)
        elif metric == 'SSIM' :
            return 1 - self.ssim_module(output, gt)
        elif metric == 'LPIPS' :
            return F.l1_loss(self.LPIPS_Net(output), self.LPIPS_Net(gt))

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
    def forward(self, edge, gt):
        return F.l1_loss(edge, gt)
    
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
    def forward(self, input, gt):
        return F.binary_cross_entropy(input, gt)
    
class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss
    
class FullLoss(torch.nn.Module):
    def __init__(self):
        super(FullLoss, self).__init__()
        vgg_model = vgg19(pretrained=True).features
        vgg_model = vgg_model.cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_module = VGG(vgg_model)
        self.ssim_module = SSIM()
        self.style_module = StyleLoss()
    def forward(self, out_img, gt_img, flag):
        if flag == 'pixel' :
            l1_loss = F.l1_loss(out_img, gt_img)
            ssim_loss = self.ssim_module(out_img, gt_img)
            p_loss = []
            s_loss = []
            inp_features = self.vgg_module(out_img)
            gt_features = self.vgg_module(gt_img)
            for i in range(5):
                p_loss.append(F.mse_loss(inp_features[i],gt_features[i]))
                # s_loss.append(F.mse_loss(gram_matrix(inp_features[i]),gram_matrix(gt_features[i])))
            perc_loss = sum(p_loss)/len(p_loss)
            # style_loss = sum(s_loss)/len(s_loss)
            style_loss = self.style_module(out_img, gt_img)
            
            return l1_loss, perc_loss, style_loss
        else :
            inp_features = self.vgg_module(out_img)
            gt_features = self.vgg_module(gt_img)
            perc_loss = F.mse_loss(inp_features[flag], gt_features[flag])
            return perc_loss

def DeepLoss(criterion, Pull_outs, Pull_core, Push_outs1, Push_outs2, level, Pull_list_flag = True, Push_list_flag = True):
    loss = 0
    if Pull_list_flag == True :
        for ind in range(len(Pull_outs)):
            loss += criterion(Pull_outs[ind], Pull_core, level-1)
    else :
        loss += criterion(Pull_outs, Pull_core, level-1)
        
    if Push_list_flag == True :
        for ind in range(len(Push_outs1)):
            loss -= 0.1 * criterion(Push_outs1[ind], Push_outs2[ind], level-1)
    else :
        loss -= 0.1 * criterion(Push_outs1, Push_outs2, level-1)

    return loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return (1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average))

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), \
            sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight_data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def Gaussiansmoothing(img, channel=3, window_size = 11):
    window = create_window(window_size, channel, sigma=5)
    
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    x_smooth = F.conv2d(img, window, padding = window_size//2, groups = channel)
    
    # return x_smooth, torch.clamp(img - x_smooth, min=0)
    return x_smooth, img - x_smooth


def Folder_Create(save_dir):
    os.makedirs(os.path.join(save_dir, 'Completion'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'Every_N_Lines'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'Expand'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'MediumStrokes'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'Nearest_Neighbor'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ThickStrokes'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ThinStrokes'), exist_ok=True)

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out