import os
import math
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
from model.acquire_model import getModel
from dataset import VideoVimeoDataset, CustomToTensor
from utils import *
from metric.psnr import get_psnr
from metric.ms_ssim import get_ssim, get_msssim

parser = argparse.ArgumentParser(description='Deep Image Compression Evaluator.')
parser.add_argument('--data_dir', type=str, default='', help='')
parser.add_argument('--lst', type=str, default='', help='')
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--image', type=str, default='', help='')
parser.add_argument('--output_dir', type=str, default='', help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--scale_max', type=float, default=5, help='')
parser.add_argument('--hyper_gaussian', action='store_true', help='')
parser.add_argument('--N', type=int, default=192, help='')
parser.add_argument('--test_type', type=int, default=1, help='')
parser.add_argument('--CLIC', action='store_true', help='')
parser.add_argument('--scale_bound', type=float, default=1e-9, help='')
parser.add_argument('--is_int', action='store_true', default=False)
parser.add_argument('--e_num', type=int, default=2, help='')
parser.add_argument('--num', type=int, default=2, help='')
parser.add_argument('--p_num', type=int, default=2, help='')
parser.add_argument('--eval_num', type=int, default=-1, help='')
parser.add_argument('--model', type=str, default='multi2d', help='')
parser.add_argument('--type_fusion', type=str, default='2d', help='')
parser.add_argument('--type_encoder', type=str, default='2d', help='')
parser.add_argument('--others', type=str, default='', help='')
parser.add_argument('--entropy', type=str,
                    choices=['gaussian', 'laplacian'], default='gaussian', help='')

args, unparsed = parser.parse_known_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    """The main function.
    """
    assert os.path.isfile(args.resume)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # create model
    auto_encoder, image_num = getModel(args, is_eval=True)
    eval_dataset = VideoVimeoDataset(image_num, args.data_dir, args.lst, transforms.Lambda(lambda images: torch.stack([CustomToTensor()(image) for image in images])), args.test_type)

    auto_encoder.to(device)

    # restore from a checkpoint
    auto_encoder.load_resume(args.resume)
    if args.others:
        auto_encoder.load_others(args.others)
    if args.image:
        auto_encoder.load_image(args.image)

    cudnn.benchmark = True

    # Data loading code.

    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              pin_memory=True)

    eval(eval_loader, auto_encoder)


def eval(eval_loader, model):
    """The eval function.
    """
    # switch to eval mode
    model.eval()

    bpp_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ms_ssim_meter = AverageMeter()
    list_m = []
    list_b = []
    list_i = []

    with torch.no_grad():
        # create the progress bar
        pbar = tqdm(total=len(eval_loader))
        x_last = 0
        for i, (image_name, x_all) in enumerate(eval_loader):
            if i == args.eval_num:
                break
            # padding for invalid resolution
            x_all = x_all.to(device, non_blocking=True)
            bs, num, c, h, w = x_all.size()
            # bs, num, c, h, w = x_all.size()
            _, x = x_all.view(bs, num, c, h, w).split([num - 1, 1], dim=1)
            x = x.view(-1, c, h, w)
            if args.model == 'yflowr' and i % 100 != 0:
                x_all[:,0] = x_last
            # save_image("feature", ('ref.png', 0), x)
            w_diff = int(math.ceil(w / 64) * 64) - w
            h_diff = int(math.ceil(h / 64) * 64) - h
            x_inflate = F.pad(x_all.view(-1, c, h, w), (w_diff, 0, h_diff, 0),  mode='replicate').view(bs, num, c, h + h_diff, w + w_diff)

            if args.model == 'ybasef':
                x_tilde, y_likelihoods, z_likelihoods, fy_li, fz_li = model(x_inflate)#, image_name)
            else:
                x_tilde, y_likelihoods, z_likelihoods, fy_li, fz_li = model(x_inflate)#, image_name)
            x_last = x_tilde[:,:,h_diff:, w_diff:]
            x_hat = x_tilde.round()
            x_hat = x_hat[:, :, h_diff:, w_diff:]
            # save_image("rec", ('last' + str(i+1) +'.png', 0), x_all[:,0])
            # save_image("rec", ('rec' + str(i+1) +'.png', 0), x_hat)
            # save_image("rec", ('real' + str(i+1) +'.png', 0), x_all[:,1])

            ##################################################################
            # save the reconstructed image
            # if not args.CLIC:
            # get total pixel number in an image batch
            pixel_number = np.prod(x.shape) / 3
            # get bpp & psnr & ms-ssim metric
            y_bpp = -y_likelihoods.log2().sum() / pixel_number
            z_bpp = -z_likelihoods.log2().sum() / pixel_number

            # if args.model == 'ybasef':
            fy_bpp = -fy_li.log2().sum() / pixel_number
            fz_bpp = -fz_li.log2().sum() / pixel_number

            # print(fy_bpp)
            #p rint(fz_bpp)
            psnr = get_psnr(x, x_hat, cast_to_int=True)
            ssim = get_ssim(x, x_hat)[0]
            ms_ssim = get_msssim(x, x_hat)
            list_i.append(i)
            list_b.append(y_bpp.item())
            list_m.append(ms_ssim.item())
            # update meter
            if args.model == 'ybasef':
                bpp_meter.update((y_bpp + z_bpp + fy_bpp + fz_bpp).item())
            else:
                bpp_meter.update((y_bpp + z_bpp + fy_bpp + fz_bpp).item())
                # print(y_bpp)
                # print(z_bpp)
                # print(fy_bpp)
                # print(fz_bpp)
                # bpp_meter.update((y_bpp + z_bpp).item())
            psnr_meter.update(psnr.item())
            ssim_meter.update(ssim.item())
            ms_ssim_meter.update(ms_ssim.item())
            '''
            '''
            # update the progress bar
            pbar.update(1)
        # close the progress bar
        pbar.close()

    print('BPP: {bpp_meter.avg:.5f}\t'
          'PSNR: {psnr_meter.avg:.4f}\t'
          'SSIM: {ssim_meter.avg:.4f}\t'
          'MS-SSIM: {ms_ssim_meter.avg:.5f}'.format(bpp_meter=bpp_meter,
                                                    psnr_meter=psnr_meter,
                                                    ssim_meter=ssim_meter,
                                                    ms_ssim_meter=ms_ssim_meter))

'''
    plt.plot(list_i, list_m, 'r', linewidth=2)
    plt.title(r'rate %.4f msssim %.4f' % (bpp_meter.avg, ms_ssim_meter.avg))
    plt.ylabel(r'msssim')
    plt.xlabel(r'index')
    plt.savefig('./msssim.png')
    plt.clf()

    plt.plot(list_i, list_b, 'r', linewidth=2)
    plt.title(r'rate %.4f msssim %.4f' % (bpp_meter.avg, ms_ssim_meter.avg))
    plt.ylabel(r'bpp')
    plt.xlabel(r'index')
    plt.savefig('./bpp.png')
    plt.clf()
'''

if __name__ == '__main__':
    main()
