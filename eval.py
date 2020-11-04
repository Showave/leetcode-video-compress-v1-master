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
from model.autoencoder import AutoEncoder
from dataset import CodecDataset, CustomToTensor
from utils import *
from metric.psnr import get_psnr
from metric.ms_ssim import get_ssim, get_msssim

parser = argparse.ArgumentParser(description='Deep Image Compression Evaluator.')
parser.add_argument('--data_dir', type=str, default='', help='')
parser.add_argument('--lst', type=str, default='', help='')
parser.add_argument('--resume', type=str, default='', help='')
parser.add_argument('--output_dir', type=str, default='', help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--scale_max', type=float, default=5, help='')
parser.add_argument('--hyper_gaussian', action='store_true', help='')
parser.add_argument('--N', type=int, default=192, help='')
parser.add_argument('--CLIC', action='store_true', help='')
parser.add_argument('--scale_bound', type=float, default=1e-9, help='')
parser.add_argument('--is_int', action='store_true', default=False)
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
    auto_encoder = AutoEncoder(channel=args.N, type_entropy=args.entropy, scale_bound=args.scale_bound)
    auto_encoder.to(device)

    # restore from a checkpoint
    auto_encoder.load_resume(args.resume)

    cudnn.benchmark = True

    # Data loading code.
    eval_dataset = CodecDataset(args.data_dir, args.lst, CustomToTensor())

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
        for i, (image_name, x) in enumerate(eval_loader):
            x = x.to(device, non_blocking=True)
            # padding for invalid resolution
            _, _, h, w = x.shape
            w_diff = int(math.ceil(w / 64) * 64) - w
            h_diff = int(math.ceil(h / 64) * 64) - h
            x_inflate = F.pad(x, (w_diff, 0, h_diff, 0),  mode='replicate')

            x_tilde, y_likelihoods, z_likelihoods = model(x_inflate, image_name[0])
            x_hat = x_tilde.round()
            x_hat = x_hat[:, :, h_diff:, w_diff:]

            ##################################################################
            # save the reconstructed image
            # if not args.CLIC:
                # save_image(args.output_dir, image_name, x_hat)
            # get total pixel number in an image batch
            pixel_number = np.prod(x.shape) / 3
            # get bpp & psnr & ms-ssim metric
            y_bpp = -y_likelihoods.log2().sum() / pixel_number
            z_bpp = -z_likelihoods.log2().sum() / pixel_number

            psnr = get_psnr(x, x_hat, cast_to_int=True)
            ssim = get_ssim(x, x_hat)[0]
            ms_ssim = get_msssim(x, x_hat)
            # update meter
            bpp_meter.update((y_bpp + z_bpp).item())
            psnr_meter.update(psnr.item())
            ssim_meter.update(ssim.item())
            ms_ssim_meter.update(ms_ssim.item())

            list_i.append(i)
            list_b.append((y_bpp + z_bpp).item())
            list_m.append(ms_ssim.item())

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

if __name__ == '__main__':
    main()
