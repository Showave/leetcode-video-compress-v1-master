#import moxing as mox
#mox.file.shift('os', 'mox')

import os
import time
import torch
import random
import tarfile
import argparse
import torch.optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_ as cgn

from model.autoencoder import AutoEncoder
from dataset import CodecDataset, CustomResize, CustomCrop, CustomToTensor
from utils import *
from metric.ms_ssim import MSSSIM

parser = argparse.ArgumentParser(description='Deep Image Compression Trainer.')
# using modelarts for training
parser.add_argument('--is_cloud', action='store_true', help='')
parser.add_argument('--tar_file', type=str, default='', help='')
parser.add_argument('--data_url', type=str, default='', help='')
# parameters of dataset
parser.add_argument('--lst', type=str, default='', help='')
parser.add_argument('--data_dir', type=str, default='', help='')
parser.add_argument('--train_url', type=str, default='', help='')
parser.add_argument('--resume', type=str, default='', help='')
# training parameters
parser.add_argument('--N', type=int, default=192, help='')
parser.add_argument('--beta', type=float, default=0.01, help='')
parser.add_argument('--workers', type=int, default=4, help='')
parser.add_argument('--epochs', type=int, default=6, help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--crop_size', type=int, default=256, help='')
parser.add_argument('--crop_number', type=int, default=1, help='')
parser.add_argument('--print_freq', type=int, default=10, help='')
parser.add_argument('--l1_reg', action='store_true', help='')
parser.add_argument('--scale_bound', type=float, default=1e-9, help='')
parser.add_argument('--scale_max', type=float, default=5, help='')
parser.add_argument('--entropy', type=str,
                    choices=['gaussian', 'laplacian'], default='gaussian', help='')
parser.add_argument('--loss_type', type=str,
                    choices=['mse', 'msssim'], default='mse', help='')
# parameters of optimazition
parser.add_argument('--lr_steps', type=str, default='4', help='')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--wd', type=float, default=0, help='')
parser.add_argument('--seed', type=int, default=123, help='')
parser.add_argument('--is_int', action='store_true', default=False)

args, unparsed = parser.parse_known_args()

# Decompress the tar file to cache.
if args.is_cloud:
    local_tar_file = '/cache/' + os.path.basename(args.tar_file)
    # get the path of download.py & download
    exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download.py')
    mox_copy_with_timeout_retry(args.tar_file, local_tar_file, 3, 7200, exe_path, 'True')
    # decompress
    tar = tarfile.open(local_tar_file)
    tar.extractall(path='/cache')
    tar.close()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    """The main function.
    """
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # create model
    auto_encoder = AutoEncoder(channel=args.N, type_entropy=args.entropy, scale_bound=args.scale_bound)
    auto_encoder.to(device)

    # define loss function (criterion) and optimizer
    criterion = {}
    criterion['mse'] = nn.MSELoss()
    criterion['msssim'] = MSSSIM()
    criterion['reg'] = nn.L1Loss()

    optimizer = torch.optim.Adam(auto_encoder.get_params_list(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        args.start_epoch = auto_encoder.load_resume(args.resume)

    # Parallel the model.
    auto_encoder = torch.nn.DataParallel(auto_encoder)

    cudnn.benchmark = True

    # Data loading code.
    train_dataset = CodecDataset(
        args.data_dir,
        args.lst,
        transforms.Compose([
            CustomResize([1024, 512, 256]),
            transforms.RandomHorizontalFlip(p=0.5),
            CustomCrop(args.crop_size, args.crop_number),
            transforms.Lambda(lambda crops: torch.stack([CustomToTensor()(crop) for crop in crops]))
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True,
        num_workers=args.workers * torch.cuda.device_count(), pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        # decay the learning rate
        scheduler.step()
        # train for one epoch
        train(train_loader, auto_encoder, criterion, optimizer, epoch)
        # save checkpoint
        auto_encoder.module.save_model(args.train_url, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    """The train function.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rate_meter = AverageMeter()
    hyper_rate_meter = AverageMeter()
    distortion_meter = AverageMeter()
    loss_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    cnt = 0
    for i, (_, x) in enumerate(train_loader):
        cnt += 1
        # measure data loading time
        data_time.update(time.time() - end)

        # fuse batch size and ncrops
        bs, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = x.to(device, non_blocking=True)

        x_tilde, y_likelihoods, z_likelihoods = model(x)

        # get rate loss & distortion loss
        N, C, H, W = x.shape
        y_rate_loss = -y_likelihoods.log2().sum() / (N * H * W)
        z_rate_loss = -z_likelihoods.log2().sum() / (N * H * W)

        if args.loss_type == 'mse':
            distortion_loss = criterion['mse'](x, x_tilde)
            distortion = 10 * np.log10(255 * 255 / distortion_loss.item())
        else:
            msssim_loss = criterion['msssim'](x, x_tilde)
            distortion_loss = 5000 * msssim_loss
            distortion = 1 - msssim_loss.item()

        loss = y_rate_loss + z_rate_loss + args.beta * distortion_loss

        # add L1 reg loss 0.2 * |y - mu|
        if args.l1_reg:
            loss = loss + 0.2 * criterion['reg'](y_tilde, mean)

        rate_meter.update(y_rate_loss.item(), x.size(0))
        hyper_rate_meter.update(z_rate_loss.item(), x.size(0))
        distortion_meter.update(distortion, x.size(0))
        loss_meter.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} {batch_time.avg:.3f}\t'
                  'Data {data_time.val:.3f} {data_time.avg:.3f}\t'
                  'Rate {rate_meter.val:.4f} {rate_meter.avg:.4f}\t'
                  'HyperRate {hyper_rate_meter.val:.4f} {hyper_rate_meter.avg:.4f}\t'
                  'Distortion {distortion_meter.val:.4f} {distortion_meter.avg:.4f}\t'
                  'Loss {loss_meter.val:.4f} {loss_meter.avg:.4f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, rate_meter=rate_meter, hyper_rate_meter=hyper_rate_meter,
                      distortion_meter=distortion_meter, loss_meter=loss_meter))

if __name__ == '__main__':
    main()
