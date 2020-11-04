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

from model.acquire_model import getModel
from dataset import VideoVimeoDataset, CodecDataset, CustomResizeAll, CustomCropAll, CustomToTensor, CustomTTS
from utils import *
from metric.ms_ssim import MSSSIM

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


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
parser.add_argument('--others', type=str, default='', help='')
parser.add_argument('--image', type=str, default='', help='')
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
parser.add_argument('--model', type=str, default='3d', help='')
parser.add_argument('--model_name', type=str, default='', help='')
parser.add_argument('--flow_start', action='store_true', help='')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--lr_d', type=int, default=0, help='')
parser.add_argument('--wd', type=float, default=0, help='')
parser.add_argument('--seed', type=int, default=123, help='')
parser.add_argument('--num', type=int, default=2, help='')
parser.add_argument('--e_num', type=int, default=2, help='')
parser.add_argument('--p_num', type=int, default=2, help='')
parser.add_argument('--is_int', action='store_true', default=False)
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--mp', action='store_true', help='')


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
    if args.is_cloud:
        # Upload
        local_tar_file = '/cache/' + os.path.basename(args.tar_file)
        print('Ready to copy...')
        print('{}---->{}'.format(args.tar_file, local_tar_file))
        mox.file.copy_parallel(args.tar_file, local_tar_file)
        print('Copy success!')
        # Decompress
        tar = tarfile.open(local_tar_file)
        tar.extractall(path='/cache')
        tar.close()
        print('Decompress success!')
        # Delete
        if os.path.isfile(local_tar_file):
            os.remove(local_tar_file)
            print('Delete train tar file success!')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    ngpus_per_node = torch.cuda.device_count()

    if args.mp:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(None, ngpus_per_node, args)
 

def main_worker(gpu, ngpus_per_node, args):
    """The main function.
    """
    args.gpu = gpu

    # create model
    auto_encoder, image_num = getModel(args)

    auto_encoder.weight_init()


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
        if args.flow_start:
            # args.start_epoch = auto_encoder.load_resume(args.resume, start=args.flow_start)
            auto_encoder.load_resume(args.resume, start=args.flow_start)
        else:
            # args.start_epoch = auto_encoder.load_resume(args.resume)
            auto_encoder.load_resume(args.resume)
    if args.others:
        auto_encoder.load_others(args.others)
    if args.image:
        auto_encoder.load_image(args.image)


    if args.mp:
        args.rank = args.rank * ngpus_per_node + gpu
        os.environ['MASTER_ADDR'] = '127.0.0.1' 
        os.environ['MASTER_PORT'] = '12345' 
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        auto_encoder = auto_encoder.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        auto_encoder = DDP(auto_encoder, device_ids=[args.gpu])
    else:
        # Parallel the model.
        auto_encoder = torch.nn.DataParallel(auto_encoder).cuda()
        

    cudnn.benchmark = True

    # Data loading code.
    train_dataset = VideoVimeoDataset(
        image_num, 
        args.data_dir,
        args.lst,
        transforms.Compose([
            CustomCropAll(args.crop_size, args.crop_number),
			CustomTTS()
            # transforms.Lambda(lambda crops: torch.stack([CustomToTensor()(crop) for crop in crops]))
        ]))

    if args.mp == True: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else: 
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for i in range(args.lr_d):
        scheduler.step()


    for epoch in range(args.start_epoch, args.epochs):
        # decay the learning rate
        scheduler.step()
        if args.mp:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, auto_encoder, criterion, optimizer, epoch, args)
        # save checkpoint
        if not args.mp or (args.mp and args.gpu == 0):
            auto_encoder.module.save_model(args.train_url, epoch)


def train(train_loader, model, criterion, optimizer, epoch, args):
    """The train function.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rate_meter = AverageMeter()
    hyper_rate_meter = AverageMeter()
    distortion_meter = AverageMeter()
    loss_meter = AverageMeter()
    flow_meter = AverageMeter()
    res_meter = AverageMeter()
    pred_meter = AverageMeter()
    rate_pred_meter = AverageMeter()
    rate_context_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    cnt = 0
    for i, (image_name, x_all) in enumerate(train_loader):
        cnt += 1
        # measure data loading time
        data_time.update(time.time() - end)

        # fuse batch size and ncrops
        bs, num, c, h, w = x_all.size()
        x_all = x_all.cuda(args.gpu, non_blocking=True)

        _, x = x_all.split([num-1, 1], dim=1)
        x = x.view(-1, c, h, w)
        # x = x_all[:,1:].contiguous().view(-1, c, h, w)

        result = model(x_all)

        '''
        for name, parms in model.named_parameters():  
            if parms.grad is not None:
                parms.grad = torch.clamp(parms.grad, min=-1, max=-1)
        '''


        if args.model == 'ybasef':
            x_tilde, y_likelihoods, z_likelihoods, mean, y_tilde, fy_li, fz_li, f, f_hat = result
        else:
            x_tilde, y_likelihoods, z_likelihoods = result
        # x_tilde, y_likelihoods, z_likelihoods, x_pred, x_pred_tilde= model(x_all)
        # get rate loss & distortion loss
        # N, C, H, W = x.shape
        y_rate_loss = -y_likelihoods.log2().sum() / (bs * h * w)
        z_rate_loss = -z_likelihoods.log2().sum() / (bs * h * w)

        if args.loss_type == 'mse':
            distortion_loss = criterion['mse'](x, x_tilde)
            distortion = 10 * np.log10(255 * 255 / distortion_loss.item())
        else:
            msssim_loss = criterion['msssim'](x, x_tilde)
            distortion_loss = 5000 * msssim_loss
            distortion = 1 - msssim_loss.item()

        '''
        pred_loss = criterion['mse'](x, x_pred)
        flow_loss = criterion['mse'](x, x_pred_tilde)
        res_loss = pred_loss
        '''

        loss = y_rate_loss + z_rate_loss + args.beta * distortion_loss 

        # add L1 reg loss 0.2 * |y - mu0
        if args.l1_reg:
            loss = loss + 0.2 * criterion['reg'](y_tilde, mean)

        if args.model == 'ybasef':
            fy_r_loss = -fy_li.log2().sum()/(bs*h*w)
            fz_r_loss = -fz_li.log2().sum()/(bs*h*w)
            flow_loss = criterion['mse'](f, f_hat)
            pred_meter.update(fy_r_loss.item(), x.size(0))
            flow_meter.update(fz_r_loss.item(), x.size(0))
            res_meter.update(flow_loss.item(), x.size(0))

            # loss = y_rate_loss * 10 + fy_r_loss + fz_r_loss + flow_loss
            # loss = y_rate_loss * 10 + flow_loss * 10 + fy_r_loss + fz_r_loss
            # loss = fy_r_loss + fz_r_loss + 100 * flow_loss
            # loss = fy_r_loss + fz_r_loss + 100 * flow_loss


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
				'FY_li {pred_meter.val:.4f} {pred_meter.avg:.4f}\t'
				'FZ_li {flow_meter.val:.4f} {flow_meter.avg:.4f}\t'
				'F_dis {res_meter.val:.4f} {res_meter.avg:.4f}\t'
				'Loss {loss_meter.val:.4f} {loss_meter.avg:.4f}'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, rate_meter=rate_meter, hyper_rate_meter=hyper_rate_meter,
					distortion_meter=distortion_meter, res_meter=res_meter, flow_meter=flow_meter, pred_meter=pred_meter, loss_meter=loss_meter))

if __name__ == '__main__':
    main()
