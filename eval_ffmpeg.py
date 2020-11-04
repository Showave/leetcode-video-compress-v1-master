import torch
import subprocess
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from metric.psnr import get_psnr
from metric.ms_ssim import get_ssim, get_msssim
import argparse
import cv2

parser = argparse.ArgumentParser(description='Deep Image Compression Evaluator.')
parser.add_argument('--rec_dir', type=str, default='', help='')
parser.add_argument('--ori_dir', type=str, default='', help='')
args, unparsed = parser.parse_known_args()

def eval(path_com, path_ori):
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    ms_ssim_meter = AverageMeter()
    length = 99
    pixelnum = 1920 * 1080
    # path_ori = '/home/shiyibo/dataset/UVG/ShakeNDry/'
    # path_com = '/home/shiyibo/dataset/UVG/ShakeNDry_h265/crf_25/'
    # path_ori = '/cache/yibo/code/VTM/bin/rec/image/'

    x = []
    y = []
    '''

    output = subprocess.check_output('ffprobe -show_frames ' + path_com + 'out_c.mkv | grep pkt_size', shell=True)
    img_size_str = output.decode('utf-8').split('\n')
    img_size = []
    sum_bpp = 0

    avg_loss = []
    cnt= 0
    I = 1000

    for line in img_size_str:
        if len(line) >= 1 and line[0] == 'p':
            if cnt % I != -1 and cnt >=0 :
                size = int(line[9:])
                img_size.append(size/pixelnum * 8)
                sum_bpp += size/pixelnum * 8
                print(size/pixelnum * 8)
            else:
                length-=1
            cnt += 1

    '''
    sum_msssim=0
    sum_psnr=0
    I=10000

    cur = 0
    for i in range(0, length):
        if i % I != -1:
            im1 = cv2.imread(path_ori + 'im' + str(i+2) + '.png')
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

            im2 = cv2.imread(path_com + 'im' + str(i+1) + '.png')
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    
            im1_t = torch.from_numpy(im1.transpose((2, 0, 1))).unsqueeze(0).float().cuda()
            im2_t = torch.from_numpy(im2.transpose((2, 0, 1))).unsqueeze(0).float().cuda()
    
            psnr = get_psnr(im1_t, im2_t, cast_to_int=True)
            # ssim = get_ssim(im1_t, im2_t)[0]
            ms_ssim = get_msssim(im1_t, im2_t)
            sum_psnr += psnr.item()
            sum_msssim += ms_ssim.item()
            # x.append(cur)
            # y.append(ms_ssim.item())
            # avg_loss.append(img_size[cur] * 1/10- y[cur] + 1)
            cur += 1
    print(sum_msssim/length)
    print(sum_psnr/length)
'''
    
    plt.plot(x, y, 'r', linewidth=2)
    plt.title(r'rate %.4f msssim %.4f' % (sum_bpp/length, sum_msssim/length))
    plt.ylabel(r'msssim')
    plt.xlabel(r'index')
    plt.savefig(path_com + './rate_%.4f_msssim.png' % (sum_bpp/length))
    plt.clf()

    plt.plot(x, img_size, 'r', linewidth=2)
    plt.title(r'rate %.4f msssim %.4f' % (sum_bpp/length, sum_msssim/length))
    plt.ylabel(r'bpp')
    plt.xlabel(r'index')
    plt.savefig(path_com + './rate_%.4f_bit.png' % (sum_bpp/length))
    plt.clf()

    plt.plot(x, avg_loss, 'r', linewidth=2)
    plt.title(r'rate %.4f msssim %.4f' % (sum_bpp/length, sum_msssim/length))
    plt.ylabel(r'loss')
    plt.xlabel(r'index')
    plt.savefig(path_com + './rate_%.4f_loss.png' % (sum_bpp/length))
    plt.clf()


    list1 = [0.3930, 0.4600, 0.4919, 0.5181, 0.5197]
    list2 = [0.9664, 0.9460, 0.9336, 0.9256, 0.9294]



path_com = '/home/shiyibo/dataset/construct/pts_4_4/'
x = [1, 2, 3, 4, 5]
list1 = [0.3930, 0.4600, 0.4919, 0.5181, 0.5197]
list2 = [0.9664, 0.9460, 0.9336, 0.9256, 0.9294]

plt.plot(x, list1, 'r', linewidth=2)
plt.title(r'speed to bbp')
plt.ylabel(r'bbp')
plt.xlabel(r'length')
plt.savefig(path_com + './list1.png')
plt.clf()

plt.plot(x, list2, 'r', linewidth=2)
plt.title(r'speed to msssim')
plt.ylabel(r'msssim')
plt.xlabel(r'length')
plt.savefig(path_com + './list2.png')
plt.clf()
'''

path_com = args.rec_dir
path_ori = args.ori_dir
eval(path_com, path_ori)

'''
com=["BasketballPass_22","BlowingBubbles_22","BQSquare_22","RaceHorses_22","BasketballPass_27","BlowingBubbles_27","BQSquare_27","RaceHorses_27","BasketballPass_32","BlowingBubbles_32","BQSquare_32","RaceHorses_32"]
ori=["BasketballPass","BlowingBubbles","BQSquare","RaceHorses","BasketballPass","BlowingBubbles","BQSquare","RaceHorses","BasketballPass","BlowingBubbles","BQSquare","RaceHorses"]
for i in range(len(com)):
    print(com[i])
    path_com = '/cache/yibo/code/VTM/bin/image/' + com[i] +'/'
    path_ori = '/cache/yibo/datasets/HEVC_video/image/'+  ori[i] + '/'
    eval(path_com, path_ori)
'''

