import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def mkdir(path): 
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:

        return False

# [batch, c, h, w]
def get_local_mini(feat):
    b, c, h, w = feat.shape
    feat = feat.view(b * c, 1, h, w)

    mini_conv = torch.zeros([9, 1, 3, 3]).cuda()
    for i in range(3):
        for j in range(3):
            if i != 1 or j != 1:
                mini_conv[i * 3 + j,0,i,j]=-1
                mini_conv[i * 3 + j,0,1,1] = 1
    mini_conv[4, 0, 0, 0] = -1
    mini_conv[4, 0, 1, 1] = 1
    
    dis_feat = -torch.abs(F.conv2d(feat, mini_conv,padding=1))
    pool = nn.MaxPool2d(3)
    dis_feat = dis_feat.transpose(1,2).transpose(2,3).contiguous().view(b*c*h*w, 1, 3, 3)
    mini_feat= -pool(dis_feat).view(b, c, h, w)
    return mini_feat

    



def save_feature_c(feat, name, id):
    # for i in range(feat.shape[1]):
    feature = feat[:, id, :, :].cpu()
    feature = feature.view(feature.shape[1], feature.shape[2])
    feature = feature.data.numpy()
    # use sigmod to [0,1]
    feature = 1.0 / (1 + np.exp(-1 * feature))
    # to [0,255]
    feature = np.round(feature * 255)
    # mkdir('./feature/' + str(i))
    cv2.imwrite('./feature/' + name, feature)


def save_feature(feat, name):
    # for i in range(feat.shape[1]):
     for i in range(feat.shape[1]):
        feature = feat[:, i, :, :].cpu()
        feature = feature.view(feature.shape[1], feature.shape[2])
        feature = feature.data.numpy()
        # use sigmod to [0,1]
        feature = 1.0 / (1 + np.exp(-1 * feature))
        # to [0,255]
        feature = np.round(feature * 255)
        mkdir('./feature/' + str(i))
        cv2.imwrite('./feature/' +str(i)+ '/' + name, feature)



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

def save_checkpoint(state, filename='checkpoint.pth'):
    """Save the checkpoint.
    """
    torch.save(state, filename)

def weights_init(m):
    """Init the weights.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



def to_one_hot(t, depth):
    """One hot transform.
    Args:
        t  (tensor): the tensor to be transformed
        depth (int): the class number in one hot label
    Returns:
        a new tensor with one hot value
    """
    one_hot_shape = list(t.shape)
    one_hot_shape.append(depth)
    t = t.unsqueeze(dim=-1)
    one_hot = torch.zeros(one_hot_shape).cuda()
    return one_hot.scatter(-1, t, 1)


def save_image(image_dir, image_name, image_data):
    """Save image with `png` format.
    Args:
        image_dir  (str): image dir to be saved
        image_name (str): image name to be saved
        image_data (tensor): the N3HW tensor
    """
    image_name = image_name[0]
    assert image_name.endswith('.png')
    assert image_data.dim() == 4 and image_data.shape[0] == 1 and image_data.shape[1] == 3
    abs_image_name = os.path.join(image_dir, image_name)
    os.makedirs(os.path.dirname(abs_image_name), exist_ok=True)
    # float => uint8
    image_data = image_data.to(dtype=torch.uint8)
    # CUDA tensor => ndarray
    image_data = image_data.cpu().numpy()
    # NCHW => HWC
    image_data = np.transpose(image_data[0, :, :, :], (1, 2, 0))
    image = Image.fromarray(image_data)
    image.save(abs_image_name)


def mox_copy_with_timeout_retry(src_data, dst_data, retry_num, timeout, path, file_or_not):
    """Use this Func to substitude moxi.file.copy and moxi.file.copy_parallel.
    `("s3://wolfros-net/datasets/imagenet.tar", "/cache/imagenet.tar", 4, 3600, path, "True") -> None`
    """
    status = 0
    # set the cmd string to excute
    cmd = 'timeout %(timeout)s python %(path)s %(src)s %(dst)s %(file_or_not)s' % {
        'timeout': timeout, 'src': src_data, 'dst': dst_data, 'path': path, 'file_or_not': file_or_not}
    print(cmd)
    for i in range(0, retry_num):
        ret = os.system(cmd)
        if ret == 0:
            print('copy success')
            status = 1
            break
        print('ret: %d retry' % (i + 1))
    if status != 1:
        print('copy fail exit')
        return 1
    else:
        return 0
