import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.gdn import GDN, GDN3d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=2):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

def conv3x3_t(in_planes, out_planes, stride=2):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)

def conv5x5_t(in_planes, out_planes, stride=2):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, output_padding=1, bias=False)

class MaskedConv2d(nn.Conv2d):
    """
     ------------------------------------
    |  1       1       1       1       1 |
    |  1       1       1       1       1 |
    |  1       1    1 if B     0       0 |   H // 2
    |  0       0       0       0       0 |   H // 2 + 1
    |  0       0       0       0       0 |
     ------------------------------------
       0       1     W//2    W//2+1
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, H, W = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, H // 2, W // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, H // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class Encoder(nn.Module):

    def __init__(self, N=192, num = 1, flow=False, res=False):
        super(Encoder, self).__init__()
        self.flow = flow
        self.res = res
        c_in = 3 * num
        if flow:
            c_in = 2
        self.conv1 = conv5x5(c_in, N)
        self.gdn1 = GDN(N)
        self.conv2 = conv5x5(N, N)
        self.gdn2 = GDN(N)
        self.conv3 = conv5x5(N, N)
        self.gdn3 = GDN(N)
        self.conv4 = conv5x5(N, N)

    def set(self, temp):
        self.conv2.weight.data = temp.conv2.weight.data
        self.conv3.weight.data = temp.conv3.weight.data
        self.conv4.weight.data = temp.conv4.weight.data

    def forward(self, x):
        if self.flow:
            x = normalize_f(x)
        elif self.res:
            x = normalize_r(x)
        else:
            x = normalize(x)

        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x

class Decoder(nn.Module):

    def __init__(self, N=192, num = 1, flow=False, res=False):
        super(Decoder, self).__init__()
        c_out = 3 * num
        if flow:
            c_out = 2
        self.flow = flow
        self.res = res
        self.conv1_t = conv5x5_t(N, N)
        self.igdn1 = GDN(N, inverse=True)
        self.conv2_t = conv5x5_t(N, N)
        self.igdn2 = GDN(N, inverse=True)
        self.conv3_t = conv5x5_t(N, N)
        self.igdn3 = GDN(N, inverse=True)
        self.conv4_t = conv5x5_t(N, c_out)

    def set(self, temp):
        self.conv2_t.weight.data = temp.conv2_t.weight.data
        self.conv3_t.weight.data = temp.conv3_t.weight.data
        self.conv1_t.weight.data = temp.conv1_t.weight.data

    def forward(self, x):
        x = self.igdn1(self.conv1_t(x))
        x = self.igdn2(self.conv2_t(x))
        x = self.igdn3(self.conv3_t(x))
        x = self.conv4_t(x)
        if self.flow:
            x = denormalize_f(x)
        elif self.res:
            x = denormalize_r(x)
            x = clip_res(x)
        else:
            x = denormalize(x)
            x = clip_image(x)

        return x

class HyperEncoder(nn.Module):

    def __init__(self, N=192):
        super(HyperEncoder, self).__init__()
        self.conv1 = conv3x3(N, N)
        self.conv2 = conv5x5(N, N)
        self.conv3 = conv5x5(N, N)

    def forward(self, x):
        x = x.abs()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x


class HyperDecoder(nn.Module):

    def __init__(self, N=192):
        super(HyperDecoder, self).__init__()
        self.conv1 = conv5x5_t(N, N)
        self.conv2 = conv5x5_t(N, N + N //2)
        self.conv3 = conv3x3(N + N //2, N * 2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x

class HyperDecoder_0(nn.Module):

    def __init__(self, N=192):
        super(HyperDecoder_0, self).__init__()
        self.conv1 = conv5x5_t(N, N)
        self.conv2 = conv5x5_t(N, N)
        self.conv3 = conv3x3(N, N)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Gather2_mini(nn.Module):

    def __init__(self, c_in=192*4, c_out=192*2):
        super(Gather2_mini, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_in * 2, 1, bias=False)
        self.conv3 = nn.Conv2d(c_in * 2, c_in * 1, 1, bias=False)
        self.conv4 = nn.Conv2d(c_in * 1, c_in * 1, 1, bias=False)
        self.conv5 = nn.Conv2d(c_in * 1, c_out * 2, 1, bias=False)
        self.conv6 = nn.Conv2d(c_out * 2, c_out * 1, 1, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out)) + x
        out = F.leaky_relu(self.conv5(out))
        out = self.conv6(out)
        return out


class Gather2(nn.Module):

    def __init__(self, c_in=192*4, c_out=192*2):
        super(Gather2, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in * 2, 1, bias=False)
        self.conv2 = nn.Conv2d(c_in * 2, c_in * 4, 1, bias=False)
        self.conv3 = nn.Conv2d(c_in * 4, c_in * 2, 1, bias=False)
        self.conv4 = nn.Conv2d(c_in * 2, c_in * 1, 1, bias=False)
        self.conv5 = nn.Conv2d(c_in * 1, c_out * 2, 1, bias=False)
        self.conv6 = nn.Conv2d(c_out * 2, c_out * 1, 1, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out)) + x
        out = F.leaky_relu(self.conv5(out))
        out = self.conv6(out)
        return out

class Gather(nn.Module):

    def __init__(self, c_in=192*4, c_out=192*2):
        super(Gather, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 640, 1, bias=False)
        self.conv2 = nn.Conv2d(640, 512, 1, bias=False)
        self.conv3 = nn.Conv2d(512, c_out, 1, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Context(nn.Module):

    def __init__(self, N=192, out=2, mask_type='A', kernel_size=5):
        super(Context, self).__init__()
        self.mask_conv = MaskedConv2d(mask_type, N, out * N, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        x = self.mask_conv(x)
        return x

class Fusion2d(nn.Module):
    def __init__(self, N=192, num=1, out=2):
        super(Fusion2d, self).__init__()
        self.conv1 = nn.Conv2d(N * num, N*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N * 2, N*4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(N * 4, N*out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if x.dim()==5:
            b, c, n, h, w = x.shape
            x = x.view(b, c * n, h, w)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        return x;

class FusionMask(nn.Module):
    def __init__(self, N=192, num=1, out=2, mask_type='B'):
        super(FusionMask, self).__init__()
        self.mask_conv1 = MaskedConv2d(mask_type, N * num, N*2, kernel_size=3, bias=False)
        self.mask_conv2 = MaskedConv2d('B', N*2, N*4, kernel_size=3, bias=False)
        self.mask_conv3 = MaskedConv2d('B', N*4, N*out, kernel_size=3, bias=False)

    def forward(self, x):
        if x.dim()==5:
            b, c, n, h, w = x.shape
            x = x.view(b, c * n, h, w)

        x = F.leaky_relu(self.mask_conv1(F.pad(x, (1,1,1,1), value=0)))
        x = F.leaky_relu(self.mask_conv2(F.pad(x, (1,1,1,1), value=0)))
        x = self.mask_conv3(F.pad(x, (1,1,1,1), value=0))
        return x

def get_x_grid(flow, w, h):
    gridX, gridY = np.meshgrid(np.arange(w), np.arange(h))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()

    u = flow[:,0]
    v = flow[:,1]
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v

    x = 2 * (x / (w - 1) - 0.5)
    y = 2 * (y / (h - 1) - 0.5)
    grid = torch.stack((x, y), dim=3)
    return grid


def get_y_grid(flow, w, h, c):
    gridX, gridY = np.meshgrid(np.arange(w), np.arange(h))
    gridZ = np.arange(c)
    gridX = torch.tensor(gridX, requires_grad=False).cuda().unsqueeze(0)
    gridY = torch.tensor(gridY, requires_grad=False).cuda().unsqueeze(0)
    gridZ = torch.tensor(gridZ, requires_grad=False).cuda().unsqueeze(1).unsqueeze(2)

    d = flow[:,0]
    u = flow[:,1]
    v = torch.zeros_like(flow[:,1]).cuda()

    z = gridZ.unsqueeze(0).expand_as(d).float()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v

    x = 2 * (x / (w - 1) - 0.5)
    y = 2 * (y / (h - 1) - 0.5)
    z = 2 * (z / (c - 1) - 0.5)
    grid = torch.stack((x, y, z), dim=4)

    return grid  

def normalize_f(x):
    return x / 10


def denormalize_f(x):
    return x * 10

def normalize_r(x):
    return x / 100

def denormalize_r(x):
    return x * 100

def clip_res(x):
    return torch.clamp(x, -256, 255)

def get_hat(x):
    return x.detach().round() - x.detach() + x

def get_tilde(x):
    return x + torch.empty_like(x).uniform_(-0.5, 0.5)

def normalize(x):
    return (x - 127.5) / 128

def denormalize(x):
    return x * 128 + 127.5


def clip_image(x):
    return torch.clamp(x, 0, 255)

def get_hat(x):
    return x.detach().round() - x.detach() + x

def get_tilde(x):
    return x + torch.empty_like(x).uniform_(-0.5, 0.5)
