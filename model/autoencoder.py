import os
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.common import *
from model.entropy import *
from utils import *

class AutoEncoder(nn.Module):
    def __init__(self, channel=192, type_entropy='gaussian', scale_bound=1e-9, is_int=False, scale_max=5, is_eval=False):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(N=channel)
        self.decoder = Decoder(N=channel)

        if is_int:
            self.entropy = IntGaussianConditional(scale_max=scale_max)
        elif type_entropy == 'gaussian':
            self.entropy = GaussianConditional(scale_bound=scale_bound)
        else:
            self.entropy = LaplacianConditional(scale_bound=scale_bound)
        
        self.hyper_encoder = HyperEncoder(N=channel)

        self.hyper_decoder = HyperDecoder(N=channel)

        self.hyper_entropy = FactorizedEntropy(channels=channel)
        '''
        '''

        self.context = Context(N=channel)

        # self.fusion = Fusion2d(N=channel)

        self.gather = Gather(c_in=channel*4, c_out=channel*2)
        self.is_eval = is_eval

    def weight_init(self):
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        # self.fusion.apply(weights_init)
        self.hyper_encoder.apply(weights_init)
        self.hyper_decoder.apply(weights_init)
        self.context.apply(weights_init)
        self.gather.apply(weights_init)

    def load_checkpoint(self, checkpoint):
        print(type(checkpoint))
        checkpoint_t = OrderedDict()
        for k in checkpoint:
            if k[0] == 'e':
                checkpoint_t[k[8:]] = checkpoint[k]
        self.encoder.load_state_dict(checkpoint_t)
        checkpoint_t = OrderedDict()
        for k in checkpoint:
            if k[0] == 'd':
                checkpoint_t[k[8:]] = checkpoint[k]
        self.decoder.load_state_dict(checkpoint_t)
                
    
    def load_resume(self, file):
        epoch = 0

        if os.path.isfile(file):
            print("=> loading checkpoint '{}'".format(file))
            checkpoint = torch.load(file)

            epoch = checkpoint['epoch']

            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            # self.hyper_encoder.load_state_dict(checkpoint['hyper_encoder'])
            # self.hyper_decoder.load_state_dict(checkpoint['hyper_decoder'])
            # self.hyper_entropy.load_state_dict(checkpoint['hyper_entropy'])

            # self.context.load_state_dict(checkpoint['context'])
            # self.context.load_state_dict(checkpoint['context'])
            # self.gather.load_state_dict(checkpoint['gather'])
            print("=> loaded checkpoint '{}' (epoch {})".format(file, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(file))

        return epoch

    
    def save_model(self, folder, epoch):
        if not os.path.exists(folder):
            os.makedirs(folder)
        checkpoint_name = str(epoch) + '.pth'
        abs_checkpoint_name = os.path.join(folder, checkpoint_name)

        save_checkpoint({
            'epoch': epoch + 1,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'hyper_encoder': self.hyper_encoder.state_dict(),
            'hyper_decoder': self.hyper_decoder.state_dict(),
            'hyper_entropy': self.hyper_entropy.state_dict(),
            'context': self.context.state_dict(),
            'gather': self.gather.state_dict()
        }, abs_checkpoint_name)
    
    def get_params_list(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()}, 
                {'params': self.hyper_encoder.parameters()}, 
                {'params': self.hyper_decoder.parameters()}, 
                {'params': self.hyper_entropy.parameters()}, 
                {'params': self.context.parameters()}, 
                {'params': self.gather.parameters()}]
    
    def forward(self, x, name=''):
        '''
        x_last = x + torch.empty_like(x).uniform_(-0.5, 0.5)
        y_last = self.encoder(x_last)
        y_last = y_last.unsqueeze(2)
        psi = self.fusion(y_last)
        '''

        y = self.encoder(x)

        y_no_grad = y.detach()
        y_hat = y_no_grad.round() - y_no_grad + y
        y_tilde = y + torch.empty_like(y).uniform_(-0.5, 0.5)

        z = self.hyper_encoder(y)
        z_no_grad = z.detach()
        z_hat = z_no_grad.round() - z_no_grad + z
        z_tilde = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        '''
        '''

        # z_likelihoods = torch.ones(100).cuda()
        psi = self.hyper_decoder(z_hat)

        phi = self.context(F.pad(y_hat, (2,2,2,2), value=0))

        # mean, scale = self.gather(phi).chunk(2, dim=1)
        mean, scale = self.gather(torch.cat((phi, psi), dim=1)).chunk(2, dim=1)

        real_scale = (mean - y_hat) * (mean - y_hat)
        
        if self.is_eval:
            z_likelihoods = self.hyper_entropy(z_hat)
            y_likelihoods = self.entropy(y_hat, mean=mean, scale=scale.abs())
        else:
            z_likelihoods = self.hyper_entropy(z_tilde)
            y_likelihoods = self.entropy(y_tilde, mean=mean, scale=scale.abs())

        x_tilde = self.decoder(y_hat)

        return x_tilde, y_likelihoods, z_likelihoods
