import os
import numpy as np
import torch
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset

class VideoSeqDataset(Dataset):
    def __init__(self, num, root, lst, transform=None):
        """
        Args:
            lst (string): Path list.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lst = lst
        self.root = root
        self.transform = transform
        self.sequence = []

        with open(self.lst, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            self.sequence.append(line.strip().split(' ')[0])

        self.num = num

    def __getitem__(self, index):
        """Transform per element to image.
        """
        path = self.sequence[index]
        images = []

        st = index
        print(st)

        for i in range(st, st + self.num):
            # abs_path = os.path.join(self.root, path, 'im' + str(i) + '.png')
            abs_path = os.path.join(self.root, path, 'im' + str(i) + '.png')
            image = Image.open(abs_path).convert('RGB')

            images.append(image)
        
        if self.transform:
            image = self.transform(images)
            
        return path, image

    def __len__(self):
        """Get the length of the dataset.
        """
        return len(self.sequence)



class VideoVimeoDataset(Dataset):
    """vimeo-90k"""
    def __init__(self, num, root, lst, transform=None, test=0):
        """
        Args:
            lst (string): Path list.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lst = lst
        self.root = root
        self.transform = transform
        self.sequence = []

        with open(self.lst, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            self.sequence.append(line.strip().split(' ')[0])

        self.num = num
        self.test = test

    def blur(self, image, kernel_size, sigma):
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        blur_PIL = Image.fromarray(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
        return blur_PIL


    def __getitem__(self, index):
        """Transform per element to image.
        """
        path = self.sequence[index]
        images = []

        if self.test == 0:
            st = random.randint(1, 8-self.num)
        elif self.test == 1:
            st = random.randint(1, 8-self.num)
        elif self.test == 2:
            st = index % 594
        # st = 0 + index
        # st = 593
        jump=1
        # st = index * 30


        for i in range(0, self.num):
            # abs_path = os.path.join(self.root, path, 'im' + str(i) + '.png')
            abs_path = os.path.join(self.root, path, 'im' + str(st * jump + i*jump) + '.png')
            # image = self.blur(Image.open(abs_path).convert('RGB'), 3, 1.5)
            image = Image.open(abs_path).convert('RGB')
            w, h = image.size

            images.append(image)
        
        if self.transform:
            image = self.transform(images)
            
        return path, image

    def __len__(self):
        """Get the length of the dataset.
        """
        return len(self.sequence)





class CodecDataset(Dataset):
    """Codec Dataset."""

    def __init__(self, root, lst, transform=None):
        """
        Args:
            lst (string): Path list.
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lst = lst
        self.root = root
        self.transform = transform
        self.sequence = []

        with open(self.lst, 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.sequence.append(line.strip().split(' ')[0])

    def __getitem__(self, index):
        """Transform per element to image.
        """
        path = self.sequence[index]
        abs_path = os.path.join(self.root, path)
        image = Image.open(abs_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return path, image

    def __len__(self):
        """Get the length of the dataset.
        """
        return len(self.sequence)

class CustomTTS(object):
    """To tensor and stack
    Converts each PIL Image (H x W x C) in the range [0, 255] to 
    a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0],
    then stack those tensor.
    """

    def __call__(self, pics):
        """
        Args:
            pics (PIL Image): Images to be converted to tensor.
        Returns:
            Tensor: Tensor stacked by converted image.
        """
        return custom_to_tensor_and_stack(pics)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def custom_to_tensor_and_stack(pics):
    """To tensor and stack
    Args:
        pics (PIL Image): Images to be converted to tensor.
    Returns:
        Tensor: Tensor stacked by converted image.
    """
    # handle PIL Image
    tensor_list = []
    for pic in pics:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        img = img.transpose(0, 1).transpose(0, 2).contiguous().float()
        tensor_list.append(img)
    return torch.stack(tensor_list)



class CustomToTensor(object):
    """Convert a ``PIL Image`` to tensor.
    Converts a PIL Image (H x W x C) in the range [0, 255] to 
    a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return custom_to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. See ``ToTensor`` for more details.
    Args:
        pic (PIL Image): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    # handle PIL Image
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float()

class CustomCropAll(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (int): Desired output size of the crop, a square crop (size, size) is made.
        num  (int): The number of regions to be croped from the raw image.
    """

    def __init__(self, size, num):
        self.size = (size, size)
        self.num = num

    @staticmethod
    def get_params(img, output_size, output_num):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img     (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
            output_num    (int): Expected number of regions.
        Returns:
            coordinates  (list): Store all the coordinates to be cropped.
        """
        w, h = img.size
        th, tw = output_size

        coordinates = []

        for _ in range(output_num):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            coordinates.append((i, j, th, tw))

        return coordinates

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            image_container (list): The list storing all the cropped images.
        """
        image_container = []

        for coordinate in self.get_params(imgs[0], self.size, self.num):
            for img in imgs:
                image_container.append(custom_crop(img, *coordinate))

        return image_container

    def __repr__(self):
        return self.__class__.__name__ + '()'



class CustomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (int): Desired output size of the crop, a square crop (size, size) is made.
        num  (int): The number of regions to be croped from the raw image.
    """

    def __init__(self, size, num):
        self.size = (size, size)
        self.num = num

    @staticmethod
    def get_params(img, output_size, output_num):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img     (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
            output_num    (int): Expected number of regions.
        Returns:
            coordinates  (list): Store all the coordinates to be cropped.
        """
        w, h = img.size
        th, tw = output_size

        coordinates = []

        for _ in range(output_num):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            coordinates.append((i, j, th, tw))

        return coordinates

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            image_container (list): The list storing all the cropped images.
        """
        image_container = []

        for coordinate in self.get_params(img, self.size, self.num):
            image_container.append(custom_crop(img, *coordinate))

        return image_container

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))

class CustomResizeAll(object):
    """Resize the input PIL Image to the given size.
    Args:
        threshold (sequence): Desired output size. =>(1024, 512, 256)
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    """

    def __init__(self, threshold, interpolation=Image.BILINEAR):
        self.threshold = threshold
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        w, h = imgs[0].size
        short = min(w, h)
        size = self.threshold[-1]
        for t in self.threshold:
            if short > t:
                size = t
                break
        new_imgs = []
        for img in imgs:
            new_imgs.append(custom_resize(img, size, self.interpolation))
        return new_imgs

    def __repr__(self):
        return self.__class__.__name__ + '()'



class CustomResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        threshold (sequence): Desired output size. =>(1024, 512, 256)
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    """

    def __init__(self, threshold, interpolation=Image.BILINEAR):
        self.threshold = threshold
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        short = min(w, h)
        size = self.threshold[-1]
        for t in self.threshold:
            if short > t:
                size = t
                break
        return custom_resize(img, size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def custom_resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (int): Desired output size.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
