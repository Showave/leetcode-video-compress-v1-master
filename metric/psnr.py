import torch
import torch.nn as nn
import skimage.measure


def get_psnr_by_skimage(img1, img2):
    """Get PSNR through the skimage lib.
    Args:
        img1 (N3HW): original image
        img2 (N3HW): reconstructed image
    Returns:
        a ndarray (the average psnr value)
    """
    img1 = img1.to(dtype=torch.uint8)
    img2 = img2.to(dtype=torch.uint8)

    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    return skimage.measure.compare_psnr(img1, img2)


def get_psnr(img1, img2, cast_to_int=False):
    """Get PSNR through custom logic.
    Args:
        img1 (N3HW): original image
        img2 (N3HW): reconstructed image
        cast_to_int: for evaluating
    Returns:
        a tensor (the average psnr value)
    """
    if cast_to_int:
        img1 = img1.int()
        img2 = img2.int()
    error = (img1 - img2) * (img1 - img2)
    mse_per_image = torch.mean(error.float(), dim=(1, 2, 3))
    psnr_per_image = 10 * torch.log10(255 * 255 / mse_per_image)
    psnr = torch.mean(psnr_per_image)
    return psnr


class PSNR(nn.Module):
    """Wrap for PSNR metric.
    """

    def __init__(self, cast_to_int=False):
        super(PSNR, self).__init__()
        self.cast_to_int = cast_to_int

    def forward(self, img1, img2):
        return 100 - get_psnr(img1, img2, self.cast_to_int)
