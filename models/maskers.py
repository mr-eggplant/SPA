import torch
import math
import warnings
import torch.nn.functional as FF
import torchvision.transforms.functional as F
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np


class LearnablePatchErasing(torch.nn.Module):
    """ Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, patch_size=16, img_size=224, noise_ratio=0.4):
        super().__init__()
        self.img_size = img_size

        self.patch_size = patch_size

        h = w = self.img_size // self.patch_size
        self.alpha = nn.Parameter(torch.ones(h * w)* math.sqrt(noise_ratio), requires_grad=True)

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        assert img.shape[-1] == img.shape[-2] and img.shape[-1] % self.patch_size == 0

        img_c = img.shape[-3]
        h = w = img.shape[-1] // self.patch_size # h 是个数
        patches = [(i*self.patch_size, j*self.patch_size) for i in range(h) for j in range(w)]
        new_img = torch.empty_like(img)
        for _, patch in enumerate(patches):
            v = torch.empty([img_c, self.patch_size, self.patch_size], dtype=torch.float32, device=img.device).normal_()
            new_img[...,patch[0]:patch[0]+self.patch_size, patch[1]:patch[1]+self.patch_size] = \
                (1 - self.alpha[_]**2) * img[..., patch[0]:patch[0]+self.patch_size, patch[1]:patch[1]+self.patch_size] + self.alpha[_]**2 * v
        new_img.clip_(-1, 1)
        return new_img