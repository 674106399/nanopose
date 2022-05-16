import torch
import torch.nn as nn
import numpy as np
from kornia import image_to_tensor
from kornia.augmentation import (
    AugmentationSequential,
    ColorJitter,
    RandomSolarize,
    RandomMotionBlur,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomEqualize,
    RandomPosterize,
    RandomInvert,
    RandomSharpness,

)

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mode='bg') -> None:
        super().__init__()

        if mode == 'hand':
            self.jit = nn.Sequential(
                RandomMotionBlur((31, 41), 45., 1.0, p=0.1),
            )
        else:
            self.jit = nn.Sequential(
                ColorJitter(0.3, 0.3, 0.5, 0.1, p=1.0),
                # RandomPosterize(5, p=0.1),
                # RandomInvert(p=0.1),
                # RandomEqualize(p=0.1),
                RandomSharpness(p=0.1),
                # RandomSolarize(p=0.1),
                RandomMotionBlur((11, 31), 45., 1.0, p=0.1),
                RandomGaussianBlur((3, 7), (0.1, 2.0), p=0.15),
            )
        # self.transforms = AugmentationSequential(
        #     # RandomMotionBlur(11, 45., 0.5, p=0.3),
        #     # RandomGaussianBlur((3,3), (0.1, 2.0), p=0.3),
        #     RandomAffine(45, (0.1, 0.1), (0.7, 1.2), p=1.),
        #     data_keys=['input', 'keypoints'],
        #     return_transform=False,
        #     # random_apply=1
        # )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.jit(x)
        # x = self.transforms(x)  # BxCxHxW
        return x

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x_tmp = np.array(x)  # HxWxC
        x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.