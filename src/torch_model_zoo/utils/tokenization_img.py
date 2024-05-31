#######################################################################################################################
# This script contains image tokenization approaches.                                                                 #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.11.7                                                                                        #
# PyTorch Version:      2.3.0                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################
from typing import Tuple

import torch
from torch import nn


class ImgTokenizer(nn.Module):
    """
    This class tokenizes images.
    """

    def __init__(
        self,
        size_img: Tuple[int, int, int] = (1, 1024, 1024),
        size_patch: int = 32,
        len_token: int = 768,
    ):
        """
        Constructor.

        Parameters
        ----------
        size_img: Tuple[int, int, int]
            Shape of input images (C, H, W).
        size_patch: int
            Height and width of patches.
        len_token: int
            Final token length.

        Returns
        -------
        None
        """
        super().__init__()

        # get user input
        C, H, W = size_img

        assert (
            H % size_patch == 0
        ), f"Height {H} is not divisible by size_patch {size_patch}."
        assert (
            W % size_patch == 0
        ), f"Width {W} is not divisible by size_patch {size_patch}."

        self.size_img = size_img
        self.size_patch = size_patch
        self.len_token = len_token
        self.num_tokens = (H / size_patch) * (W / size_patch)

        # define layers
        self.split = nn.Unfold(
            kernel_size=size_patch, stride=size_patch, padding=0
        )
        self.project = nn.Linear(self.size_patch**2 * C, self.len_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.split(x).transpose(2, 1)
        x = self.project(x)

        return x
