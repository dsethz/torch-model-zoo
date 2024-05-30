#######################################################################################################################
# This script contains utility functions.                                                                             #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.11.7                                                                                        #
# PyTorch Version:      2.3.0                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################
from typing import List

import numpy as np
import torch


def _get_prediction_token(tokens: torch.Tensor) -> torch.Tensor:
    """
    This function concatenates the prediction token to tokens.

    Parameters
    -----------
    tokens: torch.Tensor
        Tokenized image.

    Returns
    -------
    torch.Tensor
        Tokenized image with prediction token.

    """
    B, N, C = tokens.shape
    prediction_token = torch.zeros(B, 1, C, device=tokens.device)
    tokens = torch.cat([prediction_token, tokens], dim=1)

    return tokens


def _get_sinusoid_encoding(num_tokens: int, len_token: int) -> torch.Tensor:
    """
    This function generates sinusoid positional encoding.

    Parameters
    ----------
    num_tokens: int
        Number of tokens.
    len_token: int
        Length of token.

    Returns
    -------
    torch.Tensor
        Sinusoid positional encoding.
    """

    def _get_positional_angle_vec(i: int) -> List[float]:
        return [
            i / np.power(10000, 2 * (j // 2) / len_token)
            for j in range(len_token)
        ]

    sinusoid_table = np.array(
        [_get_positional_angle_vec(i) for i in range(num_tokens)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
