#######################################################################################################################
# This script contains a Visual Transformer implemented in torch and lightning                                        #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.11.7                                                                                        #
# PyTorch Version:      2.3.0                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

from typing import Optional

from torch import nn


class HeadAttention(nn.Module):
    """
    This class defines a single attention head.
    """

    def __init__(
        self,
        n_embd: int,
        size_head: int,
        bias: bool = False,
        scale: Optional[float] = None,
        attn_drop: float = 0.2,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_embd: int
            Input channels of single head.
        size_head: int
            Output channels of single head.
        bias: bool
            True if bias should be used for linear layers of Q, K, and V.
        scale: Optional[float]
            Scaling factor of self-attention. If None (default) use 1/sqrt(size_head).
        attn_drop: float
            Dropout rate of attention layer.

        Returns
        -------
        None
        """
        super().__init__()

        assert isinstance(
            n_embd, int
        ), f"n_embd must be int, but is {type(n_embd)}."
        assert isinstance(
            size_head, int
        ), f"size_head must be int, but is {type(size_head)}."
        assert isinstance(
            bias, bool
        ), f"bias must be bool, but is {type(bias)}."
        assert scale is None or isinstance(
            scale, float
        ), f"scale must be float, but is {type(scale)}."
        assert isinstance(
            attn_drop, float
        ), f"attn_drop must be float, but is {type(attn_drop)}."

        self.n_embd = n_embd
        self.size_head = size_head
        self.scale = scale if scale is not None else size_head**-0.5

        self.key = nn.Linear(n_embd, size_head, bias=bias)
        self.query = nn.Linear(n_embd, size_head, bias=bias)
        self.value = nn.Linear(n_embd, size_head, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        # calculate query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        return attn @ v
