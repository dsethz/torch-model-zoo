#######################################################################################################################
# This script contains a Visual Transformer implemented in torch and lightning                                        #
# Author:               Daniel Schirmacher                                                                            #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.11.7                                                                                        #
# PyTorch Version:      2.3.0                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################
from typing import Optional

import torch
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
        drop: float = 0.2,
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
        drop: float
            Dropout rate.

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
            drop, float
        ), f"drop must be float, but is {type(drop)}."

        self.n_embd = n_embd
        self.size_head = size_head
        self.scale = scale if scale is not None else size_head**-0.5

        self.key = nn.Linear(n_embd, size_head, bias=bias)
        self.query = nn.Linear(n_embd, size_head, bias=bias)
        self.value = nn.Linear(n_embd, size_head, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape

        # calculate query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        return attn @ v


class MultiHeadAttention(nn.Module):
    """
    This class defines an MSA block.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        size_head: int,
        bias: bool = False,
        scale: Optional[float] = None,
        drop: float = 0.2,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_embd: int
            Input channels of single head.
        n_heads: int
            Number of attention heads.
        size_head: int
            Output channels of single head.
        bias: bool
            True if bias should be used for linear layers of Q, K, and V.
        scale: Optional[float]
            Scaling factor of self-attention. If None (default) use 1/sqrt(size_head).
        drop: float
            Dropout rate.

        Returns
        -------
        None
        """
        super().__init__()

        assert isinstance(
            n_embd, int
        ), f"n_embd must be int, but is {type(n_embd)}."
        assert isinstance(
            n_heads, int
        ), f"n_heads must be int, but is {type(n_heads)}."
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
            drop, float
        ), f"drop must be float, but is {type(drop)}."

        self.n_heads = n_heads
        self.size_head = size_head

        self.heads = nn.ModuleList(
            [
                HeadAttention(
                    n_embd=n_embd,
                    size_head=size_head,
                    bias=bias,
                    scale=scale,
                    drop=drop,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(size_head * n_heads, n_embd, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drop(out)

        return out


class FeedForward(nn.Module):
    """
    This class defines an feed forward block.
    """

    def __init__(
        self,
        n_embd: int,
        bias: bool = False,
        drop: float = 0.2,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_embd: int
            Input channels of single head.
        bias: bool
            True if bias should be used for linear layers of Q, K, and V.
        drop: float
            Dropout rate.

        Returns
        -------
        None
        """
        super().__init__()

        assert isinstance(
            n_embd, int
        ), f"n_embd must be int, but is {type(n_embd)}."
        assert isinstance(
            bias, bool
        ), f"bias must be bool, but is {type(bias)}."
        assert isinstance(
            drop, float
        ), f"drop must be float, but is {type(drop)}."

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4, bias=bias),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd, bias=bias),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    This class defines a full Encoder block including MSA and feed forward.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        bias: bool = False,
        scale: Optional[float] = None,
        drop: float = 0.2,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_embd: int
            Input channels of single head.
        n_heads: int
            Number of attention heads.
        bias: bool
            True if bias should be used for linear layers of Q, K, and V.
        scale: Optional[float]
            Scaling factor of self-attention. If None (default) use 1/sqrt(size_head).
        drop: float
            Dropout rate.

        Returns
        -------
        None
        """
        super().__init__()

        assert isinstance(
            n_embd, int
        ), f"n_embd must be int, but is {type(n_embd)}."
        assert isinstance(
            n_heads, int
        ), f"n_heads must be int, but is {type(n_heads)}."
        assert isinstance(
            bias, bool
        ), f"bias must be bool, but is {type(bias)}."
        assert scale is None or isinstance(
            scale, float
        ), f"scale must be float, but is {type(scale)}."
        assert isinstance(
            drop, float
        ), f"drop must be float, but is {type(drop)}."
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads."

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.size_head = n_embd // n_heads
        self.bias = bias
        self.scale = scale
        self.drop = drop

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.msa = MultiHeadAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            size_head=self.size_head,
            bias=bias,
            scale=scale,
            drop=drop,
        )

        self.ff = FeedForward(
            n_embd=n_embd,
            bias=bias,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
