import numpy as np
import pytest
import torch

from torch_model_zoo.models.vit import (
    FeedForward,
    HeadAttention,
    MultiHeadAttention,
)


def test_head_init():
    with pytest.raises(AssertionError):
        _ = HeadAttention(
            n_embd=0.5, size_head=4, bias=False, scale=None, drop=0.2
        )

    with pytest.raises(AssertionError):
        _ = HeadAttention(
            n_embd=5, size_head=0.4, bias=False, scale=None, drop=0.2
        )

    with pytest.raises(AssertionError):
        _ = HeadAttention(n_embd=5, size_head=4, bias=2, scale=None, drop=0.2)

    with pytest.raises(AssertionError):
        _ = HeadAttention(
            n_embd=5, size_head=4, bias=False, scale="abc", drop=0.2
        )

    with pytest.raises(AssertionError):
        _ = HeadAttention(
            n_embd=5, size_head=4, bias=False, scale=None, drop=False
        )


def test_head():
    b, n, c = 3, 10, 8  # tokens size
    size_head = 4

    x = np.random.rand(b, n, c)
    x_t = torch.from_numpy(x).float()
    head = HeadAttention(n_embd=c, size_head=size_head)
    out = head(x_t)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (b, n, size_head)


def test_mhead_init():
    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=0.5,
            n_heads=2,
            size_head=4,
            bias=False,
            scale=None,
            drop=0.2,
        )

    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=5,
            n_heads=0.2,
            size_head=4,
            bias=False,
            scale=None,
            drop=0.2,
        )

    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=5,
            n_heads=2,
            size_head=0.4,
            bias=False,
            scale=None,
            drop=0.2,
        )

    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=5, n_heads=2, size_head=4, bias=2, scale=None, drop=0.2
        )

    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=5, n_heads=2, size_head=4, bias=False, scale="abc", drop=0.2
        )

    with pytest.raises(AssertionError):
        _ = MultiHeadAttention(
            n_embd=5,
            n_heads=2,
            size_head=4,
            bias=False,
            scale=None,
            drop=False,
        )


def test_mhead():
    b, n, c = 3, 10, 8  # tokens size
    size_head = 4
    n_heads = 2

    x = np.random.rand(b, n, c)
    x_t = torch.from_numpy(x).float()
    heads = MultiHeadAttention(n_embd=c, n_heads=n_heads, size_head=size_head)
    out = heads(x_t)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (b, n, c)


def test_ff_init():
    with pytest.raises(AssertionError):
        _ = FeedForward(n_embd=0.5, bias=False, drop=None)

    with pytest.raises(AssertionError):
        _ = FeedForward(n_embd=5, bias=2, drop=None)

    with pytest.raises(AssertionError):
        _ = FeedForward(n_embd=5, bias=False, drop=True)


def test_ff():
    b, n, c = 3, 10, 8  # input shape

    x = np.random.rand(b, n, c)
    x_t = torch.from_numpy(x).float()
    ff = FeedForward(n_embd=c, bias=False, drop=0.2)
    out = ff(x_t)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (b, n, c)
