import numpy as np
import pytest
import torch

from torch_model_zoo.models.vit import HeadAttention


def test_head_init():
    with pytest.raises(AssertionError):
        _ = HeadAttention(n_embd=0.5, size_head=4, bias=False, scale=None)

    with pytest.raises(AssertionError):
        _ = HeadAttention(n_embd=5, size_head=0.4, bias=False, scale=None)

    with pytest.raises(AssertionError):
        _ = HeadAttention(n_embd=5, size_head=4, bias=2, scale=None)

    with pytest.raises(AssertionError):
        _ = HeadAttention(n_embd=5, size_head=4, bias=False, scale="abc")


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
