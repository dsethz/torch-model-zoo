import numpy as np
import pytest
import torch

from torch_model_zoo.utils.tokenization_img import ImgTokenizer


def test_tokenizer_img_init():
    with pytest.raises(AssertionError):
        _ = ImgTokenizer(size_img=(3, 32, 32), size_patch=7, len_token=8)


def test_tokenizer_img():
    c, w, h = 3, 32, 32  # image size
    p = 8  # patch size
    tl = 8  # token length
    num_tokens = (w // p) * (h // p)

    img = np.random.rand(c, h, w)
    img_t = torch.from_numpy(img).float().unsqueeze(0)
    tokenizer = ImgTokenizer(size_img=img.shape, size_patch=p, len_token=tl)
    tokens = tokenizer(img_t)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (1, num_tokens, tl)
    assert tokens.dtype == torch.float32
