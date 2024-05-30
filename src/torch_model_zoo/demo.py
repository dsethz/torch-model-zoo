import torch
from skimage import io

from torch_model_zoo.utils.functional import (
    _get_prediction_token,
    _get_sinusoid_encoding,
)
from torch_model_zoo.utils.tokenization_img import ImgTokenizer

# global parameters
size_patch = 32
len_token = 768


# Load the image
img = io.imread("/Users/schidani/Desktop/test.png")
img_t = torch.from_numpy(img).float()
img_t = img_t[None, None, :]

# convert to tokens
tokenizer = ImgTokenizer(
    size_img=(img_t.shape[1], img_t.shape[2], img_t.shape[3]),
    size_patch=size_patch,
    len_token=len_token,
)

tokens = tokenizer(img_t)

# add prediction token
tokens = _get_prediction_token(tokens)

# add positional encoding
positional_encoding = _get_sinusoid_encoding(
    num_tokens=tokens.shape[1], len_token=len_token
)
tokens = tokens + positional_encoding
