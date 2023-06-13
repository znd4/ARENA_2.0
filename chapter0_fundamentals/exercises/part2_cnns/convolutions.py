import torch as t
from collections import namedtuple

import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"


def conv1d_minimal_simple(
    x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]
) -> Float[Tensor, "ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    """
    n_x = x.shape[0]
    n_w = weights.shape[0]
    return (
        x.as_strided(
            size=(n_x - n_w + 1, n_w),
            stride=(1, 1),
        )
        * weights[None, :]
    ).sum(dim=1)


def conv1d_minimal(
    x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    b, ic, w = x.shape
    oc, ic, kw = weights.shape
    ow = w - kw + 1
    return einops.einsum(
        x.as_strided(
            size=(
                b,
                oc,
                ic,
                ow,
                kw,
            ),
            stride=(
                x.stride()[0],
                0,
                x.stride()[1],
                x.stride()[2],
                x.stride()[2],
            ),
        )
        * weights[None, :, :, None, :],
        "b oc ic ow kw -> b oc ow",
    )


def conv2d_minimal(
    x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape
    oh = h - kh + 1
    ow = w - kw + 1
    s_b, s_ic, s_h, s_w = x.stride()
    x_size, x_stride = zip(
        (b, s_b),
        (ic, s_ic),
        (oh, s_h),
        (ow, s_w),
        (kh, s_h),
        (kw, s_w),
    )
    x_strided = x.as_strided(
        size=x_size,
        stride=x_stride,
    )
    return einops.einsum(
        x_strided,
        weights,
        "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow",
    )


def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    """
    b, ic, w = x.shape
    result = x.new_full(
        size=(b, ic, w + left + right),
        fill_value=pad_value,
    )
    result[..., left : left + w] = x
    return result


def pad2d(
    x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float
) -> t.Tensor:
    """Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    """
    b, ic, h, w = x.shape
    result = x.new_full(
        size=(b, ic, h + top + bottom, w + left + right),
        fill_value=pad_value,
    )
    result[..., top : top + h, left : left + w] = x
    return result


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)
