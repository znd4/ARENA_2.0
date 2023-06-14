import os
import more_itertools as mi
from operator import mul
from functools import reduce
from dataclasses import dataclass
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

arr = np.load(section_dir / "numbers.npy")


def arr1():
    return einops.rearrange(arr, "b c h w -> c h (b w)")


def arr2():
    return einops.repeat(arr[0], "c h w -> c (2 h) w")


def arr3():
    return einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")


def arr4():
    return einops.repeat(arr[0], "c h w -> c (h 2) w")


def arr5():
    return einops.rearrange(arr[0], "c h w -> h (c w)")


def arr6():
    return einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)


def arr7():
    return einops.reduce(arr, "b c h w ->  h (b w)", "max")


def arr8():
    return einops.reduce(arr, "b c h w -> h w", "min")


def arr9():
    return einops.rearrange(arr[1], "c h w -> c w h")


def arr10():
    return einops.reduce(arr, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", "max", b1=2)


def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return np.einsum("ii->", mat)


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return np.einsum("mn,n->m", mat, vec)


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return np.einsum("ij,jk->ik", mat1, mat2)


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return np.einsum("i,i->", vec1, vec2)


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return np.einsum("i,j->ij", vec1, vec2)


# if MAIN:
#     tests.test_einsum_trace(einsum_trace)
#     tests.test_einsum_mv(einsum_mv)
#     tests.test_einsum_mm(einsum_mm)
#     tests.test_einsum_inner(einsum_inner)
#     tests.test_einsum_outer(einsum_outer)


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


def conv1d(
    x: Float[Tensor, "b ic w"],
    weights: Float[Tensor, "oc ic kw"],
    stride: int = 1,
    padding: int = 0,
) -> Float[Tensor, "b oc ow"]:
    """
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    """
    x = pad1d(x, padding, padding, 0.0)

    b, ic, w = x.shape
    oc, ic, kw = weights.shape

    ow = (w - kw) // stride + 1

    s_b, s_ic, s_w = x.stride()
    s_w *= stride
    size, stride = zip([b, s_b], [ic, s_ic], [ow, s_w], [kw, s_w // stride])
    x_strided = x.as_strided(size=size, stride=stride)
    return einops.einsum(
        x_strided,
        weights,
        "b ic ow kw, oc ic kw -> b oc ow",
    )


IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]


def force_pair(v: IntOrPair) -> Pair:
    """Convert v to a pair of int, if it isn't already."""
    if isinstance(v, int):
        return (v, v)

    if not isinstance(v, tuple):
        raise ValueError(v)

    if len(v) != 2:
        raise ValueError(v)

    return (int(v[0]), int(v[1]))


def conv2d(
    x: Float[Tensor, "b ic h w"],
    weights: Float[Tensor, "oc ic kh kw"],
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b oc oh ow"]:
    """
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    """
    h_pad, w_pad = force_pair(padding)
    x = pad2d(x, w_pad, w_pad, h_pad, h_pad, 0.0)

    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape

    h_stride, w_stride = force_pair(stride)

    oh = (h - kh) // h_stride + 1
    ow = (w - kw) // w_stride + 1
    s_b, s_ic, s_h, s_w = x.stride()

    x_size, x_stride = zip(
        (b, s_b),
        (ic, s_ic),
        (oh, s_h * h_stride),
        (ow, s_w * w_stride),
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


def maxpool2d(
    x: Float[Tensor, "b ic h w"],
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
) -> Float[Tensor, "b ic oh ow"]:
    """
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    """
    stride = stride or kernel_size
    h_pad, w_pad = force_pair(padding)
    x = pad2d(x, w_pad, w_pad, h_pad, h_pad, -np.inf)
    b, ic, h, w = x.shape

    h_stride, w_stride = force_pair(stride)
    kh, kw = force_pair(kernel_size)

    oh = (h - kh) // h_stride + 1
    ow = (w - kw) // w_stride + 1

    s_b, s_ic, s_h, s_w = x.stride()
    size, stride = zip(
        (kh, s_h),
        (kw, s_w),
        (b, s_b),
        (ic, s_ic),
        (oh, s_h * h_stride),
        (ow, s_w * w_stride),
    )
    return x.as_strided(
        size=size,
        stride=stride,
    ).amax(dim=(0, 1))


@dataclass
class DataclassModule(nn.Module):
    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        nn.Module.__init__(inst)
        return inst


@dataclass(eq=False)
class MaxPool2d(DataclassModule):
    kernel_size: IntOrPair
    stride: Optional[IntOrPair] = None
    padding: IntOrPair = 1

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Call the functional version of maxpool2d."""
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)


class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.clamp(min=0.0)


@dataclass(eq=False)
class Flatten(DataclassModule):
    start_dim: int = 1
    end_dim: int = -1

    def forward(self, input: t.Tensor) -> t.Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape
        flatten_dims = (*shape[self.start_dim : self.end_dim], shape[self.end_dim])
        if not flatten_dims:
            return input

        target_shape = (
            *shape[: self.start_dim],
            reduce(mul, flatten_dims),
            *shape[self.end_dim :][1:],
        )
        return t.reshape(input, target_shape)


if MAIN:
    tests.test_flatten(Flatten)
