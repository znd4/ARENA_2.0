# %%

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
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

# %%

arr = np.load(section_dir / "numbers.npy")

# %%

if MAIN:
    display_array_as_img(arr[1])

# %%

arr1 = arr[0]

if MAIN:
    display_array_as_img(arr1)

# %%

arr2 = einops.rearrange(arr, 'nums channels y x -> channels y (nums x)')

if MAIN:
    display_array_as_img(arr2)
# %%

arr3 = einops.repeat(arr[0], 'channels y x -> channels (2 y) x')

if MAIN:
    display_array_as_img(arr3)
# %%

arr4 = einops.repeat(arr[:2], 'nums channels y x -> channels (nums y) (2 x)')

if MAIN:
    display_array_as_img(arr4)

# %%

arr5 = einops.repeat(arr[0], 'channels y x -> channels (y 2) x')

if MAIN:
    display_array_as_img(arr5)

# %%

arr6 = einops.rearrange(arr[0], 'channels y x -> y (channels x)')


if MAIN:
    display_array_as_img(arr6)

# %%

arr7 = einops.rearrange(arr, '(a b) channels y x -> channels (a y) (b x)', a=2, b=3)


if MAIN:
    display_array_as_img(arr7)

# %%

arr8 = einops.reduce(arr, 'nums channels y x -> y (nums x)', 'max')
if MAIN:
    display_array_as_img(arr8)

# %%


arr9 = einops.reduce(arr.astype(float), 'nums channels y x -> y (nums x)', 'min')
if MAIN:
    display_array_as_img(arr9)


# %%

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, "i i -> ")

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, "i j, j -> i")

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, "i j, j k -> i k")

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, "i, i ->")

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, "i, j -> i j")


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)