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

arr = np.load(section_dir / "numbers.npy")

arr1 = lambda: einops.rearrange(arr, "b c h w -> c h (b w)")
arr2 = lambda: einops.repeat(arr[0], "c h w -> c (2 h) w")
arr3 = lambda: einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
arr4 = lambda: einops.repeat(arr[0], "c h w -> c (h 2) w")
arr5 = lambda: einops.rearrange(arr[0], "c h w -> h (c w)")
arr6 = lambda: einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1 = 2)
arr7 = lambda: einops.reduce(arr, "b c h w ->  h (b w)", "max")
arr8 = lambda: einops.reduce(arr, "b c h w -> h w", "min")
arr9 = lambda: einops.rearrange(arr[1], "c h w -> c w h")
arr10 = lambda: einops.reduce(arr, "(b1 b2) c (h 2) (w 2) -> c (b1 h) (b2 w)", "max", b1 = 2)

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return np.einsum("ii->", mat)

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return np.einsum("mn,n->m", mat, vec)

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return np.einsum("ij,jk->ik", mat1, mat2)

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return np.einsum("i,i->", vec1, vec2)

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return np.einsum("i,j->ij", vec1, vec2)


# if MAIN:
#     tests.test_einsum_trace(einsum_trace)
#     tests.test_einsum_mv(einsum_mv)
#     tests.test_einsum_mm(einsum_mm)
#     tests.test_einsum_inner(einsum_inner)
#     tests.test_einsum_outer(einsum_outer)


