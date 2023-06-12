
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
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"
if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )

    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")

def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    i = mat.shape[0]
    return mat.as_strided(
        size=(i,),
        stride=(i+1,),
    ).sum()


if MAIN:
    tests.test_trace(as_strided_trace)

def as_strided_mv(
    mat: Float[Tensor, "i j"],
    vec: Float[Tensor, "j"]
) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''

    i, j = mat.shape
    result = (
        mat * vec.as_strided(
            size=(i, j),
            stride=(0, vec.stride()[0]),
        )
    ).sum(axis=1)
    return result


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)


def as_strided_mm(
    matA: Float[Tensor, "i j"],
    matB: Float[Tensor, "j k"]
) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    i, j = matA.shape
    j, k = matB.shape
    return (
        matA.as_strided(size=(i, j, k), stride=(*matA.stride(), 0))
        *matB.as_strided(size=(i, j, k), stride=(0, *matB.stride()))
    ).sum(dim=1)



if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)
