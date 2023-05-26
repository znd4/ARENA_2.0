# %%
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_backprop"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"
# %%


def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return np.divide(grad_out, x)


if MAIN:
    tests.test_log_back(log_back)
# %%


def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    """
    while original.ndim < broadcasted.ndim:
        broadcasted = broadcasted.sum(axis=0)

    for i, (broadcasted_len, original_len) in enumerate(
        zip(broadcasted.shape, original.shape)
    ):
        if original_len < broadcasted_len and original_len == 1:
            broadcasted = broadcasted.sum(axis=i, keepdims=True)

    return broadcasted


if MAIN:
    tests.test_unbroadcast(unbroadcast)


# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)

    return unbroadcast(grad_out * y, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)

    return unbroadcast(grad_out * x, y)


if MAIN:
    tests.test_multiply_back(multiply_back0, multiply_back1)
    tests.test_multiply_back_float(multiply_back0, multiply_back1)


# %%
def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    """
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)
    dg_dg = 1
    dg_df = log_back(dg_dg, g, f)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)
    dg_dc = log_back(dg_de, e, c)
    return dg_da, dg_db, dg_dc


if MAIN:
    tests.test_forward_and_back(forward_and_back)
# %%


@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."


# %%
class BackwardFuncLookup:
    def __init__(self) -> None:
        self.functions: Dict[Tuple[Callable, int], Callable] = {}

    def add_back_func(
        self, forward_fn: Callable, arg_position: int, back_fn: Callable
    ) -> None:
        self.functions[forward_fn, arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.functions[forward_fn, arg_position]


if MAIN:
    BACK_FUNCS = BackwardFuncLookup()
    BACK_FUNCS.add_back_func(np.log, 0, log_back)
    BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
    BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

    assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
    assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
    assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

    print("Tests passed - BackwardFuncLookup class is working as expected!")
# %%
Arr = np.ndarray


class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.
    """

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return multiply(other, self)

    def __truediv__(self, other) -> "Tensor":
        return true_divide(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return true_divide(other, self)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return matmul(other, self)

    def __eq__(self, other) -> "Tensor":
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError(
                "bool value of Tensor with more than one value is ambiguous"
            )
        return bool(self.item())


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


# %%
def log_forward(x: Tensor) -> Tensor:
    """Performs np.log on a Tensor object."""

    result = tensor(np.log(x.array))

    if grad_tracking_enabled and (x.recipe is not None or x.requires_grad):
        result.requires_grad = True
        result.recipe = Recipe(
            func=np.log, args=(x.array,), kwargs={}, parents=dict(enumerate([x]))
        )

    return result


if MAIN:
    log = log_forward
    tests.test_log(Tensor, log_forward)
    tests.test_log_no_grad(Tensor, log_forward)
    a = Tensor([1], requires_grad=True)
    grad_tracking_enabled = False
    b = log_forward(a)
    grad_tracking_enabled = True
    assert (
        not b.requires_grad
    ), "should not require grad if grad tracking globally disabled"
    assert (
        b.recipe is None
    ), "should not create recipe if grad tracking globally disabled"


# %%
def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    """Performs np.log on a Tensor object."""
    assert isinstance(a, Tensor) or isinstance(b, Tensor)

    parents = {}

    a_val = a
    a_requires_grad = False
    if isinstance(a, Tensor):
        a_val = a.array
        a_requires_grad = a.recipe is not None or a.requires_grad
        parents[0] = a

    b_val = b
    b_requires_grad = False
    if isinstance(b, Tensor):
        b_val = b.array
        b_requires_grad = b.recipe is not None or b.requires_grad
        parents[1] = b

    result = tensor(np.multiply(a_val, b_val))
    if grad_tracking_enabled and (a_requires_grad or b_requires_grad):
        result.requires_grad = True
        result.recipe = Recipe(
            func=np.multiply,
            args=(a_val, b_val),
            kwargs={},
            parents={k: v for k, v in {0: a, 1: b}.items() if isinstance(v, Tensor)},
        )

    return result


if MAIN:
    multiply = multiply_forward
    tests.test_multiply(Tensor, multiply_forward)
    tests.test_multiply_no_grad(Tensor, multiply_forward)
    tests.test_multiply_float(Tensor, multiply_forward)
    a = Tensor([2], requires_grad=True)
    b = Tensor([3], requires_grad=True)
    grad_tracking_enabled = False
    b = multiply_forward(a, b)
    grad_tracking_enabled = True
    assert (
        not b.requires_grad
    ), "should not require grad if grad tracking globally disabled"
    assert (
        b.recipe is None
    ), "should not create recipe if grad tracking globally disabled"


# %%
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        parents = {}
        vals = []
        requires_grad = []

        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                vals.append(arg.array)
                requires_grad.append(arg.recipe is not None or arg.requires_grad)
                parents[i] = arg
            else:
                vals.append(arg)
                requires_grad.append(False)

        result = tensor(numpy_func(*vals, **kwargs))
        if grad_tracking_enabled and any(requires_grad) and is_differentiable:
            result.requires_grad = True
            result.recipe = Recipe(
                func=numpy_func,
                args=vals,
                kwargs=kwargs,
                parents=parents,
            )

        return result

    return tensor_func


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)


if MAIN:
    log = wrap_forward_fn(np.log)
    multiply = wrap_forward_fn(np.multiply)
    eq = wrap_forward_fn(np.equal, is_differentiable=False)
    sum = wrap_forward_fn(_sum)

    tests.test_log(Tensor, log)
    tests.test_log_no_grad(Tensor, log)
    tests.test_multiply(Tensor, multiply)
    tests.test_multiply_no_grad(Tensor, multiply)
    tests.test_multiply_float(Tensor, multiply)
    tests.test_sum(Tensor)


# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    """
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    """
    # SOLUTION

    result: List[
        Node
    ] = []  # stores the list of nodes to be returned (in reverse topological order)
    perm: set[
        Node
    ] = set()  # same as `result`, but as a set (faster to check for membership)
    temp: set[
        Node
    ] = set()  # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        """
        Recursive function which visits all the children of the current node, and appends them all
        to `result` in the order they were found.
        """
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children(cur):
            visit(next)

        result.append(cur)
        temp.remove(cur)

    visit(node)
    return result


# %%
def sorted_computational_graph(tensor: Tensor) -> List[Tensor]:
    """
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph,
    in reverse topological order (i.e. `tensor` should be first).
    """
    nodes_to_tensors = {}

    def inner(tensor: Tensor) -> Node:
        if tensor.recipe:
            node = Node(*(inner(parent) for parent in tensor.recipe.parents.values()))
        else:
            node = Node()

        nodes_to_tensors[node] = tensor
        return node

    nodes_sorted = topological_sort(inner(tensor), get_children)
    return [nodes_to_tensors[node] for node in nodes_sorted][::-1]


if MAIN:
    a = Tensor([1], requires_grad=True)
    b = Tensor([2], requires_grad=True)
    c = Tensor([3], requires_grad=True)
    d = a * b
    e = c.log()
    f = d * e
    g = f.log()
    name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

    print([name_lookup[t] for t in sorted_computational_graph(g)])


# %%
def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    """
    # gradients = {end_node: end_grad.array if end_grad is not None else np.array((1,))}
    gradients = {end_node: np.ones_like(end_node.array) if end_grad is None else end_grad.array}

    graph = sorted_computational_graph(end_node)
    for tensor in graph:
        if tensor.is_leaf:
            if tensor.requires_grad:
                tensor.grad = Tensor(gradients[tensor])
            continue

        for arg_position, parent in tensor.recipe.parents.items():
            if parent not in gradients:
                gradients[parent] = np.zeros_like(parent.array)
            back_func = BACK_FUNCS.get_back_func(tensor.recipe.func, arg_position)
            gradient = back_func(
                gradients[tensor], tensor.array, *tensor.recipe.args, **tensor.recipe.kwargs
            )
            gradients[parent] = np.add(gradients[parent], gradient)


# def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
# 	'''Accumulates gradients in the grad field of each leaf node.

# 	tensor.backward() is equivalent to backprop(tensor).

# 	end_node:
# 		The rightmost node in the computation graph.
# 		If it contains more than one element, end_grad must be provided.
# 	end_grad:
# 		A tensor of the same shape as end_node.
# 		Set to 1 if not specified and end_node has only one element.
# 	'''

# 	# Get value of end_grad_arr
# 	end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

# 	# Create dict to store gradients
# 	grads: Dict[Tensor, Arr] = {end_node: end_grad_arr}

# 	# Iterate through the computational graph, using your sorting function
# 	for node in sorted_computational_graph(end_node):

# 		# Get the outgradient from the grads dict
# 		outgrad = grads.pop(node)
# 		# We only store the gradients if this node is a leaf & requires_grad is true
# 		if node.is_leaf and node.requires_grad:
# 			# Add the gradient to this node's grad (need to deal with special case grad=None)
# 			if node.grad is None:
# 				node.grad = Tensor(outgrad)
# 			else:
# 				node.grad.array += outgrad

# 		# If node has no parents, then the backtracking through the computational
# 		# graph ends here
# 		if node.recipe is None or node.recipe.parents is None:
# 			continue

# 		# If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
# 		for argnum, parent in node.recipe.parents.items():

# 			# Get the backward function corresponding to the function that created this node
# 			back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)

# 			# Use this backward function to calculate the gradient
# 			in_grad = back_fn(outgrad, node.array, *node.recipe.args, **node.recipe.kwargs)

# 			# Add the gradient to this node in the dictionary `grads`
# 			# Note that we only set node.grad (from the grads dict) in the code block above
# 			if parent not in grads:
# 				grads[parent] = in_grad
# 			else:
# 				grads[parent] += in_grad



if MAIN:
	tests.test_backprop(Tensor)
	tests.test_backprop_branching(Tensor)
	tests.test_backprop_requires_grad_false(Tensor)
	tests.test_backprop_float_arg(Tensor)


# %% 3️⃣ MORE FORWARD & BACKWARD FUNCTIONS

def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))



if MAIN:
    argmax = wrap_forward_fn(_argmax, is_differentiable=False)

    a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
    b = a.argmax()
    assert not b.requires_grad
    assert b.recipe is None
    assert b.item() == 3

# %%

def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    return np.full_like(x, -1) * grad_out



if MAIN:
    negative = wrap_forward_fn(np.negative)
    BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

    tests.test_negative_back(Tensor)

# %%

def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out



if MAIN:
    exp = wrap_forward_fn(np.exp)
    BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

    tests.test_exp_back(Tensor)

# %%

def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)



if MAIN:
    reshape = wrap_forward_fn(np.reshape)
    BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

    tests.test_reshape_back(Tensor)

# %%

def invert_transposition(axes: tuple) -> tuple:
    '''
    axes: tuple indicating a transition

    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x

    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 1, 2)
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    '''

    # Slick solution:
    return tuple(np.argsort(axes))

    # Slower solution, which makes it clearer what operation is happening:
    reversed_transposition_map = {num: idx for (idx, num) in enumerate(axes)}
    reversed_transposition = [reversed_transposition_map[idx] for idx in range(len(axes))]
    return tuple(reversed_transposition)

def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))



if MAIN:
    BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
    permute = wrap_forward_fn(np.transpose)

    tests.test_permute_back(Tensor)

# %%


if MAIN:
    x = np.array([1, 2, 3])

    np.broadcast_to(x, (3, 3))

# %%


if MAIN:
    x = np.array([[1], [2], [3]])

    np.broadcast_to(x, (3, 3)) # x has shape (3, 1); broadcasting is done along rows

# %%


if MAIN:
    x = t.tensor([[1], [2], [3]])

    x.expand(-1, 3)

# %%

def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)

def _expand(x: Arr, new_shape) -> Arr:
    '''
    Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''

    n_added = len(new_shape) - x.ndim
    shape_non_negative = tuple([x.shape[i - n_added] if s == -1 else s for i, s in enumerate(new_shape)])
    return np.broadcast_to(x, shape_non_negative)



if MAIN:
    expand = wrap_forward_fn(_expand)
    BACK_FUNCS.add_back_func(_expand, 0, expand_back)

    tests.test_expand(Tensor)
    tests.test_expand_negative_length(Tensor)

# %%

def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    '''Basic idea: repeat grad_out over the dims along which x was summed'''

    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = np.array(grad_out)

    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))

    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)

    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    return np.sum(x, axis=dim, keepdims=keepdim)


if MAIN:
    sum = wrap_forward_fn(_sum)
    BACK_FUNCS.add_back_func(_sum, 0, sum_back)

    tests.test_sum_keepdim_false(Tensor)
    tests.test_sum_keepdim_true(Tensor)
    tests.test_sum_dim_none(Tensor)

# %%

if MAIN:
    Index = Union[int, Tuple[int, ...], Tuple[Arr], Tuple[Tensor]]

def coerce_index(index: Index) -> Union[int, Tuple[int, ...], Tuple[Arr]]:
    '''
    If index is of type signature `Tuple[Tensor]`, converts it to `Tuple[Arr]`.
    '''
    if isinstance(index, tuple) and set(map(type, index)) == {Tensor}:
        return tuple([i.array for i in index])
    else:
        return index

def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]

def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''
    Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out



if MAIN:
    getitem = wrap_forward_fn(_getitem)
    BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

    tests.test_coerce_index(coerce_index, Tensor)
    tests.test_getitem_int(Tensor)
    tests.test_getitem_tuple(Tensor)
    tests.test_getitem_integer_array(Tensor)
    tests.test_getitem_integer_tensor(Tensor)

# %%


if MAIN:
    add = wrap_forward_fn(np.add)
    subtract = wrap_forward_fn(np.subtract)
    true_divide = wrap_forward_fn(np.true_divide)

    # FLAT SOLUTION
    # Your code here - add to the BACK_FUNCS object
    BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
    BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y))
    BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
    BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y))
    BACK_FUNCS.add_back_func(np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out/y, x))
    BACK_FUNCS.add_back_func(np.true_divide, 1, lambda grad_out, out, x, y: unbroadcast(grad_out*(-x/y**2), y))
    # FLAT SOLUTION END

# %%


if MAIN:
    tests.test_add_broadcasted(Tensor)
    tests.test_subtract_broadcasted(Tensor)
    tests.test_truedivide_broadcasted(Tensor)

# %%

# def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
#     '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
#     np.add(x.array, other.array * alpha, out=x.array)
#     return x


# def safe_example():
#     '''This example should work properly.'''
#     a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
#     b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
#     a.add_(b)
#     c = a * b
#     c.sum().backward()
#     assert a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0])
#     assert b.grad is not None and np.allclose(b.grad.array, [2.0, 4.0, 6.0, 8.0])


# def unsafe_example():
#     '''This example is expected to compute the wrong gradients.'''
#     a = Tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
#     b = Tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
#     c = a * b
#     a.add_(b)
#     c.sum().backward()
#     if a.grad is not None and np.allclose(a.grad.array, [2.0, 3.0, 4.0, 5.0]):
#         print("Grad wrt a is OK!")
#     else:
#         print("Grad wrt a is WRONG!")
#     if b.grad is not None and np.allclose(b.grad.array, [0.0, 1.0, 2.0, 3.0]):
#         print("Grad wrt b is OK!")
#     else:
#         print("Grad wrt b is WRONG!")



# if MAIN:
#     safe_example()
#     unsafe_example()

# %%


if MAIN:
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    (a * 2).sum().backward()
    b = Tensor([0, 1, 2, 3], requires_grad=True)
    (2 * b).sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, b.grad.array)

# %%

def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt x.'''
    bool_sum = ((x > y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, x)

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt y.'''
    bool_sum = ((x < y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, y)



if MAIN:
    maximum = wrap_forward_fn(np.maximum)

    BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
    BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)

    tests.test_maximum(Tensor)
    tests.test_maximum_broadcasted(Tensor)

# %%

def relu(x: Tensor) -> Tensor:
    '''Like torch.nn.function.relu(x, inplace=False).'''
    return maximum(x, 0.0)



if MAIN:
    tests.test_relu(Tensor)
# %%
