import tempfile
from itermplot import imgcat
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import io
import typeguard
import torch as t
import einops

# ruff: noqa: F722

import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(
    ray: Float[t.Tensor, "n_points=2 n_dim=3"],
    segment: Float[t.Tensor, "n_points n_dim"],
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """

    D = ray[1, :2]
    Origin = ray[0, :2]
    L1 = segment[0, :2]
    L2 = segment[1, :2]

    try:
        A = t.stack((D, L1 - L2), dim=1)
        B = (L1 - Origin)[:, None]

        u, v = t.linalg.solve(A, B)
    except Exception:
        return False

    return u.item() >= 0 and (0 <= v.item() <= 1)


@jaxtyped
# @typeguard.typechecked
def intersect_rays_1d(
    rays: Float[t.Tensor, "nrays 2 3"], segments: Float[t.Tensor, "nsegments 2 3"]
) -> Bool[t.Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """

    rays = rays[..., :2]
    segments = segments[..., :2]

    Os = rays[..., 0, :]
    Ds = rays[..., 1, :]

    L1s = segments[..., 0, :]
    L2s = segments[..., 1, :]

    nrays = rays.shape[0]
    nsegments = segments.shape[0]

    A = t.stack(
        (
            einops.repeat(
                Ds, "nrays dims -> nrays nsegments dims", nsegments=nsegments
            ),
            einops.repeat(
                L1s - L2s, "nsegments dims -> nrays nsegments dims", nrays=nrays
            ),
        ),
        dim=3,
    )
    B = (L1s[None, ...] - Os[:, None, ...])[..., None]

    singular_mask = A.det().abs() < 1e-6

    A[singular_mask, ...] = t.eye(2)

    x = t.linalg.solve(A, B).squeeze()

    u = x[..., 0]
    v = x[..., 1]
    return ((u >= 0) & (0 <= v) & (v <= 1) & (~(singular_mask))).any(dim=1)


@jaxtyped
# @typeguard.typechecked
def make_rays_2d(
    num_pixels_y: int,
    num_pixels_z: int,
    y_limit: float,
    z_limit: float,
) -> Float[t.Tensor, "num_pixels_y * num_pixels_z 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    result = t.zeros((num_pixels_y * num_pixels_z, 2, 3))
    y_vals = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_vals = t.linspace(-z_limit, z_limit, num_pixels_z)

    result[:, 1] = t.cartesian_prod(t.ones((1,)), y_vals, z_vals)

    return result


Point = Float[t.Tensor, "points=3"]


@jaxtyped
# @typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    print(f"{A=}")
    a = t.stack((-D, (B - A), (C - A)), dim=1)
    b = (O - A)[:, None]
    if t.linalg.det(a) == 0:
        return False

    try:
        s, u, v = t.linalg.solve(a, b).squeeze()
    except Exception as e:
        print(e)
        return False

    if (s < 0) or (u < 0) or (v < 0) or (u + v > 1):
        return False

    return True


@jaxtyped
def raytrace_triangle(
    rays: Float[t.Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[t.Tensor, "trianglePoints=3 dims=3"],
) -> Bool[t.Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    O = rays[:, 0]
    D = rays[:, 1]
    A = triangle[0][None, :]
    B = triangle[1][None, :]
    C = triangle[2][None, :]
    nrays = rays.shape[0]

    As: Float[t.Tensor, "nrays 3 3"] = t.stack(
        (
            -D,
            einops.repeat(
                (B - A),
                f"1 dims -> {nrays} dims",
            ),
            einops.repeat(
                (C - A),
                f"1 dims -> {nrays} dims",
            ),
        ),
        dim=2,
    )
    Bs = O - A
    singular_mask = As.det().abs() < 1e-6
    As[singular_mask, ...] = t.eye(3)
    x = t.linalg.solve(As, Bs).squeeze()
    s = x[..., 0]
    u = x[..., 1]
    v = x[..., 2]

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (~singular_mask)




def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    rays = einops.repeat(
        rays,
        "nrays rayPoints dims -> nrays ntriangles rayPoints dims",
        ntriangles=ntriangles,
    )
    triangles = einops.repeat(
        triangles,
        "ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims",
        nrays=nrays,
    )
    O = rays[..., 0, :]
    D = rays[..., 1, :]

    A = triangles[..., 0, :]
    B = triangles[..., 1, :]
    C = triangles[..., 2, :]

    As: t.Tensor = t.stack(
        (
            -D,
            (B - A),
            (C - A),
        ),
        dim=3,
    )
    Bs = O - A
    singular_mask = As.det().abs() < 1e-6
    As[singular_mask] = t.eye(3)
    x = t.linalg.solve(As, Bs).squeeze()
    s = x[..., 0]
    u = x[..., 1]
    v = x[..., 2]

    intersections = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & (~singular_mask)
    dists = t.where(intersections, s, t.tensor(float("inf")))
    return dists.min(dim=1).values


def main():
    """
    use with
    >>> %load_ext imgcat
    >>> from part1_ray_tracing import answers
    >>> %imgcat answers.main()
    """
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    print(f"{dists=}")
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(
        img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000
    )
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]):
        fig.layout.annotations[i]["text"] = text
    fig.show()


if __name__ == "__main__":
    main()
