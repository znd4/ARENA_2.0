from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard
import torch as t
import einops

# ruff: noqa: F722


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
                Ds,
                "nrays dims -> nrays nsegments dims",
                nsegments=nsegments
            ),
            einops.repeat(
                L1s - L2s,
                "nsegments dims -> nrays nsegments dims",
                nrays=nrays
            ),
        ),
        dim=3,
    )
    B = (L1s[None, ...] - Os[:, None, ...])[..., None]

    singular_mask = (A.det().abs() < 1e-6)

    A[singular_mask, ...] = t.eye(2)


    x = t.linalg.solve(A, B).squeeze()

    u = x[..., 0]
    v = x[..., 1]
    return ((u >= 0) & (0 <= v) & (v <= 1) & (~(singular_mask))).any(dim=1)

@jaxtyped
@typeguard.typechecked
def make_rays_2d(
    num_pixels_y: int,
    num_pixels_z: int,
    y_limit: float,
    z_limit: float,
) -> Float[t.Tensor, "num_pixels_y * num_pixels_z 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    result = t.zeros((num_pixels_y * num_pixels_z, 2, 3))
    y_vals = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_vals = t.linspace(-z_limit, z_limit, num_pixels_z)

    @jaxtyped
    def permute_vectorized() -> Float[t.Tensor, "num_pixels_y num_pixels_z 3"]:
        return t.stack(
            (
                t.ones((1,))[:, None],
                y_vals[:, None],
                z_vals[None, :],
            ),
            dim=2,
        )
    result[:, 1, :] = permute_vectorized()
    return result

