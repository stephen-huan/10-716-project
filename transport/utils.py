from bisect import bisect_left
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from gpjax.kernels import DenseKernelComputation
from jax import Array, jit, lax

KeyArray = Array
Kernel = tuple[nnx.GraphDef, nnx.State]

dense = DenseKernelComputation()
gram, cross_covariance, diagonal = (
    dense.gram,
    dense.cross_covariance,
    dense.diagonal,
)


def index_dtype(
    x: int | Array, unsigned: bool = True, axis: int | None = None
):
    """Return the smallest integer datatype that can represent indices in x."""
    max_value = lambda dtype: jnp.iinfo(dtype).max  # noqa: E731
    dtypes = sorted(
        (
            [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]
            if unsigned
            else [jnp.int8, jnp.int16, jnp.int32, jnp.int64]
        ),
        key=max_value,
    )
    sizes = list(map(max_value, dtypes))
    n = (
        x
        if isinstance(x, int)
        else (x.size if axis is None else x.shape[axis])
    )
    return dtypes[bisect_left(sizes, n - 1)]


@jit
def argmax_masked(x: Array, mask: Array) -> Array:
    """Argmax of x restricted to the indices in mask, including nans."""
    return jnp.nanargmax(
        jnp.where(jnp.isnan(x), -jnp.inf, x) + jnp.where(mask, 0.0, jnp.nan)
    )


@jit
def argmin_masked(x: Array, mask: Array) -> Array:
    """Argmin of x restricted to the indices in mask, including nans."""
    return jnp.nanargmin(
        jnp.where(jnp.isnan(x), jnp.inf, x) + jnp.where(mask, 0.0, jnp.nan)
    )


@jit
def identity_like(m: Array) -> Array:
    """Assuming m is square, make an identity matrix like m."""
    return jnp.identity(m.shape[0], dtype=m.dtype)


@jit
def logdet(m: Array) -> Array:
    """Computes the logarithm of the determinant of m."""
    return jnp.linalg.slogdet(m)[1]


@jit
def solve(A: Array, b: Array) -> Array:
    """Solve the system Ax = b for symmetric positive definite A."""
    return jax.scipy.linalg.solve(A, b, assume_a="pos")


@jit
def inv(m: Array) -> Array:
    """Inverts a symmetric positive definite matrix m."""
    return solve(m, jnp.identity(m.shape[0]))


@partial(jit, static_argnames="retry")
def cholesky(
    m: Array,
    *,
    upper: bool = False,
    sigma: Array | float | None = None,
    retry: bool = False,
) -> Array:
    """Robust Cholesky factorization."""
    noise = jnp.finfo(m.dtype).eps if sigma is None else sigma
    Id = jnp.identity(m.shape[0])
    L = jnp.linalg.cholesky(m + noise * Id, upper=upper)
    return (
        lax.while_loop(
            lambda state: jnp.isnan(state[0]).any(),
            lambda state: (
                jnp.linalg.cholesky(
                    m + (noise := 10 * state[1]) * Id, upper=upper
                ),
                noise,
            ),
            (L, noise),
        )[0]
        if retry
        else L
    )


@jit
def inv_order(order: Array) -> Array:
    """Find the inverse permutation of the given order permutation."""
    n = order.shape[0]
    inv_order = jnp.empty_like(order)
    return inv_order.at[order].set(jnp.arange(n))
