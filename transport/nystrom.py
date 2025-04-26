from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from flax import nnx
from gpjax.kernels import AbstractKernel
from jax import Array, jit, lax
from scipy.sparse.linalg import LinearOperator

from .utils import (
    Kernel,
    argmax_masked,
    cross_covariance,
    diagonal,
    identity_like,
    index_dtype,
    inv_order,
    solve,
)


def __chol_update(
    cond_var: Array, factor: Array, i: int, k: int | Array
) -> tuple[Array, Array]:
    """Condition the i-th column of the Cholesky factor by the k-th point."""
    n = cond_var.shape[0]
    # update Cholesky factor by left looking
    # -factor[:, :i] @ factor[k, :i] is more efficient but size must be static
    row = factor[k]
    row = row.at[i].set(0.0)
    factor = factor.at[:, i].add(-factor @ row)
    # https://github.com/google/jax/issues/19162
    factor = factor.at[:, i].multiply(jnp.reciprocal(jnp.sqrt(factor[k, i])))
    # update conditional variance
    cond_var = cond_var.at[:].add(-jnp.square(factor[:n, i]))
    return cond_var, factor


# @partial(jit, static_argnums=2, static_argnames="full_rank")
def nystrom(
    nnx_kernel: Kernel,
    x: Array,
    s: int,
    *,
    sigma: Array | float = 0.0,
    full_rank: bool = False,
) -> tuple[Array, Array]:
    """Nyström approximation."""
    kernel: AbstractKernel = nnx.merge(*nnx_kernel)
    n = x.shape[0]
    s = min(s, n)
    # initialization
    int_dtype = index_dtype(x)
    indices = jnp.zeros(s, dtype=int_dtype)
    candidates = jnp.ones(n, dtype=jnp.bool_)
    cond_var = diagonal(kernel, x).diag + sigma
    factor = jnp.zeros((n, s), dtype=cond_var.dtype)
    State: TypeAlias = tuple[Array, Array, Array, Array]  # pyright: ignore
    state = (indices, candidates, cond_var, factor)

    def body_fun(i: int, state: State) -> State:
        """Select the best index on the i-th iteration."""
        indices, candidates, cond_var, factor = state
        # pick best entry --- could randomly sample here
        # (RPCholesky) but use maximum for simplicity
        k = argmax_masked(cond_var, candidates)
        # update data structures
        cov_k = cross_covariance(kernel, x, x[k, jnp.newaxis])
        factor = factor.at[:, i].set(cov_k.flatten())
        return (
            indices.at[i].set(int_dtype(k)),
            candidates.at[k].set(False),
            *__chol_update(cond_var, factor, i, k),
        )

    indices, candidates, _, factor = lax.fori_loop(0, s, body_fun, state)
    # add the remaining unselected indices arbitrarily
    order = jnp.concatenate((indices, jnp.arange(n)[candidates]))
    return factor, order


def make_LinearOperator(
    linop: Array | LinearOperator, matmat: Callable[[Array], Array]
) -> LinearOperator:
    """Make a symmetric LinearOperator from the provided data."""
    return LinearOperator(
        shape=linop.shape,
        dtype=linop.dtype,
        matvec=matmat,  # pyright: ignore[reportCallIssue]
        rmatvec=matmat,  # pyright: ignore[reportCallIssue]
        matmat=matmat,  # pyright: ignore[reportCallIssue]
        rmatmat=matmat,  # pyright: ignore[reportCallIssue]
    )


def cholesky_LinearOperator(L: Array) -> LinearOperator:
    """Return a LinearOperator object from a Cholesky factor."""

    @jit
    def matmat(x: Array) -> Array:
        """Compute L (L^T x)"""
        return L.dot(L.T.dot(x))

    return make_LinearOperator(L, matmat)


def permutation_LinearOperator(
    linop: LinearOperator, order: Array
) -> LinearOperator:
    """Return a LinearOperator with re-ordering."""
    order_inv = inv_order(order)

    # @jit
    def matmat(x: Array) -> Array:
        """Apply an ordering to x."""
        return linop(x[order])[order_inv]  # pyright: ignore

    return make_LinearOperator(linop, matmat)


def jacobi(
    nnx_kernel: Kernel, x: Array, *, sigma: Array | float = 0.0
) -> LinearOperator:
    """Jacobi preconditioning."""
    kernel: AbstractKernel = nnx.merge(*nnx_kernel)
    var = diagonal(kernel, x).diag + sigma

    @jit
    def matmat(x: Array) -> Array:
        """Apply Jacobi preconditioning."""
        return x / var

    return make_LinearOperator(identity_like(var), matmat)


def sketch_solve(
    nnx_kernel: Kernel, x: Array, s: int, *, sigma: Array | float = 0.0
) -> LinearOperator:
    """Sketch-and-solve preconditioning."""
    # factor just the kernel matrix without the additive noise
    factor, _ = nystrom(nnx_kernel, x, s, sigma=0)
    m = factor.T @ factor + sigma * identity_like(factor.T)

    @jit
    def matmat(x: Array) -> Array:
        """Apply sketch-and-solve preconditioning."""
        # sherman-morrison-woodbury
        return -factor @ solve(m, factor.T @ x) / sigma + x / sigma

    # no need to handle order as column permutations cancel
    return make_LinearOperator(identity_like(factor), matmat)


def sketch_solve_eig(
    nnx_kernel: Kernel, x: Array, s: int, *, sigma: Array | float = 0.0
) -> LinearOperator:
    """Sketch-and-solve preconditioning."""
    # factor just the kernel matrix without the additive noise
    factor, _ = nystrom(nnx_kernel, x, s, sigma=0)
    # factor @ factor.T = u @ jnp.diag(eigvals) @ u.T
    u, eigvals, _ = jnp.linalg.svd(factor, full_matrices=False)
    eigvals = jnp.square(eigvals)
    diagonal = jnp.reciprocal(eigvals + sigma) - jnp.reciprocal(sigma)

    @jit
    def matmat(x: Array) -> Array:
        """Apply sketch-and-solve preconditioning."""
        # sherman-morrison-woodbury
        return u @ (diagonal * (u.T @ x)) + x / sigma

    # no need to handle order as column permutations cancel
    return make_LinearOperator(identity_like(factor), matmat)
