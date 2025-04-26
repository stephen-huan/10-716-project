from functools import partial

import jax.numpy as jnp
from jax import Array, jit

from .utils import logdet, solve

# TODO: scalable randomized approximations


@jit
def frobenius_norm(a: Array) -> Array:
    """Frobenius norm of a."""
    return jnp.linalg.matrix_norm(a, ord="fro")


# TODO: eigensolver, by default this uses svd


@jit
def operator_norm(a: Array) -> Array:
    """Operator norm of a."""
    return jnp.linalg.matrix_norm(a, ord=2)


@jit
def trace_norm(x: Array, y: Array) -> Array:
    """Trace norm error trace(x - y)."""
    return jnp.trace(x - y)


@jit
def kl_div(x: Array, y: Array) -> float:
    """
    Computes the KL divergence between the multivariate Gaussians
    at 0 with covariance x and y, i.e. D_KL(N(0, x) || N(0, y)).
    """
    n = x.shape[0]
    return (jnp.trace(solve(y, x)) + logdet(y) - logdet(x) - n) / 2


@jit
def logdet_inv_factor(L: Array) -> Array:
    """Compute the logdet given a Cholesky factor of the precision."""
    return -2 * jnp.sum(jnp.log(jnp.diagonal(L)))


@jit
def kl_div_inv_factor(a: Array, L: Array) -> float:
    """
    Computes the KL divergence assuming L is optimal.

    Equivalent to kl_div(a, inv(L @ L.T)).
    """
    logdet_theta = a if a.ndim == 0 else logdet(a)
    return (logdet_inv_factor(L) - logdet_theta) / 2


@jit
def kaporin(a: Array) -> Array:
    """Kaporin condition number of A."""
    n = a.shape[0]
    return jnp.trace(a) / (n * jnp.exp(logdet(a) / n))


@partial(jit, static_argnames="log")
def kaporin_precond(x: Array, y: Array, *, log: bool = True) -> Array:
    """Kaporin condition number of y^{-1} x."""
    n = x.shape[0]
    logk = (
        n * jnp.log(jnp.trace(solve(y, x)))
        + logdet(y)
        - logdet(x)
        - n * jnp.log(n)
    ) / n
    return logk if log else jnp.exp(logk)
