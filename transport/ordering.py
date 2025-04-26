from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax


@jit
def euclidean(x: Array, y: Array) -> Array:
    """Return the distance between points in x and y."""
    x2 = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    y2 = jnp.sum(jnp.square(y), axis=1, keepdims=True)
    return jnp.sqrt(jnp.maximum(x2 - 2 * (x @ y.T) + y2.T, 0))


@partial(jit, static_argnames="reverse")
def reverse_maximin(
    x: Array, initial: Array | None = None, *, reverse: bool = True
) -> tuple[Array, Array]:
    """Return the reverse maximin ordering and length scales."""
    n = x.shape[0]
    # arbitrarily select the first point
    if initial is None or initial.shape[0] == 0:
        dists = jnp.inf * jnp.ones(n)
        start = n - 1
    # use the initial points
    else:
        dists = jnp.min(euclidean(x, initial), axis=1)
        start = n - 1

    def body_fun(
        state: tuple[int, Array], _: None
    ) -> tuple[tuple[int, Array], tuple[Array, Array]]:
        """Select the i-th point."""
        i, dists = state
        # select point with largest minimum distance
        k = jnp.argmax(dists)
        d = dists[k]
        # update distances
        dists = jnp.minimum(dists, euclidean(x, x[k, jnp.newaxis]).flatten())
        return (i - 1, dists), (k, d)

    return lax.scan(body_fun, (start, dists), length=n, reverse=reverse)[1]
