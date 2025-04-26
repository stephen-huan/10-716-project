import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import random

from transport.ordering import euclidean, reverse_maximin

# enable int64/float64
jax.config.update("jax_enable_x64", True)
# set random seed
rng = random.key(0)


def np_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the distance between points in x and y."""
    return scipy.spatial.distance.cdist(x, y, "euclidean")


def np_reverse_maximin(
    x: np.ndarray, initial: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reverse maximin ordering and length scales."""
    # O(n^2)
    n = len(x)
    indices = np.zeros(n, dtype=np.int64)
    # minimum distance to a point in indexes at the time of each selection
    lengths = np.zeros(n)
    # arbitrarily select the first point
    if initial is None or initial.shape[0] == 0:
        k = 0
        # minimum distance to a point in indexes
        dists = np_euclidean(x, x[k : k + 1]).flatten()
        indices[-1] = k
        lengths[-1] = np.inf
        start = n - 2
    # use the initial points
    else:
        dists = np.min(np_euclidean(x, initial), axis=1)
        start = n - 1

    for i in range(start, -1, -1):
        # select point with largest minimum distance
        k = np.argmax(dists)
        indices[i] = k
        # update distances
        lengths[i] = dists[k]
        dists = np.minimum(dists, np_euclidean(x, x[k : k + 1]).flatten())

    return indices, lengths


if __name__ == "__main__":
    n = 1 << 10  # number of points
    m = 1 << 5
    d = 3  # spatial dimension

    rng, subkey1, subkey2 = random.split(rng, num=3)
    x = random.normal(subkey1, (n, d))
    y = random.normal(subkey2, (m, d))

    d = np_euclidean(np.array(x), np.array(y))
    assert jnp.allclose(euclidean(x, y), d), "euclidean wrong."

    order_ans, lengths_ans = np_reverse_maximin(np.array(x))
    order, lengths = reverse_maximin(x)
    assert jnp.allclose(lengths, lengths_ans), "length wrong."
