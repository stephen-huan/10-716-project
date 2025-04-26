import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import seaborn as sns
from flax import nnx
from gpjax import kernels
from jax import Array, random
from scipy.sparse.linalg import LinearOperator

from transport.metrics import (
    frobenius_norm,
    kaporin,
    kaporin_precond,
    kl_div,
    operator_norm,
    trace_norm,
)
from transport.nystrom import nystrom, sketch_solve, sketch_solve_eig
from transport.utils import KeyArray, gram

# enable int64/float64
jax.config.update("jax_enable_x64", True)
# set random seed
rng = random.key(0)

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)


def test_preconditioner(
    rng: KeyArray, m: Array, M: Array | LinearOperator | None = None
) -> list[float]:
    """Test the preconditioner with conjugate gradient."""
    n = m.shape[0]
    # multiply i.i.d. normal by covariance matrix to smoothen
    # this gives better results than generating the right hand side directly
    z = random.normal(rng, (n,))
    y = m @ z
    iters = []

    def callback(xk: Array) -> None:
        """Callback called by conjugate gradient after each iteration."""
        iters.append(jnp.linalg.vector_norm(z - xk).item())

    x0 = jnp.zeros(n)
    # jax api doesn't have callback
    _, _ = scipy.sparse.linalg.cg(  # pyright: ignore
        np.asarray(m),
        y,
        x0=x0,
        rtol=rtol,
        atol=0,
        maxiter=maxiter,
        M=M,
        callback=callback,
    )
    return iters


if __name__ == "__main__":
    n = 1 << 10  # number of points
    d = 3  # spatial dimension
    k = 1 << 6  # preconditioning size
    sigma = 1e-8  # noise
    rtol = 1e-12  # relative tolerance for conjugate gradient
    maxiter = 10**4

    kernel = kernels.Matern52(lengthscale=1)
    rng, subkey = random.split(rng)
    x = random.normal(subkey, (n, d))
    m = gram(kernel, x).to_dense() + sigma * jnp.identity(n)

    preconditioners = [
        ("none", None),
        # ("jacobi", jacobi(nnx.split(kernel), x, sigma=sigma))
        (
            "sketch-and-solve",
            sketch_solve(nnx.split(kernel), x, k, sigma=sigma),
        ),
        (
            "sketch-and-solve (eig)",
            sketch_solve_eig(nnx.split(kernel), x, k, sigma=sigma),
        ),
    ]

    for name, preconditioner in preconditioners:
        sns.lineplot(
            test_preconditioner(rng, m, preconditioner),
            label=f"preconditioner={name}",
        )
    plt.yscale("log")
    plt.title("Rate of conjugate gradient convergence")
    plt.xlabel("Iterations")
    plt.ylabel(r"Residual $\Vert x - x_k \Vert$")
    plt.savefig(figures / "cg.png")
    plt.gcf().clear()
