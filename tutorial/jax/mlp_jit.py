from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp

B, D, F = 10, 100, 400

@jax.jit
def relu(x):
    return x * (x > 0)
    

class Mlp:
    """
    Basic MLP
    """
    
    def __init__(self):
        self.params: dict[str, jax.Array] = {
            'w_in': jnp.ones((D, F)),
            'w_out': jnp.ones((F, D)),
        }

    def load_checkpoint(self, params: dict[str, jax.Array]) -> None:
        """
        Load and shard parameters.
        
        Args:
            params: Dict with 'w_in' [D, F] and 'w_out' [F, D]
        """
        self.params = params
    
    @partial(jax.jit, static_argnames=('self',))
    def forward(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Forward pass for MLP.
        
        1. H[B, F] = X[B, D] @ Win[D, F]
        2. A = Activatin(H)
        3. Out[B, D] = A[B, F] @ Wout[F, D]
        --> Out = Wout @ Activation(X @ Win)
        """
        activations = []
        activations.append(x)
        h = jnp.einsum('bd,df->bf', x, self.params['w_in'])
        a = relu(h)
        activations.append(a)
        out = jnp.einsum('bf,fd->bd', a, self.params['w_out'])
        return out, activations

    @partial(jax.jit, static_argnames=('self',))
    def backward(self, grads: jax.Array, activations: jax.Array) -> dict[str, jax.Array]:
        """Backward pass for MLP.
        
        1. get dOut[B, D]
        2. dWout[F, D] = A^T[F, B] @ dOut[B, D]

        3. A[B, F] = dOut[B, D] @ Wout^T[D, F]
        4. H = Activation^-1(A)
        5. dWin[D, F] = X^T[D, B] @ H[B, F]
        --> dWin = X^T @ (Activation^-1(dOut @ Wout^T))
        """
        out_grad = grads['layer_out/weights']
        a = activations.pop()
        w_out_grad = jnp.einsum('bf,bd->fd', a, out_grad)

        a_grad = jnp.einsum('bd,fd->bf', out_grad, self.params['w_out'])
        h_grad = a_grad * (a_grad > 0)
        x = activations.pop()
        w_in_grad = jnp.einsum('bd,bf->df', x, h_grad)

        return {
            'layer_out/weights': w_out_grad,
            'layer_in/weights': w_in_grad,
        }


if __name__ == "__main__":

    import os

    # Create 8 virtual CPU devices for testing mesh parallelism (must be set before JAX import)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

    x = jnp.ones((B,D), dtype=jnp.bfloat16)

    model = Mlp()
    out, activations = model.forward(x)
    print(out.shape)

    target = jnp.ones((B,D), dtype=jnp.bfloat16)
    grads = {
        'layer_out/weights': target-out
    }
    grads = model.backward(grads, activations)
    print(grads['layer_out/weights'].shape, grads['layer_in/weights'].shape)
