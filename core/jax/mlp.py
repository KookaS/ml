from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp


@jax.jit
def relu(x):
    return x * (x > 0)

class Mlp:
    """
    Basic MLP
    """
    
    def forward(self, w_in, w_out, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Forward pass for MLP.
        
        1. H[B, F] = X[B, D] @ Win[D, F]
        2. A = Activatin(H)
        3. Out[B, D] = A[B, F] @ Wout[F, D]
        --> Out = Wout @ Activation(X @ Win)
        """
        activations = []
        activations.append(x)
        h = jnp.einsum('bd,df->bf', x, w_in)
        a = relu(h)
        activations.append(a)
        out = jnp.einsum('bf,fd->bd', a, w_out)
        return out, activations

    def backward(self, w_out, out_grad, activations: jax.Array) -> dict[str, jax.Array]:
        """Backward pass for MLP.
        
        1. get dOut[B, D]
        2. dWout[F, D] = A^T[F, B] @ dOut[B, D]

        3. A[B, F] = dOut[B, D] @ Wout^T[D, F]
        4. H = Activation^-1(A)
        5. dWin[D, F] = X^T[D, B] @ H[B, F]
        --> dWin = X^T @ (Activation^-1(dOut @ Wout^T))
        """
        a = activations.pop()
        w_out_grad = jnp.einsum('bf,bd->fd', a, out_grad)

        a_grad = jnp.einsum('bd,fd->bf', out_grad, w_out)
        h_grad = a_grad * (a > 0)
        x = activations.pop()
        w_in_grad = jnp.einsum('bd,bf->df', x, h_grad)

        return {
            'layer_out/weights': w_out_grad,
            'layer_in/weights': w_in_grad,
        }