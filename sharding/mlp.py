from typing import Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy.typing as npt

from sharding.engine import JaxShardedEngine


def relu(x):
    return x * (x > 0)


class JaxMlpTP(JaxShardedEngine):
    """
    Tensor-parallel MLP.
    
    Sharding layout:
        x:     [B, D]  -> P('X', None)   batch-sharded, features replicated
        w_in:  [D, F]  -> P(None, 'Y')   column-parallel (split F across Y)
        w_out: [F, D]  -> P('Y', None)   row-parallel (split F across Y)
        out:   [B, D]  -> P('X', None)   batch-sharded (requires all-reduce)
    """
    
    def __init__(
        self,
        axis_shapes: tuple[int, ...] = (1, 1),
        axis_names: tuple[str, ...] = ('X', 'Y'),
        devices: Sequence[jax.Device] | None = None,
        initialize_distributed: bool = False,
    ):
        super().__init__(
            axis_shapes=axis_shapes,
            axis_names=axis_names,
            devices=devices,
            initialize_distributed=initialize_distributed,
        )
        self.params: dict[str, jax.Array] = {}
    
    def load_checkpoint(self, params: dict[str, npt.ArrayLike]) -> None:
        """
        Load and shard parameters.
        
        Args:
            params: Dict with 'w_in' [D, F] and 'w_out' [F, D]
        """
        X, Y = self.axis_names
        self.params = {
            'w_in': self.shard_array(jnp.asarray(params['w_in']), P(None, Y)),
            'w_out': self.shard_array(jnp.asarray(params['w_out']), P(Y, None)),
        }
    
    def forward(self, x: jax.Array) -> jax.Array:
        """
        Forward pass with tensor parallelism.
        
        The second matmul (a @ w_out) produces partial sums that must be
        reduced across the Y axis.
        """
        w_in = self.params['w_in']
        w_out = self.params['w_out']
        
        # x: [B, D], w_in: [D, F/Y] -> h: [B, F/Y]
        h = jnp.einsum('bd,df->bf', x, w_in)
        a = relu(h)
        
        # a: [B, F/Y], w_out: [F/Y, D] -> out_partial: [B, D] (partial sums)
        out_partial = jnp.einsum('bf,fd->bd', a, w_out)
        
        return out_partial  # Compiler inserts all-reduce based on out_shardings
    
    def forward_jit(self):
        """Return a JIT-compiled forward function."""
        X, Y = self.axis_names
        
        @partial(
            jax.jit,
            in_shardings=(
                self.sharding(P(X, None)),    # x
                self.sharding(P(None, Y)),    # w_in
                self.sharding(P(Y, None)),    # w_out
            ),
            out_shardings=self.sharding(P(X, None)),  # Triggers all-reduce
        )
        def _forward(x, w_in, w_out):
            h = jnp.einsum('bd,df->bf', x, w_in)
            a = relu(h)
            out = jnp.einsum('bf,fd->bd', a, w_out)
            return out
        
        return _forward
    
    def backward(self, grads: jax.Array) -> dict[str, jax.Array]:
        """Placeholder for backward pass."""
        raise NotImplementedError


# =============================================================================
# Usage
# =============================================================================

if __name__ == "__main__":
    # Setup
    model = JaxMlpTP(axis_shapes=(1, 1), axis_names=('X', 'Y'))

    with model:
        # Create sharded arrays
        x = jnp.zeros((8, 1024), dtype=jnp.bfloat16)
        x = model.shard_array(x, P('X', None))

        w_in = jnp.zeros((1024, 2048), dtype=jnp.bfloat16)
        w_out = jnp.zeros((2048, 1024), dtype=jnp.bfloat16)

        # Load params
        model.load_checkpoint({'w_in': w_in, 'w_out': w_out})

        # Option 1: Use JIT-compiled forward
        forward_fn = model.forward_jit()
        out = forward_fn(x, model.params['w_in'], model.params['w_out'])

        # # Option 2: Wrap the method call
        # @partial(
        #     jax.jit,
        #     in_shardings=model.sharding(P('X', None)),
        #     out_shardings=model.sharding(P('X', None)),
        # )
        # def run_forward(x):
        #     return model.forward(x)

        # out = run_forward(x)

        print(f"Output shape: {out.shape}")
        print(f"Output sharding: {out.sharding}")