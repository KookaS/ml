from typing import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy.typing as npt

from sharding.engine import JaxShardedEngine


def relu(x):
    return x * (x > 0)


class JaxMlpTP(JaxShardedEngine):
    """
    Tensor-parallel MLP with sequence parallelism.

    Sharding layout (all along Y axis only):
        x:     [B, D]  -> P(None, 'Y')   features sharded [B, D/Y]
        w_in:  [D, F]  -> P(None, 'Y')   col-parallel [D, F/Y]
        w_out: [F, D]  -> P('Y', None)   row-parallel [F/Y, D]
        out:   [B, D]  -> P(None, 'Y')   features sharded [B, D/Y]

    Communication pattern:
        1. all-gather x along Y: [B, D/Y] -> [B, D]
        2. local matmuls
        3. reduce-scatter along Y: [B, D] -> [B, D/Y]
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
    
    def forward_jit(self, x: jax.Array) -> jax.Array:
        """
        Forward pass using GSPMD (jax.jit with shardings).

        The compiler automatically inserts all-gather/reduce-scatter
        based on input/output shardings. No manual collectives needed.

        Sharding flow:
            x:     [B, D/Y]  -> all-gather -> [B, D]
            w_in:  [D, F/Y]  col-parallel
            w_out: [F/Y, D]  row-parallel
            out:   [B, D]    -> reduce-scatter -> [B, D/Y]
        """
        if not hasattr(self, '_forward_jit_compiled'):
            _, Y = self.axis_names

            @partial(
                jax.jit,
                in_shardings=(
                    self.sharding(P(None, Y)),    # x [B, D/Y]
                    self.sharding(P(None, Y)),    # w_in [D, F/Y]
                    self.sharding(P(Y, None)),    # w_out [F/Y, D]
                ),
                out_shardings=self.sharding(P(None, Y)),  # out [B, D/Y]
            )
            def _forward(x, w_in, w_out):
                # Compiler auto-inserts all-gather for x, reduce-scatter for out
                h = jnp.einsum('bd,df->bf', x, w_in)
                a = relu(h)
                out = jnp.einsum('bf,fd->bd', a, w_out)
                return out

            self._forward_jit_compiled = _forward

        return self._forward_jit_compiled(x, self.params['w_in'], self.params['w_out'])

    def forward_sm(self, x: jax.Array) -> jax.Array:
        """
        Forward pass using shard_map with explicit collectives.

        Manual control over all-gather and reduce-scatter operations.
        The function receives local shards and we explicitly call collectives.

        Sharding flow:
            x:     [B, D/Y]   local shard
            w_in:  [D, F/Y]   col-parallel shard
            w_out: [F/Y, D]   row-parallel shard

            1. x_full = all_gather(x)      -> [B, D]
            2. h = x_full @ w_in           -> [B, F/Y]
            3. a = relu(h)                 -> [B, F/Y]
            4. out_partial = a @ w_out     -> [B, D] partial sums
            5. out = reduce_scatter(out)   -> [B, D/Y]
        """
        if not hasattr(self, '_forward_sm_compiled'):
            _, Y = self.axis_names

            @jax.jit
            @partial(
                shard_map,
                mesh=self.mesh,
                in_specs=(
                    P(None, Y),    # x [B, D/Y]
                    P(None, Y),    # w_in [D, F/Y]
                    P(Y, None),    # w_out [F/Y, D]
                ),
                out_specs=P(None, Y),  # out [B, D/Y]
            )
            def _forward(x, w_in, w_out):
                # 1. All-gather x: [B, D/Y] -> [B, D]
                x_full = self.all_gather(x, axis_name=Y, axis_idx=1, tiled=True)

                # 2. First matmul: [B, D] @ [D, F/Y] -> [B, F/Y]
                h = jnp.einsum('bd,df->bf', x_full, w_in)
                a = relu(h)

                # 3. Second matmul: [B, F/Y] @ [F/Y, D] -> [B, D] (partial sums)
                out_partial = jnp.einsum('bf,fd->bd', a, w_out)

                # 4. Reduce-scatter: [B, D] -> [B, D/Y]
                out = self.reduce_scatter(out_partial, axis_name=Y, axis_idx=1, tiled=True)
                return out

            self._forward_sm_compiled = _forward

        return self._forward_sm_compiled(x, self.params['w_in'], self.params['w_out'])

    def backward(self, grads: jax.Array) -> dict[str, jax.Array]:
        """Placeholder for backward pass."""
        raise NotImplementedError


# =============================================================================
# Usage
# =============================================================================

if __name__ == "__main__":

    import os

    # Create 8 virtual CPU devices for testing mesh parallelism (must be set before JAX import)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

    with jax.profiler.trace("./trace", create_perfetto_trace=True):

        # Setup: 2x4 mesh = 8 virtual devices
        model = JaxMlpTP(axis_shapes=(1, 8), axis_names=('X', 'Y'))

        with model:
            # Create sharded arrays
            x = jnp.zeros((8, 1024), dtype=jnp.bfloat16)
            x = model.shard_array(x, P(None, 'Y'))

            w_in = jnp.zeros((1024, 2048), dtype=jnp.bfloat16)
            w_out = jnp.zeros((2048, 1024), dtype=jnp.bfloat16)

            # Load params
            model.load_checkpoint({'w_in': w_in, 'w_out': w_out})            

            # Test GSPMD approach (jax.jit with shardings)
            print("=== forward_jit (GSPMD) ===")
            out_jit = model.forward_jit(x)
            print(f"Output shape: {out_jit.shape}")
            print(f"Output sharding: {out_jit.sharding}")

            # Test shard_map approach (explicit collectives)
            print("\n=== forward_sm (shard_map) ===")
            out_sm = model.forward_sm(x)
            print(f"Output shape: {out_sm.shape}")
            print(f"Output sharding: {out_sm.sharding}")

            # Verify both produce same result
            print(f"\nResults match: {jnp.allclose(out_jit, out_sm)}")