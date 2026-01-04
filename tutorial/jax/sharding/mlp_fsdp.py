import os

# Create 8 virtual CPU devices for testing mesh parallelism (must be set before JAX import)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from tutorial.jax.mlp import Mlp
from tutorial.jax.util import benchmark, inspect_array


B, D, F = 80, 640, 2560

if __name__ == "__main__":

    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, P('X'))
    model = Mlp()

    """
    Keeps the Data local and stable, moves the Weights.
    - shard input along B to keep the data local
    - shard Win and Wout NOT along contracting dimension, to reduce storing heavy data such as weights, gradients, optimizer states
    """

    x = jnp.ones((B,D), dtype=jnp.bfloat16, device=NamedSharding(mesh, P('X', None)))
    inspect_array(x, "X -- Input")
    w_in = jnp.ones((D, F), dtype=jnp.float32, device=NamedSharding(mesh, P('X', None)))
    w_out = jnp.ones((F, D), dtype=jnp.float32, device=NamedSharding(mesh, P(None, 'X')))
    
    out, activations = model.forward(w_in, w_out, x)
    inspect_array(out, "Out")

    # simulated loss gradient (dLoss/dOut)
    grad_out = jnp.ones((B,D), dtype=jnp.bfloat16, device=NamedSharding(mesh, P('X', None)))
    grads = model.backward(w_out, grad_out, activations.copy())
    inspect_array(grads['layer_out/weights'], "dWout")
    inspect_array(grads['layer_in/weights'], "dWin")

    benchmark("Forward", model.forward, w_in, w_out, x)
    benchmark("Backward", model.backward, w_out, grad_out, activations)