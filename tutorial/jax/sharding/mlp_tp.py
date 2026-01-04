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

    import os

    # Create 8 virtual CPU devices for testing mesh parallelism (must be set before JAX import)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

    devices = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices, P('Y'))
    model = Mlp()

    """
    Keeps the Weights local and stable, moves the Data.
    - Input is sharded along D dimension to avoid storing the entire input, required during backprop
    - Win and Wout are sharded along contracting dimension to shard the compute
    """
    x = jnp.ones((B,D), dtype=jnp.bfloat16, device=NamedSharding(mesh, P(None, 'Y')))
    inspect_array(x, "X -- Input")
    w_in = jnp.ones((D, F), dtype=jnp.float32, device=NamedSharding(mesh, P(None, 'Y')))
    w_out = jnp.ones((F, D), dtype=jnp.float32, device=NamedSharding(mesh, P('Y', None)))
    
    out, activations = model.forward(w_in, w_out, x)
    inspect_array(out, "Out")

    # simulated loss gradient (dLoss/dOut)
    grad_out = jnp.ones((B,D), dtype=jnp.bfloat16, device=NamedSharding(mesh, P(None, 'Y')))
    grads = model.backward(w_out, grad_out, activations.copy())
    inspect_array(grads['layer_out/weights'], "dWout")
    inspect_array(grads['layer_in/weights'], "dWin")

    benchmark("Forward", model.forward, w_in, w_out, x)
    benchmark("Backward", model.backward, w_out, grad_out, activations)