import os

from util import inspect_array

# Create 8 virtual CPU devices for testing mesh parallelism (must be set before JAX import)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# mesh creation
devices = mesh_utils.create_device_mesh((4,2))
mesh = Mesh(devices, axis_names=('X', 'Y'))

# memory allocation
x = jnp.zeros((100, 100), dtype=jnp.bfloat16, device=NamedSharding(mesh, P('X', 'Y'))) # In[Bx, Dy]
w = jnp.zeros((100, 400), dtype=jnp.bfloat16, device=NamedSharding(mesh, P('Y', None))) # W[Dy, F]

# ==================
# JIT auto
# ==================

# # JIT compilation
# @jax.jit
# def main(x, w):
#     return jnp.einsum('bd,df->bf', x, w)  # Out[Bx, F]

# main(x,w).block_until_ready() # Out[B, F]
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     main(x,w).block_until_ready()

# # AOT compilation
def main(x, w):
    return jnp.einsum('bd,df->bf', x, w) # Out[Bx, F]

jit_main = jax.jit(main).lower(x, w, ).compile()
jit_main(x,w) # Out[B, F]

# Control the dimensions yourself
def main(x, w):
    out = jnp.einsum('bd,df->bf', x, w)
    return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, P('X', 'Y'))) # Out[Bx, Fy]

jit_main = jax.jit(main).lower(x, w).compile()
jit_main(x,w) # Out[B, F]

# ==================
# MANUAL shard_map
# ==================

# Manual control over collectives with shard_map
@jax.shard_map(mesh=mesh, in_specs=(P('X','Y'), P('Y', None)), out_specs=P('X', None))
def main(x, w):
    out = jnp.einsum('bd,df->bf', x, w) # In[Bx, Dy] *Dy W[Dy, F] -> Out[Bx, F]{Uy}
    inspect_array(out, "Out -- Matmul")

    # einsum with shard_map does not perform all-reduce automatically along the contracting dimension D

    out = jax.lax.psum(out, axis_name='Y') # all-reduce accross the contracting dimension, Out[Bx, F]{Uy} -> Out[Bx, F]
    inspect_array(out, "Out -- All-Reduce")
    return out

jit_main = jax.jit(main).lower(x, w).compile()
out = jit_main(x,w)
inspect_array(out, "Out -- Final")


# we are trying here to mimic all-reduce with reduce-scatter + all-gather
# check_rep=False is required because the compiler cannot guarantee of the mesh dimensions
@jax.shard_map(mesh=mesh, in_specs=(P('X','Y'), P('Y', None)), out_specs=P('X', None), check_vma=False)
def main(x, w):
    out = jnp.einsum('bd,df->bf', x, w) # In[Bx, Dy] *Dy W[Dy, F] -> Out[Bx, F]{Uy}
    inspect_array(out, "Out -- Matmul")

    # axis_name='Y': The mesh axis we communicate over
    # axis=1 | scatter_dimension=1: The array dimension we concatenate along
    # tiled=True: Concatenates data (shape 100,400) instead of stacking it (shape 4,25,400)

    out = jax.lax.psum_scatter(out, axis_name='Y',scatter_dimension=1, tiled=True) # reduce-scatter accross the contracting dimension, Out[Bx, F]{Uy}
    inspect_array(out, "Out -- Reduce-Scatter")

    out = jax.lax.all_gather(out, axis_name='Y', axis=1, tiled=True) # all-gather the results, Out[Bx, F]{Uy} -> Out[Bx, F]
    inspect_array(out, "Out -- All-Gather")
    return out

jit_main = jax.jit(main).lower(x, w).compile()
out = jit_main(x,w)
inspect_array(out, "Out -- Final")