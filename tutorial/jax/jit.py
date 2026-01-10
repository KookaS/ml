import jax
import jax.numpy as jnp

@jax.jit
def main(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    c = a * b
    d = c + x
    e = d * y
    return jax.nn.relu(e)    

x = jnp.ones((1000, 1000))
y = jnp.ones((1000, 1000)) # float32


# JIT compilation
main(x, y).block_until_ready() # warmup
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     main(x,y).block_until_ready()

# AOT compilation
main_jit = jax.jit(main).lower(x,y).compile()
main_jit(x,y)
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     main_jit(x,y)
