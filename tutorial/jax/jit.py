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
y = jnp.ones((1000, 1000)) # flaot32


# JIT compilation
main(x, y).block_until_ready() # warmup
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     main(x,y).block_until_ready()

# AOT compilation
main_jit = jax.jit(main).lower(x,y).compile()
main_jit(x,y)
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     main_jit(x,y)

# ================
#   CONDITION
# ================

# # if else won't work unless
# @jax.jit
# def conditional(x,y):
#     if x > y:
#         return y
#     else:
#         return jnp.exp(x)

@jax.jit
def conditional(x,y):
    jax.lax.cond(x > y, lambda: y, lambda: jnp.exp(x))
    
conditional(3., 4.)

# However if you know the inputs the if/else should be replaced by in-line operations
from functools import partial
@partial(jax.jit, static_argnames=('x',))
def conditional(x,y):
    if x < 1:
        return y
    else:
        return jnp.exp(y)

conditional(3., 4.)