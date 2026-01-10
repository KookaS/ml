import jax
import jax.numpy as jnp

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