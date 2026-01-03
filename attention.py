from functools import partial
import jax
import jax.numpy as jnp

B, D, N = 8, 256, 2

class MultiAttentionHead:

    def __init__(self):
        self.params = {}
    
    def load_checkpoint(self, params: dict[str, jax.Array]) -> None:
        """
        Load and shard parameters.
        
        Args:
            params: Dict with 'w_q' [D, N, H] and 'w_k' [D, N, H], 'w_v' [D, N, H], layer_out/weights [N, H, D]

        D == N*H
        """
        self.params = {
            'w_q': params['w_q'],
            'w_k': params['w_k'],
            'w_v': params['w_v'],
            'layer_out/weights': params['layer_out/weights'],
        }
    
    @partial(jax.jit, static_argnames=('self',))
    def forward(self, x):
        activations = []
        q = jnp.einsum('btd,dnh->btnh', x, self.params['w_q'])
        k = jnp.einsum('bsd,dnh->bsnh', x, self.params['w_k'])
        v = jnp.einsum('bsd,dnh->bsnh', x, self.params['w_v'])

        scores = jnp.softmax(jnp.einsum('btnh,bsnh->btns', q, k)) # H is the contracting dimension
        out = jnp.einsum('btns,bsnh->btnh', scores, v) # S is contracting dimension
        # out = jnp.einsum('btnh,nhd->btd', out, self.params['layer_out/weights']) # reshape NH to D
        return out, activations

    def backward(self, grads, activations):
        ...