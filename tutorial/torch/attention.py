import torch
from torch import einsum

class Attention:
    """
    Multi head attention

    B batch size
    S query size
    T key/value size (or time)
    N number of heads
    H head dimension
    """

    def __init__(self, d_model, n_heads, head_dim):
        self.activations = []
        # init the weights for the optimizer
        self.wq = torch.zeros((d_model, n_heads, head_dim), dtype=torch.float32)
        self.wk = torch.zeros((d_model, n_heads, head_dim), dtype=torch.float32)
        self.wv = torch.zeros((d_model, n_heads, head_dim), dtype=torch.float32)
        self.wo = torch.zeros((d_model, n_heads, head_dim), dtype=torch.float32)

    def load_checkpoint(self, params):
        # refill the empty tensors
        self.wq[...] = params['query/weights'][...]
        self.wk[...] = params['key/weights'][...]
        self.wv[...] = params['value/weights'][...]
        self.wo[...] = params['out/weights'][...]
    
    def forward(self, x):
        """
        Q[B, S, N, H] = X[B, S, D] @ Wq[D, N, H]
        K[B, T, N, H] = X[B, T, D] @ Wk[D, N, H]
        V[B, T, N, H] = X[B, T, D] @ Wv[D, N, H]

        scores[B, S, N, T] = softmax(Q @ K)
        scaling usually happens by 1/sqrt(head_dim)
        qkv[B, S, N, H] = scores[B, S, N, T] @ V[B, T, N, H]

        att[B, S, D] = qkv[B, S, N, H] @ Wo[B, D, N, H]
        x = x + att

        x = layerNormRMS(x)
        """
        q = einsum('bsd,dnh->bsnh', x, self.wq)
        k = einsum('btd,dnh->btnh', x, self.wk)
        v = einsum('btd,dnh->btnh', x, self.wv)

        # SCALING
        qk = einsum('bsnh,btnh->bsnt', q, k) # contract over head dimensions
        qk /= q.shape[-1]**0.5

        # MASKING (decoder only)
        seq = q.shape[1] # B S N T
        mask = torch.arange(seq)[:, None] >= torch.arange(seq)[None, :] 
        mask = torch.where(mask, 0.0, -torch.inf)
        qk += mask[None, :, None, :] # B S N T

        # ATTENTION SCORE
        # for every query, we distribute 100% of its attention capacity across the available keys
        scores = torch.softmax(qk, dim=-1) # the -inf will turn to 0 with softmax

        qkv = einsum('bsnt,btnh->bsnh', scores, v)
        
        attention = einsum('bsnh,dnh->bsd', qkv, self.wo)

        # RESIDUAL CONNECTION (for vanishing gradient problem)
        x += attention

        # LAYER NORM (post-norm), modern LLMs use (pre-norm)
        x = torch.nn.functional.rms_norm(x, (x.shape[-1],))

        return x


    def backward(self, out_grad):
        """
        """
        ...
        


if __name__ == "__main__":
    D, N = 64, 8
    H = D//N
    B, S = 2, 10 # batch, tokens per batch

    torch.manual_seed(42)
    query = torch.randn(B, S, D, dtype=torch.float32) # bf16 usually
    params = {
        'query/weights': torch.randn(D, N, H, dtype=torch.float32),
        'key/weights': torch.randn(D, N, H, dtype=torch.float32),
        'value/weights': torch.randn(D, N, H, dtype=torch.float32),
        'out/weights': torch.randn(D, N, H, dtype=torch.float32),
    }

    model = Attention(D, N, H)
    model.load_checkpoint(params)

    out = model.forward(query)

    # # simulated loss gradient (dLoss/dOut)
    # grad_out = torch.randn(B, D, dtype=torch.bfloat16)
    # grads = model.backward(grad_out)

    # print(f"Gradient W_in shape: {grads['layer_in/weights'].shape}")
    # print(f"Gradient W_out shape: {grads['layer_out/weights'].shape}")
    # print(f"Gradient X shape: {grads['input'].shape}")
