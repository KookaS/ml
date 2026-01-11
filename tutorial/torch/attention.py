import torch

from tutorial.torch.softmax import d_softmax, softmax

class AttentionFn(torch.autograd.Function):
    """
    Multi head attention

    B batch size
    S query size
    T key/value size (or time)
    N number of heads
    H head dimension
    """

    @staticmethod
    def forward(ctx, x, wq, wk, wv, wo, masking):
        """
        Q[B, S, N, H] = X[B, S, D] @ Wq[D, N, H]
        K[B, T, N, H] = X[B, T, D] @ Wk[D, N, H]
        V[B, T, N, H] = X[B, T, D] @ Wv[D, N, H]

        qk[B, S, N, T] = Q[B, S, N, H] @H K[B, T, N, H]
        qk = qk/sqrt(head_dim)
        scores[B, S, N, T] = softmax(qk)
        qkv[B, S, N, H] = scores[B, S, N, T] @ V[B, T, N, H]

        att[B, S, D] = qkv[B, S, N, H] @ Wo[B, D, N, H]
        """
        ctx.masking = masking

        q = torch.einsum('bsd,dnh->bsnh', x, wq)
        k = torch.einsum('btd,dnh->btnh', x, wk)
        v = torch.einsum('btd,dnh->btnh', x, wv)

        # SCALING
        qk = torch.einsum('bsnh,btnh->bsnt', q, k) # contract over head dimensions
        qk /= q.shape[-1]**0.5

        # MASKING (decoder only), in production you should never use if/else with GPUs
        if masking:
            seq = q.shape[1] # B S N T
            mask = torch.arange(seq)[:, None] >= torch.arange(seq)[None, :] 
            mask = torch.where(mask, 0.0, -torch.inf)
            qk += mask[None, :, None, :] # B S N T
            # mask = torch.tril(torch.ones(S, T)) == 0
            # qk = qk.masked_fill(mask, float('-inf'))

        # ATTENTION SCORE
        # for every query, we distribute 100% of its attention capacity across the available keys
        scores = softmax(qk, dim=-1) # the -inf will turn to 0 with softmax

        qkv = torch.einsum('bsnt,btnh->bsnh', scores, v)
        
        attention = torch.einsum('bsnh,dnh->bsd', qkv, wo)

        # # RESIDUAL CONNECTION (for vanishing gradient problem)
        # x += attention
        # # LAYER NORM (post-norm), modern LLMs use (pre-norm)
        # x = torch.nn.functional.rms_norm(x, (x.shape[-1],))

        ctx.save_for_backward(x, q, k, v, wq, wk, wv, wo, scores)

        return attention

    @staticmethod
    def backward(ctx, grad_out):
        """
        dAtt = dL/dAttention = grad_out[B, S, D]
        1. gradient through Wo
        dWo[D, N, H] = qkv[B, S, N, H] @{B,S} dAtt[B, S, D]

        2. gradient through Wv
        dqkv[B, S, N, H] = datt[B, S, D] @D Wo[B, D, N, H]
        dV[B, T, N, H] = scores[B, S, N, T] @S dqkv[B, S, N, H]
        dWv[D, N, H] = X[B, T, D] @(B, T) dV[B, T, N, H]

        3. gradient through softmax
        dScores[B, S, N, T] = dqkv[B, S, N, H] @H V[B, T, N, H]
        ---
        d_softmax(grad_out, s) = grad_in
        grad_in = s * (grad_out - sum(grad_out * s))
        grad_in is d_qk, grad_out is dScores, s is scores
        ---
        dqk = d_softmax(dScores, scores) --> s * (dout - (dout * s))
        dqk = dqk/sqrt(head_dim)

        4. gradient through masking
        Not needed because nul scores produce nul gradient

        5. gradient through Wq
        dQ[B, S, N, H] = dqk[B, S, N, T] @T K[B, T, N, H]
        dWq[D, N, H] = X[B, S, D] @{B, S} dQ[B, S, N, H]

        6. gradient through Wk
        dK[B, T, N, H] = Q[B, S, N, H] @S dqk[B, S, N, T]
        dWk[D, N, H] = X[B, T, D] @{B, S} dQ[B, T, N, H]

        7. gradient through X
        dXq[B, S, D] = Wq[D, N, H] @{N, H} dQ[B, S, N, H]
        dXk[B, T, D] = Wk[D, N, H] @{N, H} dK[B, T, N, H]
        dXv[B, T, D] = Wv[D, N, H] @{N, H} dV[B, T, N, H]
        """
        x, q, k, v, wq, wk, wv, wo, scores = ctx.saved_tensors

        # 1. Backprop through Wo
        # d_attention / d_wo
        qkv = torch.einsum('bsnt,btnh->bsnh', scores, v)
        d_wo = torch.einsum('bsnh,bsd->dnh', qkv, grad_out)

        # 2. Backprop through Wv
        d_qkv = torch.einsum('bsd,dnh->bsnh', grad_out, wo)
        d_v = torch.einsum('bsnt,bsnh->btnh', scores, d_qkv)
        d_wv = torch.einsum('btd,btnh->dnh', x, d_v)

        # 3. Backprop through softmax
        d_scores = torch.einsum('bsnh,btnh->bsnt', d_qkv, v)
        d_qk = d_softmax(d_scores, scores, dim=-1)
        # d_qk = scores * (d_scores - (d_scores * scores).sum(dim=-1, keepdims=True))
        d_qk /= q.shape[-1]**0.5

        # 4. Backprop through masking
        # We don't need to explicitly "remove" the mask.
        # Gradients at masked positions are effectively killed because 'scores' is 0 there,
        # so d_qk_scaled becomes 0.

        # 5. Backprop through Wq
        d_q = torch.einsum('bsnt,btnh->bsnh', d_qk, k)
        d_wq = torch.einsum('bsd,bsnh->dnh', x, d_q)

        # 6. Backprop through Wk
        d_k = torch.einsum('bsnh,bsnt->btnh', q, d_qk)
        d_wk = torch.einsum('btd,btnh->dnh', x, d_k)

        # 7. Backprop through X
        d_xq = torch.einsum('dnh,bsnh->bsd', wq, d_q)
        d_xk = torch.einsum('dnh,btnh->btd', wk, d_k)
        d_xv = torch.einsum('dnh,btnh->btd', wv, d_v)
        d_x = d_xq + d_xk + d_xv

        # forward was given x, wq, wk, wv, wo, masking
        return d_x, d_wq, d_wk, d_wv, d_wo, None
    
class Attention(torch.nn.Module):
    def __init__(self, d_model, n_heads, head_dim, masking=True):
        super().__init__()
        self.masking=masking

        # init the weights for the optimizer
        self.wq = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wk = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wv = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))
        self.wo = torch.nn.Parameter(torch.empty(d_model, n_heads, head_dim))

        # xavier noise to keep the variance small, for controlled gradients
        torch.nn.init.xavier_normal_(self.wq)
        torch.nn.init.xavier_normal_(self.wk)
        torch.nn.init.xavier_normal_(self.wv)
        torch.nn.init.xavier_normal_(self.wo)

    def load_checkpoint(self, params):
        # we change the memory, we don't rebuild the graph
        with torch.no_grad():
            self.wq.copy_(params['wq'])
            self.wk.copy_(params['wk'])
            self.wv.copy_(params['wv'])
            self.wo.copy_(params['wo'])
        
    def forward(self, x):
        # We pass our weights into the static function
        return AttentionFn.apply(x, self.wq, self.wk, self.wv, self.wo, self.masking)


if __name__ == "__main__":
    D, N = 64, 8
    H = D//N
    B, S = 2, 10 # batch, tokens per batch

    torch.manual_seed(42)
    x = torch.randn(B, S, D, dtype=torch.float32, requires_grad=True) # bf16 usually for mixed-precision
    params = {
        'wq': torch.randn(D, N, H, dtype=torch.float32),
        'wk': torch.randn(D, N, H, dtype=torch.float32),
        'wv': torch.randn(D, N, H, dtype=torch.float32),
        'wo': torch.randn(D, N, H, dtype=torch.float32),
    }

    model = Attention(D, N, H)
    model.load_checkpoint(params)

    out = model.forward(x)

    # Create a dummy loss to trigger backprop
    loss = out.sum()

    # Compute gradients
    loss.backward()

    print(f"Gradient Q shape: {model.wq.grad.shape}")
    print(f"Gradient K shape: {model.wk.grad.shape}")
    print(f"Gradient V shape: {model.wv.grad.shape}")
    print(f"Gradient Out shape: {model.wo.grad.shape}")
    print(f"Gradient X shape: {x.grad.shape}")
