import torch
import torch.nn as nn
import math
import time

def standard_attention_forward(x, wq, wk, wv, wo):
    S, d = x.shape
    q = x @ wq.t()
    k = x @ wk.t()
    v = x @ wv.t()
    
    scores = (q @ k.t()) * (1 / math.sqrt(d))
    attn = torch.softmax(scores, dim=-1)
    o = attn @ v
    return o @ wo.t()
    
def flash_attention_forward(x, wq, wk, wv, wo, block_size=64):
    """
    This is the implementation of the flash attention in unpadded mode, meaning batch size is untouched.
    For sake of simplicity we only tile along the sequence dimension. This is suboptimal because we will process padding tokens to match all sequences to the same size.
    We tile along the sequence dimension.
    Note: this should be done as a kernel, not in pytorch usually.

    S sequence_length
    d dimension of the model
    """

    # compute q, k, v and store in HBM
    q = x @ wq.t()
    k = x @ wk.t()
    v = x @ wv.t()

    S, d = q.shape
    scale = 1 / d**(0.5)
    o = torch.zeros_like(q)

    # divide q into blocks
    for i in range(0, S, block_size):
        # load from HBM blocks and keep in SRAM
        i_end = min(i + block_size, S)
        q_tile = q[i:i_end, :] # [Br, d]

        # initial state
        m = torch.full((i_end-i, 1), -torch.inf, device=x.device)
        l = torch.zeros((i_end-i, 1), device=x.device)
        o_tile = torch.zeros((i_end-i, d), device=x.device) # [Br, d]

        # divide k, v into blocks
        for j in range(0, S, block_size):
            j_end = min(j + block_size, S)
            k_tile = k[j:j_end, :] # [Bc, d]
            v_tile = v[j:j_end, :] # [Bc, d]

            # attention block
            S_ij = torch.einsum('rd,cd->rc', q_tile, k_tile) * scale # [Br, Bc]

            # update block state
            m_ij = torch.max(S_ij, axis=-1, keepdim=True).values
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = torch.sum(P_ij, axis=-1, keepdim=True)

            # update global state
            m_new = torch.max(m, m_ij)
            old_scale = torch.exp(m - m_new) # shrink previous accumulated values
            new_scale = torch.exp(m_ij - m_new) # shrink the new values

            l = (old_scale * l) + (new_scale * l_ij)
            o_tile = (old_scale * o_tile) + (new_scale * torch.einsum('rc,cd->rd', P_ij, v_tile))
            m = m_new

        o[i:i_end, :] = o_tile / l

    return o @ wo.t()


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    # Configuration
    B = 1 # Flash operates on flattened S usually, so B=1, S=Large
    S = 2048 # Sequence Length
    D = 128 # Head Dimension
    BLOCK_SIZE = 128

    torch.manual_seed(42)
    x = torch.randn(S, D, device=device, requires_grad=True)
    wq = torch.randn(D, D, device=device, requires_grad=True)
    wk = torch.randn(D, D, device=device, requires_grad=True)
    wv = torch.randn(D, D, device=device, requires_grad=True)
    wo = torch.randn(D, D, device=device, requires_grad=True)

    print(f"\nConfig: Seq_Len={S}, D_model={D}, Block_Size={BLOCK_SIZE}")
    print("-" * 50)

    # --- Correctness Check ---
    print("Checking Correctness...")
    out_flash = flash_attention_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)
    out_std = standard_attention_forward(x, wq, wk, wv, wo)
    
    diff = (out_flash - out_std).abs().max().item()
    # Use relative tolerance since output magnitude depends on weight initialization
    rel_diff = diff / (out_std.abs().max().item() + 1e-8)
    if rel_diff > 1e-4:
        print(f"❌ OUTPUT MISMATCH: Max Diff = {diff}, Relative = {rel_diff:.2e}")
        return
    else:
        print(f"✅ Outputs match (Max Diff: {diff:.2e}, Relative: {rel_diff:.2e})")

    # Backward Check
    loss_flash = out_flash.sum()
    loss_flash.backward()
    grad_flash = x.grad.clone()
    x.grad = None

    loss_std = out_std.sum()
    loss_std.backward()
    grad_std = x.grad.clone()
    
    grad_diff = (grad_flash - grad_std).abs().max().item()
    grad_rel_diff = grad_diff / (grad_std.abs().max().item() + 1e-8)
    # Gradients accumulate more numerical error through autograd, use looser tolerance
    if grad_rel_diff > 5e-4:
        print(f"❌ GRAD MISMATCH: Max Diff = {grad_diff}, Relative = {grad_rel_diff:.2e}")
    else:
        print(f"✅ Gradients match (Max Diff: {grad_diff:.2e}, Relative: {grad_rel_diff:.2e})")

    print("-" * 50)

    # --- Speed Benchmark ---
    iterations = 50
    
    # 1. Benchmark Standard
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = standard_attention_forward(x, wq, wk, wv, wo)
    torch.cuda.synchronize()
    std_time = (time.time() - start) / iterations
    
    # 2. Benchmark Python Flash
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = flash_attention_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)
    torch.cuda.synchronize()
    flash_time = (time.time() - start) / iterations

    print(f"Standard Attention Time: {std_time*1000:.2f} ms")
    print(f"Python Flash Attn Time:  {flash_time*1000:.2f} ms")
    print(f"Ratio (Std / Flash):     {std_time/flash_time:.2f}x")
    
    print("\nNOTE: This Python implementation will be SLOWER than Standard Attention.")
    print("Why? Because PyTorch Standard Attention calls a single fused C++ kernel.")
    print("Your loop is launching O((S/Block)^2) separate kernels from Python.")
    print("To see the speedup, this logic MUST be written in Triton or CUDA.")

if __name__ == "__main__":
    benchmark()