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
    
def flash_attention_v1_forward(x, wq, wk, wv, wo, block_size=64):
    """
    Flash Attention v1: outer loop over K,V blocks, inner loop over Q blocks.

    This loop order means:
    - K,V tiles are loaded once (in outer loop)
    - Q tiles are loaded Tc times (once per K,V block)
    - O is read and written Tc times (accumulate in HBM)

    HBM accesses: 2*S*d + 3*S²*d/Br (K,V once + Q,O_read,O_write Tc times)
    """
    q = x @ wq.t()
    k = x @ wk.t()
    v = x @ wv.t()

    S, d = q.shape
    scale = 1 / d**(0.5)

    # Global state stored in HBM (read/written each inner iteration)
    o = torch.zeros_like(q)
    m = torch.full((S, 1), -torch.inf, device=x.device)
    l = torch.zeros((S, 1), device=x.device)

    # Outer loop: K, V blocks (loaded once each)
    for j in range(0, S, block_size):
        j_end = min(j + block_size, S)
        k_tile = k[j:j_end, :]  # [Bc, d] - loaded from HBM once
        v_tile = v[j:j_end, :]  # [Bc, d] - loaded from HBM once

        # Inner loop: Q blocks (loaded Tc times total)
        for i in range(0, S, block_size):
            i_end = min(i + block_size, S)

            # Load from HBM (happens every iteration - this is the inefficiency!)
            q_tile = q[i:i_end, :]           # [Br, d]
            o_tile = o[i:i_end, :].clone()   # [Br, d] - must read previous value
            m_i = m[i:i_end, :].clone()      # [Br, 1]
            l_i = l[i:i_end, :].clone()      # [Br, 1]

            # Compute attention block
            S_ij = torch.einsum('rd,cd->rc', q_tile, k_tile) * scale  # [Br, Bc]

            # Update block state
            m_ij = torch.max(S_ij, axis=-1, keepdim=True).values
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = torch.sum(P_ij, axis=-1, keepdim=True)

            # Update global state with rescaling
            m_new = torch.max(m_i, m_ij)
            old_scale = torch.exp(m_i - m_new)
            new_scale = torch.exp(m_ij - m_new)

            l_new = (old_scale * l_i) + (new_scale * l_ij)
            o_new = (old_scale * o_tile) + (new_scale * torch.einsum('rc,cd->rd', P_ij, v_tile))

            # Write back to HBM (happens every iteration!)
            o[i:i_end, :] = o_new
            m[i:i_end, :] = m_new
            l[i:i_end, :] = l_new

    # Final normalization
    o = o / l
    return o @ wo.t()


def flash_attention_v2_forward(x, wq, wk, wv, wo, block_size=64):
    """
    Flash Attention v2: outer loop over Q blocks, inner loop over K,V blocks.

    This loop order means:
    - Q tiles are loaded once (in outer loop)
    - K,V tiles are loaded Tr times (once per Q block)
    - O is written once (accumulate in SRAM, write at end)
    - O is NEVER read back from HBM (key savings!)

    HBM accesses: 2*S*d + 2*S²*d/Br (Q,O once + K,V Tr times)
    Savings vs v1: S²*d/Br bytes (eliminates O read-back)
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
    out_std = standard_attention_forward(x, wq, wk, wv, wo)
    out_v1 = flash_attention_v1_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)
    out_v2 = flash_attention_v2_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)

    # Check FA v1 vs Standard
    diff_v1 = (out_v1 - out_std).abs().max().item()
    rel_diff_v1 = diff_v1 / (out_std.abs().max().item() + 1e-8)
    if rel_diff_v1 > 1e-4:
        print(f"❌ FA v1 MISMATCH: Max Diff = {diff_v1}, Relative = {rel_diff_v1:.2e}")
        return
    else:
        print(f"✅ FA v1 matches Standard (Relative diff: {rel_diff_v1:.2e})")

    # Check FA v2 vs Standard
    diff_v2 = (out_v2 - out_std).abs().max().item()
    rel_diff_v2 = diff_v2 / (out_std.abs().max().item() + 1e-8)
    if rel_diff_v2 > 1e-4:
        print(f"❌ FA v2 MISMATCH: Max Diff = {diff_v2}, Relative = {rel_diff_v2:.2e}")
        return
    else:
        print(f"✅ FA v2 matches Standard (Relative diff: {rel_diff_v2:.2e})")

    # Check FA v1 vs FA v2 (should be identical)
    diff_v1v2 = (out_v1 - out_v2).abs().max().item()
    rel_diff_v1v2 = diff_v1v2 / (out_v2.abs().max().item() + 1e-8)
    print(f"✅ FA v1 vs FA v2 (Relative diff: {rel_diff_v1v2:.2e})")

    # Backward Check (using v2)
    loss_v2 = out_v2.sum()
    loss_v2.backward()
    grad_v2 = x.grad.clone()
    x.grad = None

    loss_std = out_std.sum()
    loss_std.backward()
    grad_std = x.grad.clone()

    grad_diff = (grad_v2 - grad_std).abs().max().item()
    grad_rel_diff = grad_diff / (grad_std.abs().max().item() + 1e-8)
    if grad_rel_diff > 5e-4:
        print(f"❌ GRAD MISMATCH: Max Diff = {grad_diff}, Relative = {grad_rel_diff:.2e}")
    else:
        print(f"✅ Gradients match (Relative diff: {grad_rel_diff:.2e})")

    print("-" * 50)

    # --- Speed Benchmark ---
    iterations = 50

    # 1. Benchmark Standard
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = standard_attention_forward(x, wq, wk, wv, wo)
    if device == "cuda":
        torch.cuda.synchronize()
    std_time = (time.time() - start) / iterations

    # 2. Benchmark FA v1
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = flash_attention_v1_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)
    if device == "cuda":
        torch.cuda.synchronize()
    v1_time = (time.time() - start) / iterations

    # 3. Benchmark FA v2
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = flash_attention_v2_forward(x, wq, wk, wv, wo, block_size=BLOCK_SIZE)
    if device == "cuda":
        torch.cuda.synchronize()
    v2_time = (time.time() - start) / iterations

    print(f"\nStandard Attention: {std_time*1000:.2f} ms")
    print(f"Flash Attention v1: {v1_time*1000:.2f} ms")
    print(f"Flash Attention v2: {v2_time*1000:.2f} ms")
    print(f"v1/v2 ratio:        {v1_time/v2_time:.2f}x")

    print("\nNOTE: These Python implementations will be SLOWER than Standard Attention.")
    print("The v1/v2 ratio does NOT reflect the HBM savings because:")
    print("  1. PyTorch abstracts memory - no real SRAM vs HBM distinction")
    print("  2. Python loop overhead dominates actual compute")
    print("  3. Each loop iteration launches separate CUDA kernels")
    print("To see real speedups, this must be written in Triton or CUDA.")

if __name__ == "__main__":
    benchmark()