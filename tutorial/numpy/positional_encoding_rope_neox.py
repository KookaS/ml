import numpy as np

def pe_rope_halved(x, keep):
    """
    RoPE using only the first 25% of embedding dimension.
    RoPE using Halved data, GPT-NeoX / PaLM style (computationally efficient).
    Pairs x[i] with x[i + dim/2].

    Refence: https://github.com/huggingface/transformers/blob/37974267efefe020168ff27081fbab8bbce04720/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L126
    
    :param x: Input [B, S, N, H]
    :param keep: kept embedding frequencies
    :return: PE [B, S, N, H]
    """

    B, S, N, H = x.shape

    # theta and pos is similar to sinusoidal positional encoder
    theta = 1.0 / 10000.0 ** (np.arange(0, keep, 2) / keep)
    pos = np.arange(S)
    freqs = pos[:, None] * theta[None, :] # [S, K/2]
    # [S, K/2] -> [S, 1, K/2]
    sin_values = np.sin(freqs)[:, None, :]
    cos_values = np.cos(freqs)[:, None, :]

    # split in kept freqs and not
    x1, x2, x_pass = x[..., :keep//2], x[..., keep//2:keep], x[..., keep:]
    # RoPE on contiguous data (i & i+K/2)
    out1 = x1 * cos_values - x2 * sin_values
    out2 = x2 * cos_values + x1 * sin_values
    out = np.concatenate([out1, out2], axis=-1)

    # add the skipped frequencies
    out = np.concatenate([out, x_pass], axis=-1)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Set the backend to non-interactive (headless)
    import matplotlib.pyplot as plt

    # 1. Setup Data
    B, S, N, H = 1, 50, 2, 64
    keep=H//4
    q = np.ones((B, S, N, H))

    # 2. Apply RoPE (usually done on Q and K)
    q_rotated = pe_rope_halved(q, keep) # [B, S, N, H]

    # 3. Plotting setup
    # Use 'plasma_r' for Yellow (start) -> Blue/Purple (end)
    cmap_time = 'plasma_r' 
    time_indices = np.arange(S)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Plot 1: High Frequency (Dims 0 & 1) ---
    ax1 = axes[0]
    target_dim = 0
    target_dim_A, target_dim_B = target_dim, target_dim + keep//2
    dim0 = q_rotated[0, :, 0, target_dim_A]
    dim1 = q_rotated[0, :, 0, target_dim_B]
    
    # Faint gray line to show connectivity
    ax1.plot(dim0, dim1, '-', color='gray', alpha=0.3, zorder=1)
    # Colored scatter points indicating time
    sc1 = ax1.scatter(dim0, dim1, c=time_indices, cmap=cmap_time, s=40, zorder=2, edgecolor='k', linewidth=0.5)
    
    # Mark Start clearly
    ax1.plot(dim0[0], dim1[0], 'rX', markersize=12, markeredgewidth=3, label='Start (t=0)', zorder=3)
    
    ax1.set_title(f"Dims {target_dim_A} & {target_dim_B} (High Frequency)\nRotates fast: Yellow $\\to$ Blue")
    ax1.set_xlabel(f"Dimension {target_dim_A} Value")
    ax1.set_ylabel(f"Dimension {target_dim_B} Value")
    ax1.grid(True)
    ax1.axis('equal')

    # --- Plot 2: Low Frequency (Dims 32 & 33) ---
    ax2 = axes[1]
    # Choose a pair from the middle frequency range
    target_dim = keep//4
    target_dim_A, target_dim_B = target_dim, target_dim + keep//2
    dimA = q_rotated[0, :, 0, target_dim_A]
    dimB = q_rotated[0, :, 0, target_dim_B]

    # Faint gray line
    ax2.plot(dimA, dimB, '-', color='gray', alpha=0.3, zorder=1)
    # Colored scatter points
    sc2 = ax2.scatter(dimA, dimB, c=time_indices, cmap=cmap_time, s=40, zorder=2, edgecolor='k', linewidth=0.5)
    
    # Mark Start
    ax2.plot(dimA[0], dimB[0], 'rX', markersize=12, markeredgewidth=3, label='Start (t=0)', zorder=3)
    
    ax2.set_title(f"Dims {target_dim_A} & {target_dim_B} (Medium Frequency)\nRotates slowly: Yellow $\\to$ Blue")
    ax2.set_xlabel(f"Dimension {target_dim_A} Value")
    ax2.set_ylabel(f"Dimension {target_dim_B} Value")
    ax2.grid(True)
    ax2.axis('equal')

    # --- Add Colorbar ---
    # Add a single colorbar at the bottom shared by both plots
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03]) # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Sequence Position (Time Step t: 0 to {S-1})', fontsize=12)
    cbar.set_ticks([0, S//2, S-1])
    cbar.set_ticklabels(['t=0 (Start)', f't={S//2}', f't={S-1} (End)'])

    # Adjust layout to make room for colorbar
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig("tmp/positional_encoding_rope_neox.png")