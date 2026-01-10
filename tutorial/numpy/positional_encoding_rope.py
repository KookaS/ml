import numpy as np

def pe_rope(x):
    """
    RoPE matrix rotating positional encoding using Interleaved data (Original paper).
    Pairs x[i] with x[i+1].
    
    :param x: Input [B, S, N, H]
    :return: PE [B, S, N, H]
    """

    B, S, N, H = x.shape

    # theta and pos is similar to sinusoidal positional encoder
    theta = 1.0 / 10000.0 ** (np.arange(0, H, 2) / H)
    pos = np.arange(S)
    freqs = pos[:, None] * theta[None, :] # [S, H/2]
    # [S, 1, H/2]
    sin_values = np.sin(freqs)[:, None, :]
    cos_values = np.cos(freqs)[:, None, :]

    # for RoPE we apply a rotation on pairs of x dimensions
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.zeros_like(x)
    # apply rotation of matrix (x * cos (pos * theta) + rotation(x) * sin(pos * theta))
    # [B, S, N, H/2] * [S, 1, H/2] -> [B, S, N, H/2]
    out[..., 0::2] = x1 * cos_values - x2 * sin_values
    out[..., 1::2] = x2 * cos_values + x1 * sin_values

    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Set the backend to non-interactive (headless)
    import matplotlib.pyplot as plt

    # 1. Setup Data
    # B=1, S=50 (Time steps), H=64 (Dimensions)
    B, S, N, H = 1, 50, 2, 64
    q = np.ones((B, S, N, H))

    # 2. Apply RoPE (usually done on Q and K)
    q_rotated = pe_rope(q) # [B, S, N, H]

    # 3. Plotting setup
    # Use 'plasma_r' for Yellow (start) -> Blue/Purple (end)
    cmap_time = 'plasma_r' 
    time_indices = np.arange(S)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Plot 1: High Frequency (Dims 0 & 1) ---
    ax1 = axes[0]
    target_dim = 0 # Highest frequencies
    target_dim_A, target_dim_B = target_dim, target_dim+1
    dimA = q_rotated[0, :, 0, target_dim_A]
    dimB = q_rotated[0, :, 0, target_dim_B]
    
    # Faint gray line to show connectivity
    ax1.plot(dimA, dimB, '-', color='gray', alpha=0.3, zorder=1)
    # Colored scatter points indicating time
    sc1 = ax1.scatter(dimA, dimB, c=time_indices, cmap=cmap_time, s=40, zorder=2, edgecolor='k', linewidth=0.5)
    
    # Mark Start clearly
    ax1.plot(dimA[0], dimB[0], 'rX', markersize=12, markeredgewidth=3, label='Start (t=0)', zorder=3)
    
    ax1.set_title(f"Dims {target_dim_A} & {target_dim_B} (High Frequency)\nRotates fast: Yellow $\\to$ Blue")
    ax1.set_xlabel(f"Dimension {target_dim_A} Value")
    ax1.set_ylabel(f"Dimension {target_dim_B} Value")
    ax1.grid(True)
    ax1.axis('equal')

    # --- Plot 2: Low Frequency (Dims 32 & 33) ---
    ax2 = axes[1]
    target_dim = 32 # Choose a pair from the middle frequency range
    target_dim_A, target_dim_B = target_dim, target_dim+1
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
    
    plt.savefig("tmp/positional_encoding_rope.png")