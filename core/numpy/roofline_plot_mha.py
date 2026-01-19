import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Hardware parameters (H100)
h100_bandwidth = 3.35e12   # HBM bandwidth bytes/s
h100_peak_flops = 9.89e14  # bfloat16 FLOPs/s

# Critical intensity (where roofline transitions from memory to compute bound)
critical_intensity = h100_peak_flops / h100_bandwidth

# Attention parameters
B = 1  # batch size
N, H = 32, 256
D = N * H  # 8192
bytes_per_param = 2  # bf16

def compute_mha_intensity(B, S, N, H, bytes_per_param=2):
    """
    Compute arithmetic intensity for Attention

    For multi-head attention, intensity grows with sequence length S because:
    - Compute is O(S²) for QK and score@V matmuls
    - Communication is O(S²) for score matrices but O(S) for Q,K,V

    At small S: linear terms dominate -> low intensity (memory bound)
    At large S: quadratic terms dominate -> intensity approaches H/2 ≈ 128
    """
    D = N * H

    # 1. compute Q,K,V: X[B,S,D] @ Wq,Wk,Wv[D,N,H] -> Q,K,V[B,S,N,H]
    comm_qkv = bytes_per_param * (B*S*D + 3*D*N*H + 3*B*S*N*H)
    compute_qkv = 3 * (2*B*S*D*N*H)

    # 2. compute QK: Q[B,N,S,H] @ K[B,N,H,S] -> scores[B,N,S,S]
    comm_qk = bytes_per_param * (2*B*S*N*H + B*N*S*S)
    compute_qk = 2*B*N*S*S*H

    # 3. softmax: 5 ops per element (sub max, exp, sum, div, scale)
    comm_softmax = bytes_per_param * (2*B*N*S*S)
    compute_softmax = 5*B*N*S*S

    # 4. score @ V: scores[B,N,S,S] @ V[B,N,S,H] -> attn[B,N,S,H]
    comm_attention = bytes_per_param * (B*N*S*S + B*S*N*H + B*S*N*H)
    compute_attention = 2*B*N*S*S*H

    # 5. output projection: attn[B,S,NH] @ Wo[NH,D] -> out[B,S,D]
    comm_out = bytes_per_param * (B*S*N*H + D*N*H + B*S*D)
    compute_out = 2*B*S*N*H*D

    comm_total = comm_qkv + comm_qk + comm_softmax + comm_attention + comm_out
    compute_total = compute_qkv + compute_qk + compute_softmax + compute_attention + compute_out
    return compute_total / comm_total

def roofline_perf(intensity, bandwidth, peak_flops):
    """Achievable performance given arithmetic intensity"""
    return np.minimum(intensity * bandwidth, peak_flops)

# Vary sequence length S (this affects attention intensity due to O(S²) terms)
seq_lengths = np.logspace(1, 5, 500)  # 10 to 100K tokens
intensities_algo = np.array([compute_mha_intensity(B, S, N, H, bytes_per_param) for S in seq_lengths])
performances = roofline_perf(intensities_algo, h100_bandwidth, h100_peak_flops)

# Find optimal sequence length (closest to critical intensity)
optimal_idx = np.argmin(np.abs(intensities_algo - critical_intensity))
optimal_seq = seq_lengths[optimal_idx]
optimal_intensity = intensities_algo[optimal_idx]
optimal_perf = performances[optimal_idx]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Custom colormap: blue -> red -> green
colors_list = ['blue', 'red', 'green']
cmap = LinearSegmentedColormap.from_list('batch_size', colors_list)

# Create color gradient based on position relative to critical intensity
def get_color_value(intensity):
    """Map intensity to color value: 0=blue, 0.5=red, 1=green"""
    log_int = np.log10(intensity)
    log_crit = np.log10(critical_intensity)
    log_min = np.log10(intensities_algo.min())
    log_max = np.log10(intensities_algo.max())

    if log_int <= log_crit:
        return 0.5 * (log_int - log_min) / (log_crit - log_min)
    else:
        return 0.5 + 0.5 * (log_int - log_crit) / (log_max - log_crit)

# Plot gradient line using short segments
for i in range(len(intensities_algo) - 1):
    color_val = get_color_value(intensities_algo[i])
    ax.plot(intensities_algo[i:i+2], performances[i:i+2],
            color=cmap(color_val), linewidth=4, solid_capstyle='round')

ax.set_xscale('log')
ax.set_yscale('log')

# Mark optimal point
ax.plot(optimal_intensity, optimal_perf, 'o', color='red', markersize=14,
        markeredgecolor='white', markeredgewidth=2, zorder=5)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Short seq\n(memory bound)', 'Optimal', 'Long seq\n(compute bound)'])

# Annotate optimal point
ax.annotate(f'Optimal\nS={optimal_seq:.0f}\nI={optimal_intensity:.0f}',
            xy=(optimal_intensity, optimal_perf),
            xytext=(optimal_intensity*0.12, optimal_perf*0.35),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

# Labels and formatting
ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
ax.set_ylabel('Performance (FLOPs/s)', fontsize=12)
ax.set_title(f'Roofline Model - H100 GPU (BF16)\nMHA: B={B}, N={N}, H={H} (varying S)', fontsize=14)
ax.set_xlim(0.1, 10000)
ax.set_ylim(1e11, 2e15)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('image/roofline/roofline_plot_mha.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Roofline Analysis (H100 BF16):")
print(f"  Peak FLOPs/s: {h100_peak_flops/1e12:.0f} TFLOPs/s")
print(f"  HBM Bandwidth: {h100_bandwidth/1e12:.2f} TB/s")
print(f"  Critical Intensity: {critical_intensity:.1f} FLOPs/byte")
print(f"\nOptimal Sequence Length: {optimal_seq:.0f} tokens")
print(f"  Intensity: {optimal_intensity:.1f} FLOPs/byte")
print(f"  Performance: {optimal_perf/1e12:.1f} TFLOPs/s")
