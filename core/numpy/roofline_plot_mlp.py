import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Hardware parameters (H100)
h100_bandwidth = 3.35e12   # HBM bandwidth bytes/s
h100_peak_flops = 9.89e14  # bfloat16 FLOPs/s

# Critical intensity (where roofline transitions from memory to compute bound)
critical_intensity = h100_peak_flops / h100_bandwidth

# MLP parameters
N, H = 32, 256
D = N * H  # 8192
F = 4 * D  # 32768
bytes_per_param = 2  # bf16
# batch_tokens for MLP is B * S

def compute_mlp_intensity(batch_tokens, D, F, bytes_per_param=2):
    """Compute arithmetic intensity for MLP: X[B,D] @ W1[D,F] @ W2[F,D]"""
    communication = bytes_per_param * (batch_tokens*D + D*F + F*D + batch_tokens*D)
    compute = 2 * batch_tokens * D * F + 2 * batch_tokens * F * D
    return compute / communication

def roofline_perf(intensity, bandwidth, peak_flops):
    """Achievable performance given arithmetic intensity"""
    return np.minimum(intensity * bandwidth, peak_flops)

# Create smooth batch size range for gradient
batch_sizes = np.logspace(0, 7, 500)  # 1 to 10M tokens
intensities_mlp = np.array([compute_mlp_intensity(bs, D, F, bytes_per_param) for bs in batch_sizes])
performances = roofline_perf(intensities_mlp, h100_bandwidth, h100_peak_flops)

# Find optimal batch size (closest to critical intensity)
optimal_idx = np.argmin(np.abs(intensities_mlp - critical_intensity))
optimal_batch = batch_sizes[optimal_idx]
optimal_intensity = intensities_mlp[optimal_idx]
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
    log_min = np.log10(intensities_mlp.min())
    log_max = np.log10(intensities_mlp.max())

    if log_int <= log_crit:
        return 0.5 * (log_int - log_min) / (log_crit - log_min)
    else:
        return 0.5 + 0.5 * (log_int - log_crit) / (log_max - log_crit)

# Plot gradient line using short segments
for i in range(len(intensities_mlp) - 1):
    color_val = get_color_value(intensities_mlp[i])
    ax.plot(intensities_mlp[i:i+2], performances[i:i+2],
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
cbar.set_ticklabels(['Low batch\n(memory bound)', 'Optimal', 'High batch\n(compute bound)'])

# Annotate optimal point
ax.annotate(f'Optimal\nB={optimal_batch:.0f}\nI={optimal_intensity:.0f}',
            xy=(optimal_intensity, optimal_perf),
            xytext=(optimal_intensity*0.12, optimal_perf*0.35),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

# Labels and formatting
ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
ax.set_ylabel('Performance (FLOPs/s)', fontsize=12)
ax.set_title(f'Roofline Model - H100 GPU (BF16)\nMLP: D={D}, F={F}', fontsize=14)
ax.set_xlim(0.1, 10000)
ax.set_ylim(1e11, 2e15)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('image/roofline/roofline_plot_mlp.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Roofline Analysis (H100 BF16):")
print(f"  Peak FLOPs/s: {h100_peak_flops/1e12:.0f} TFLOPs/s")
print(f"  HBM Bandwidth: {h100_bandwidth/1e12:.2f} TB/s")
print(f"  Critical Intensity: {critical_intensity:.1f} FLOPs/byte")
print(f"\nOptimal Batch Size: {optimal_batch:,} tokens")
print(f"  Intensity: {optimal_intensity:.1f} FLOPs/byte")
print(f"  Performance: {optimal_perf/1e12:.1f} TFLOPs/s")
