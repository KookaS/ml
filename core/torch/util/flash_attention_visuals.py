import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['grid.alpha'] = 0.5


def plot_tile_size_vs_latency():
    """
    Shows the trade-off between tile size and latency.
    Small tiles: kernel overhead dominates
    Large tiles: SRAM spill to HBM dominates
    Consistent with other plots (d=64, 164KB SRAM).
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))

        # === Parameters consistent with other plots ===
        d = 64
        SRAM_SIZE_KB = 164

        # === Tile sizes to evaluate ===
        tile_sizes = np.array([8, 16, 32, 64, 128, 256, 512, 1024])

        # === Calculate SRAM usage for each tile size ===
        def sram_required_kb(Br, d):
            elements = 4 * Br * d + 2 * Br
            return elements * 2 / 1024  # FP16, KB

        sram_usage = np.array([sram_required_kb(Br, d) for Br in tile_sizes])
        exceeds_sram = sram_usage > SRAM_SIZE_KB

        # === Latency model ===
        # 1. Kernel overhead: decreases with larger tiles (fewer kernel launches)
        #    For sequence S, need S/Br kernel launches per Q block
        kernel_overhead = 100 / tile_sizes  # relative overhead

        # 2. SRAM spill cost: zero if fits, explodes if exceeds
        spill_cost = np.where(exceeds_sram,
                              5 * ((sram_usage - SRAM_SIZE_KB) / SRAM_SIZE_KB) ** 2,
                              0)

        # 3. Base compute: slightly increases with tile size (more work per tile)
        base_compute = 0.5 + tile_sizes * 0.001

        # Total latency (normalized)
        total_latency = kernel_overhead + spill_cost + base_compute
        total_latency = total_latency / total_latency.min()  # normalize to min=1

        # === Colors consistent with other plots ===
        colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653']

        # === Plot total latency ===
        ax.plot(tile_sizes, total_latency, 'o-', color=colors[0], linewidth=2.5,
                markersize=10, label='Total Latency', zorder=3)

        # === Plot components ===
        kernel_norm = kernel_overhead / total_latency.min()
        spill_norm = spill_cost / total_latency.min()

        ax.plot(tile_sizes, kernel_norm, '--', color=colors[1], linewidth=2,
                alpha=0.7, label='Kernel Overhead')
        ax.plot(tile_sizes, spill_norm, '--', color=colors[3], linewidth=2,
                alpha=0.7, label='SRAM Spill Cost')

        # === Calculate exact SRAM boundary ===
        # SRAM = (4 * Br * d + 2 * Br) * 2 / 1024 KB
        # Solve for Br when SRAM = SRAM_SIZE_KB:
        # SRAM_SIZE_KB * 1024 / 2 = (4 * d + 2) * Br
        # Br = SRAM_SIZE_KB * 512 / (4 * d + 2)
        sram_boundary_br = SRAM_SIZE_KB * 512 / (4 * d + 2)  # ≈ 325 for d=64

        # === Mark SRAM limit ===
        ax.axvline(x=sram_boundary_br, color='black', linestyle=':', linewidth=2, alpha=0.7)
        ax.annotate(f'SRAM Limit\n({SRAM_SIZE_KB}KB)\nBr≈{sram_boundary_br:.0f}',
                    xy=(sram_boundary_br, 20),
                    fontsize=9, ha='right', va='top')

        # === Mark optimal ===
        optimal_idx = np.argmin(total_latency)
        ax.scatter([tile_sizes[optimal_idx]], [total_latency[optimal_idx]],
                   color=colors[2], s=200, zorder=5, marker='*', edgecolor='black', linewidth=1.5,
                   label=f'Optimal (Br={tile_sizes[optimal_idx]})')

        # === Shade regions ===
        # Shade SRAM exceeded region (starts at actual boundary)
        ax.axvspan(sram_boundary_br, tile_sizes[-1] * 1.5, alpha=0.15, color='red',
                   label='Exceeds SRAM')

        # === Annotations ===
        ax.annotate('Kernel overhead\ndominates', xy=(16, kernel_norm[1]),
                    xytext=(12, kernel_norm[1] + 2),
                    fontsize=9, color=colors[1], ha='center',
                    arrowprops=dict(arrowstyle='->', color=colors[1], alpha=0.7))

        if any(exceeds_sram):
            spill_idx = np.argmax(exceeds_sram)
            ax.annotate('Spills to HBM\n(10x slower)', xy=(tile_sizes[spill_idx + 1], total_latency[spill_idx + 1]),
                        xytext=(tile_sizes[spill_idx + 1] * 1.3, total_latency[spill_idx + 1] * 0.7),
                        fontsize=9, color=colors[3], ha='left',
                        arrowprops=dict(arrowstyle='->', color=colors[3], alpha=0.7))

        # === Add SRAM usage as secondary info ===
        for i, (br, sram, lat) in enumerate(zip(tile_sizes, sram_usage, total_latency)):
            if br in [64, 128, 256, 512]:
                status = '✓' if sram <= SRAM_SIZE_KB else '✗'
                ax.annotate(f'{sram:.0f}KB {status}', xy=(br, lat),
                            xytext=(0, -20), textcoords='offset points',
                            fontsize=8, ha='center', color='gray')

        # === Labels ===
        ax.set_xlabel('Block Size (Br)', fontsize=12)
        ax.set_ylabel('Relative Latency (lower is better)', fontsize=12)
        ax.set_title(f'Flash Attention: Tile Size vs Latency Trade-off\nA100 GPU (FP16), d={d}, SRAM={SRAM_SIZE_KB}KB', fontsize=14)
        ax.set_xscale('log', base=2)
        ax.set_xlim(6, 1500)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/olivier/ml/image/flash_attention/tile_size_vs_latency.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: tile_size_vs_latency.png")


def plot_memory_scaling():
    """
    Shows SRAM usage for Flash Attention vs HBM usage for Standard Attention.
    The key insight: Flash uses SRAM (KB), Standard uses HBM (GB).
    """
    with plt.style.context('default'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # === Fixed parameters ===
        d = 64  # head dimension
        SRAM_SIZE_KB = 164  # A100 shared memory

        # === Multiple Br values to compare ===
        Br_values = [64, 128, 256, 512]
        colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653']

        # === Sequence lengths to sweep ===
        seq_lengths = np.logspace(np.log10(10), np.log10(100000), 200)

        # ============ LEFT PLOT: HBM Usage ============
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Standard Attention: stores full S×S attention matrix in HBM
        standard_hbm_gb = (seq_lengths ** 2 + 4 * seq_lengths * d) * 2 / (1024**3)
        ax1.plot(seq_lengths, standard_hbm_gb, color='gray', linewidth=3, linestyle='--',
                label='Standard O(S²)', alpha=0.8)

        # Flash Attention: only Q,K,V,O in HBM - same for all Br!
        flash_hbm_gb = 4 * seq_lengths * d * 2 / (1024**3)
        ax1.plot(seq_lengths, flash_hbm_gb, color='#2a9d8f', linewidth=3,
                label='Flash O(S·d) - all Br')

        # A100 HBM limit
        ax1.axhline(y=80, color='black', linestyle=':', alpha=0.5, linewidth=1)
        ax1.annotate('A100 HBM (80GB)', xy=(12, 85), fontsize=9, color='black', va='bottom')

        ax1.set_xlabel('Sequence Length (S)', fontsize=12)
        ax1.set_ylabel('HBM Usage (GB)', fontsize=12)
        ax1.set_title('HBM (Global Memory)\nStandard needs S², Flash needs S·d', fontsize=12)
        ax1.set_xlim(10, 100000)
        ax1.set_ylim(0.0001, 200)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, which='both', alpha=0.3)

        # ============ RIGHT PLOT: SRAM Usage ============
        # SRAM is independent of sequence length - it's per-tile

        def sram_required_kb(Br, d):
            """SRAM for tiles: Q, K, V, O blocks + statistics"""
            elements = 4 * Br * d + 2 * Br  # 4 tiles + m,l stats
            return elements * 2 / 1024  # FP16, KB

        br_range = np.array([16, 32, 64, 128, 256, 512])
        sram_usage = [sram_required_kb(br, d) for br in br_range]

        # Colors: green for OK, orange for borderline, red for exceeds
        bar_colors = []
        for usage in sram_usage:
            if usage < SRAM_SIZE_KB * 0.7:
                bar_colors.append('#2a9d8f')  # green - safe
            elif usage < SRAM_SIZE_KB:
                bar_colors.append('#f4a261')  # orange - borderline
            else:
                bar_colors.append('#e63946')  # red - exceeds

        bars = ax2.bar(range(len(br_range)), sram_usage, color=bar_colors,
                       edgecolor='black', linewidth=1)

        # Color bars based on whether they exceed SRAM
        for i, (br, usage) in enumerate(zip(br_range, sram_usage)):
            if usage > SRAM_SIZE_KB:
                bars[i].set_hatch('//')
                bars[i].set_alpha(0.5)

        # SRAM limit line
        ax2.axhline(y=SRAM_SIZE_KB, color='black', linestyle='--', linewidth=2)
        ax2.annotate(f'A100 SRAM ({SRAM_SIZE_KB}KB)', xy=(5.5, SRAM_SIZE_KB + 10),
                    fontsize=10, color='black', ha='right')

        ax2.set_xticks(range(len(br_range)))
        ax2.set_xticklabels([str(br) for br in br_range])
        ax2.set_xlabel('Block Size (Br = Bc)', fontsize=12)
        ax2.set_ylabel('SRAM Usage (KB)', fontsize=12)
        ax2.set_title('SRAM (On-chip Memory)\nTiles must fit - hatched = exceeds', fontsize=12)
        ax2.set_ylim(0, 300)
        ax2.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (br, usage) in enumerate(zip(br_range, sram_usage)):
            label = f'{usage:.0f}KB'
            if usage > SRAM_SIZE_KB:
                label += '\n(spills!)'
            ax2.annotate(label, xy=(i, usage + 5), ha='center', fontsize=9)

        plt.suptitle('Flash Attention Memory: HBM vs SRAM\nA100 GPU (FP16), d=64', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('/home/olivier/ml/image/flash_attention/memory_scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: memory_scaling.png")


def compute_attention_arithmetic_intensity(S, d, method='standard', Br=64):
    """
    Compute arithmetic intensity for attention.

    FLOPs (same for both):
        - Q @ K^T: 2*S*S*d
        - softmax: ~3*S*S
        - attn @ V: 2*S*S*d
        - Total: ~4*S²*d

    Bytes moved (float32 = 4 bytes):
        Standard: reads/writes the S×S attention matrix to HBM
            - Read Q,K,V: 3*S*d
            - Write QK^T: S²
            - Read for softmax, write P: 2*S²
            - Read P, V, write O: S² + S*d + S*d
            - Total: 5*S*d + 4*S²

        Flash: tiles in SRAM, only reads Q,K,V once, writes O once
            - Read Q: S*d
            - Read K: S*d * (S/Br) [reread for each Q block]
            - Read V: S*d * (S/Br)
            - Write O: S*d
            - Total: 2*S*d + 2*S²*d/Br
    """
    flops = 4 * S * S * d

    if method == 'standard':
        bytes_moved = (5 * S * d + 4 * S * S) * 4  # float32
    else:  # flash
        bytes_moved = (2 * S * d + 2 * S * S * d / Br) * 4  # float32

    return flops / bytes_moved


def compute_flash_attention_ai(S, d, Br, M_sram=None):
    """
    Compute arithmetic intensity for Flash Attention.

    From Flash Attention paper, HBM accesses = O(N²d²/M) where M is SRAM size.
    With block size Br ≈ sqrt(M/(4d)) for FP16:

    FLOPs: 4 * S² * d  (Q@K^T and Attn@V)

    Bytes (realistic model with re-reads):
    - Q read once: S * d
    - K,V read S/Br times each: 2 * S * d * (S/Br)
    - O written once: S * d
    - Total elements: 2*S*d + 2*S²*d/Br
    - In FP16 bytes: (2*S*d + 2*S²*d/Br) * 2

    For small S (S << Br): bytes ≈ 4*S*d, AI ≈ S
    For large S (S >> Br): bytes ≈ 4*S²*d/Br, AI ≈ Br
    """
    flops = 4 * S * S * d
    bytes_elements = 2 * S * d + 2 * S * S * d / Br
    bytes_moved = bytes_elements * 2  # FP16
    return flops / bytes_moved


def plot_arithmetic_intensity():
    """
    Roofline plot for Flash Attention with varying S.
    Shows gradient trajectory on the roofline.
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))

        # === Hardware specs (A100 SXM4) ===
        mem_bandwidth = 2039e9  # bytes/s HBM2e
        peak_flops = 312e12  # 312 TFLOPS FP16
        critical_intensity = peak_flops / mem_bandwidth

        # === Fixed parameters ===
        d = 64   # head dimension
        Br = 256  # block size

        # === Sequence lengths to sweep ===
        seq_lengths = np.logspace(np.log10(10), np.log10(100000), 500)

        # === Compute AI and performance for each S ===
        ai_values = np.array([compute_flash_attention_ai(S, d, Br) for S in seq_lengths])
        perf_values = np.minimum(peak_flops, ai_values * mem_bandwidth)

        # === Find optimal point (closest to critical intensity) ===
        optimal_idx = np.argmin(np.abs(ai_values - critical_intensity))
        optimal_S = seq_lengths[optimal_idx]
        optimal_ai = ai_values[optimal_idx]
        optimal_perf = perf_values[optimal_idx]

        # === Colormap: blue -> red -> green ===
        colors_list = ['blue', 'red', 'green']
        cmap = LinearSegmentedColormap.from_list('seq', colors_list)

        # === Color mapping based on position relative to critical intensity ===
        def get_color_value(intensity):
            log_int = np.log10(intensity)
            log_crit = np.log10(critical_intensity)
            log_min = np.log10(ai_values.min())
            log_max = np.log10(ai_values.max())
            if log_int <= log_crit:
                return 0.5 * (log_int - log_min) / (log_crit - log_min + 1e-10)
            else:
                return 0.5 + 0.5 * (log_int - log_crit) / (log_max - log_crit + 1e-10)

        # === Draw trajectory as colored segments ===
        for i in range(len(ai_values) - 1):
            color_val = get_color_value(ai_values[i])
            ax.plot(ai_values[i:i+2], perf_values[i:i+2],
                    color=cmap(color_val), linewidth=4, solid_capstyle='round')

        ax.set_xscale('log')
        ax.set_yscale('log')

        # === Mark optimal point ===
        ax.plot(optimal_ai, optimal_perf, 'o', color='red', markersize=14,
                markeredgecolor='white', markeredgewidth=2, zorder=5)

        # === Colorbar with sequence length range ===
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)

        S_min = int(seq_lengths.min())
        S_max = int(seq_lengths.max())
        S_opt = int(optimal_S)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'S={S_min}\n(memory bound)', f'S={S_opt}\n(optimal)', f'S={S_max}\n(compute bound)'])

        # === Annotate optimal point ===
        ax.annotate(f'Optimal\nS={optimal_S:.0f}',
                    xy=(optimal_ai, optimal_perf),
                    xytext=(optimal_ai * 0.12, optimal_perf * 0.35),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

        # === Labels ===
        ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
        ax.set_ylabel('Performance (FLOPs/s)', fontsize=12)
        ax.set_title(f'Roofline Model - A100 GPU (FP16)\nFlash Attention: d={d}, Br={Br} (varying S)', fontsize=14)

        ax.set_xlim(0.1, 10000)
        ax.set_ylim(1e11, 2e15)
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/olivier/ml/image/flash_attention/arithmetic_intensity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: arithmetic_intensity.png")


def plot_performance_vs_block_size():
    """
    Shows % of peak performance vs sequence length for different Br values.
    Includes SRAM constraint - tiles must fit in shared memory.
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 6))

        # === Hardware specs (A100 SXM4) ===
        mem_bandwidth = 2039e9  # bytes/s HBM2e
        peak_flops = 312e12  # 312 TFLOPS FP16
        SRAM_SIZE_KB = 164  # A100 shared memory per SM (configurable max)

        # === Fixed parameters ===
        d = 64  # head dimension

        # === Multiple Br values to compare ===
        Br_values = [64, 128, 256, 512]
        colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653']

        # === Sequence lengths to sweep ===
        seq_lengths = np.logspace(np.log10(10), np.log10(100000), 200)

        ax.set_xscale('log')

        # === SRAM requirement calculation ===
        def sram_required_kb(Br, d):
            """
            SRAM needed for Flash Attention tiles (assuming Bc = Br):
            - Q tile: Br × d
            - K tile: Br × d
            - V tile: Br × d
            - O accumulator: Br × d
            - Statistics (m, l): 2 × Br
            Total elements: 4 × Br × d + 2 × Br
            In FP16 bytes: (4 × Br × d + 2 × Br) × 2
            """
            elements = 4 * Br * d + 2 * Br
            bytes_needed = elements * 2  # FP16
            return bytes_needed / 1024  # KB

        # === Standard Attention (for comparison) ===
        def compute_standard_attention_ai(S, d):
            """Standard attention AI: must materialize S×S matrix in HBM"""
            flops = 4 * S * S * d
            bytes_moved = (5 * S * d + 4 * S * S) * 2  # FP16
            return flops / bytes_moved

        std_ai_values = np.array([compute_standard_attention_ai(S, d) for S in seq_lengths])
        std_perf_values = np.minimum(peak_flops, std_ai_values * mem_bandwidth)
        std_pct_peak = std_perf_values / peak_flops * 100

        ax.plot(seq_lengths, std_pct_peak, color='gray', linewidth=3, linestyle='--',
                label='Standard', alpha=0.8)

        # === Draw curve for each Br ===
        for Br, color in zip(Br_values, colors):
            sram_kb = sram_required_kb(Br, d)
            exceeds_sram = sram_kb > SRAM_SIZE_KB

            ai_values = np.array([compute_flash_attention_ai(S, d, Br) for S in seq_lengths])
            perf_values = np.minimum(peak_flops, ai_values * mem_bandwidth)
            pct_peak = perf_values / peak_flops * 100

            # Apply SRAM penalty if tiles don't fit
            if exceeds_sram:
                # Severe penalty - must spill to HBM, loses flash attention benefit
                sram_overflow_ratio = sram_kb / SRAM_SIZE_KB
                penalty = 0.3 / sram_overflow_ratio  # More overflow = worse
                pct_peak = pct_peak * penalty
                linestyle = ':'  # Dotted for infeasible
                label = f'Flash Br={Br} (exceeds SRAM)'
            else:
                linestyle = '-'
                label = f'Flash Br={Br} ({sram_kb:.0f}KB)'

            ax.plot(seq_lengths, pct_peak, color=color, linewidth=3,
                    linestyle=linestyle, label=label)

            # Mark where this Br saturates (reaches 95% of its max)
            if not exceeds_sram:
                max_pct = pct_peak[-1]
                saturate_idx = np.argmax(pct_peak >= 0.95 * max_pct)
                if saturate_idx > 0:
                    ax.plot(seq_lengths[saturate_idx], pct_peak[saturate_idx], 'o',
                            color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)

        # === Mark peak as horizontal line ===
        ax.axhline(y=100, color='black', linestyle=':', alpha=0.5, linewidth=1)

        # === Add annotation for the insight ===
        ax.annotate('Larger Br → higher AI,\nbut must fit in SRAM',
                    xy=(50000, 70), fontsize=10, style='italic', color='gray')

        # === Labels ===
        ax.set_xlabel('Sequence Length (S)', fontsize=12)
        ax.set_ylabel('% of Peak Performance', fontsize=12)
        ax.set_title(f'Flash Attention Performance vs Sequence Length\nA100 GPU (FP16), d={d}, SRAM={SRAM_SIZE_KB}KB', fontsize=14)

        ax.set_xlim(10, 100000)
        ax.set_ylim(0, 110)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/olivier/ml/image/flash_attention/performance_vs_block_size.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: performance_vs_block_size.png")


def plot_tile_size_heatmap():
    """
    2D heatmap showing performance across Br × Bc combinations.
    Style matches plot_performance_vs_block_size.
    Uses actual SRAM calculation consistent with other plots.
    """
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(10, 8))

        tile_sizes = [16, 32, 64, 128, 256, 512]
        n = len(tile_sizes)
        d = 64  # head dimension (consistent with other plots)
        SRAM_SIZE_KB = 164  # A100 shared memory

        # === SRAM calculation (consistent with other plots) ===
        def sram_required_kb(Br, Bc, d):
            """
            SRAM needed for Flash Attention tiles:
            - Q tile: Br × d
            - K tile: Bc × d
            - V tile: Bc × d
            - O accumulator: Br × d
            - Statistics (m, l): 2 × Br
            Total elements: 2×Br×d + 2×Bc×d + 2×Br
            """
            elements = 2 * Br * d + 2 * Bc * d + 2 * Br
            bytes_needed = elements * 2  # FP16
            return bytes_needed / 1024  # KB

        # Simulated performance data
        performance = np.zeros((n, n))
        exceeds_sram = np.zeros((n, n), dtype=bool)

        for i, Br in enumerate(tile_sizes):
            for j, Bc in enumerate(tile_sizes):
                sram_kb = sram_required_kb(Br, Bc, d)
                exceeds_sram[i, j] = sram_kb > SRAM_SIZE_KB

                if exceeds_sram[i, j]:
                    # Exceeds SRAM: very poor performance (worse than any fitting tile)
                    overflow_ratio = sram_kb / SRAM_SIZE_KB
                    performance[i, j] = 0.2 / overflow_ratio  # 0.08 to 0.17
                else:
                    # Fits in SRAM: performance based on tile size and balance
                    # Larger tiles = higher arithmetic intensity = better
                    min_tile = min(Br, Bc)
                    ai_score = np.log2(min_tile) / np.log2(256)  # 0.25 to 1.0 for 16-256

                    # Penalty for unbalanced tiles
                    balance = 1 - abs(Br - Bc) / max(Br, Bc) * 0.15

                    # Small bonus for tiles close to SRAM limit (maximizing utilization)
                    utilization = min(sram_kb / SRAM_SIZE_KB, 1.0)
                    utilization_bonus = utilization * 0.1

                    performance[i, j] = ai_score * balance + utilization_bonus + 0.3

        # Normalize to 0-1
        performance = (performance - performance.min()) / (performance.max() - performance.min())

        # Colormap: red (bad) -> yellow -> green (good)
        cmap = 'RdYlGn'

        im = ax.imshow(performance, cmap=cmap, aspect='auto')

        # Labels
        ax.set_xticks(range(n))
        ax.set_xticklabels(tile_sizes)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tile_sizes)
        ax.set_xlabel('Block Size Bc (K, V dimension)', fontsize=12)
        ax.set_ylabel('Block Size Br (Q dimension)', fontsize=12)
        ax.set_title('Performance Heatmap: Br × Bc Tile Combinations\nA100 GPU (FP16)', fontsize=14)

        # Add values
        for i in range(n):
            for j in range(n):
                # Use white text on dark colors (low and high values), black on yellow middle
                if performance[i, j] < 0.3 or performance[i, j] > 0.85:
                    color = 'white'
                else:
                    color = 'black'
                ax.text(j, i, f'{performance[i, j]:.2f}', ha='center', va='center',
                       fontsize=10, color=color, fontweight='bold')

        # Mark optimal
        opt_i, opt_j = np.unravel_index(np.argmax(performance), performance.shape)
        rect = mpatches.Rectangle((opt_j - 0.5, opt_i - 0.5), 1, 1,
                                   fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(rect)

        # Add hatching to cells that exceed SRAM
        for i, Br in enumerate(tile_sizes):
            for j, Bc in enumerate(tile_sizes):
                if exceeds_sram[i, j]:
                    hatch_rect = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                     fill=False, hatch='///',
                                                     edgecolor='black', linewidth=0.5, alpha=0.7)
                    ax.add_patch(hatch_rect)

        # Find and draw SRAM boundary line (approximate)
        # The boundary is where sram_required_kb(Br, Bc, d) = SRAM_SIZE_KB
        # 2*d*(Br + Bc) + 2*Br = SRAM_SIZE_KB * 1024 / 2
        # For d=64: 128*(Br+Bc) + 2*Br = 83968 → Br + Bc ≈ 650 (roughly)
        # But we need to find where the boundary crosses the grid
        boundary_points = []
        for i, Br in enumerate(tile_sizes):
            for j, Bc in enumerate(tile_sizes):
                sram_kb = sram_required_kb(Br, Bc, d)
                if abs(sram_kb - SRAM_SIZE_KB) < 30:  # Near boundary
                    boundary_points.append((j, i))

        # Label the exceeded region
        if np.any(exceeds_sram):
            # Find center of exceeded region
            exceeded_indices = np.where(exceeds_sram)
            center_i = np.mean(exceeded_indices[0])
            center_j = np.mean(exceeded_indices[1])
            ax.text(center_j, center_i, 'SRAM\nExceeded', fontsize=10, color='black',
                    ha='center', va='center', alpha=0.9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        cbar = plt.colorbar(im, ax=ax, label='Relative Performance')

    plt.tight_layout()
    plt.savefig('/home/olivier/ml/image/flash_attention/tile_size_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: tile_size_heatmap.png")


def plot_fa1_vs_fa2():
    """
    Compares Flash Attention v1 vs v2 HBM access patterns.

    Left panel: Scaling plot showing HBM bytes transferred vs sequence length
    Right panel: Breakdown bar chart at fixed S showing where savings come from

    Key difference:
    - FA1: outer loop K,V, inner loop Q → O written Tc times per Q block
    - FA2: outer loop Q, inner loop K,V → O written once per Q block
    """
    with plt.style.context('default'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # === Parameters ===
        d = 64
        Br = 256  # block size (Br = Bc)

        # === Sequence lengths for scaling plot ===
        seq_lengths = np.logspace(np.log10(256), np.log10(65536), 100)

        # === HBM access calculations ===
        def fa1_hbm_bytes(S, d, Br):
            """
            FA1: outer K,V (Tc iters), inner Q (Tr iters)
            - K: loaded once in outer loop = S*d
            - V: loaded once in outer loop = S*d
            - Q: loaded Tc times (once per outer iter) = S*d * Tc
            - O: read Tc times (accumulate) = S*d * Tc
            - O: write Tc times = S*d * Tc
            Total elements: 2*S*d + 3*S*d*Tc = 2*S*d + 3*S²*d/Br
            """
            Tc = S / Br
            elements = 2 * S * d + 3 * S * d * Tc
            return elements * 2  # FP16

        def fa2_hbm_bytes(S, d, Br):
            """
            FA2: outer Q (Tr iters), inner K,V (Tc iters)
            - Q: loaded once in outer loop = S*d
            - K: loaded Tr times (once per Q block) = S*d * Tr
            - V: loaded Tr times = S*d * Tr
            - O: written once (no read-back needed) = S*d
            Total elements: 2*S*d + 2*S*d*Tr = 2*S*d + 2*S²*d/Br
            """
            Tr = S / Br
            elements = 2 * S * d + 2 * S * d * Tr
            return elements * 2  # FP16

        def fa1_breakdown(S, d, Br):
            """Return breakdown of HBM accesses for FA1 (in bytes)"""
            Tc = S / Br
            return {
                'Q read': S * d * Tc * 2,      # read Tc times
                'K read': S * d * 2,            # read once (outer loop)
                'V read': S * d * 2,            # read once (outer loop)
                'O read': S * d * Tc * 2,      # read Tc times (accumulate)
                'O write': S * d * Tc * 2      # write Tc times
            }

        def fa2_breakdown(S, d, Br):
            """Return breakdown of HBM accesses for FA2 (in bytes)"""
            Tr = S / Br
            return {
                'Q read': S * d * 2,            # read once (outer loop)
                'K read': S * d * Tr * 2,      # read Tr times
                'V read': S * d * Tr * 2,      # read Tr times
                'O read': 0,                    # never read back!
                'O write': S * d * 2           # write once
            }

        # ============ LEFT PANEL: Scaling plot ============
        fa1_bytes = np.array([fa1_hbm_bytes(S, d, Br) for S in seq_lengths])
        fa2_bytes = np.array([fa2_hbm_bytes(S, d, Br) for S in seq_lengths])

        # Convert to GB
        fa1_gb = fa1_bytes / (1024**3)
        fa2_gb = fa2_bytes / (1024**3)

        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Plot lines
        ax1.plot(seq_lengths, fa1_gb, color='#e63946', linewidth=3,
                label='FA1: 2Sd + 3S²d/Br', linestyle='--')
        ax1.plot(seq_lengths, fa2_gb, color='#2a9d8f', linewidth=3,
                label='FA2: 2Sd + 2S²d/Br')

        # Fill the difference
        ax1.fill_between(seq_lengths, fa2_gb, fa1_gb, alpha=0.2, color='#2a9d8f',
                        label='FA2 savings')

        # Annotate ratio at specific points
        for S_mark in [4096, 16384]:
            idx = np.argmin(np.abs(seq_lengths - S_mark))
            ratio = fa1_bytes[idx] / fa2_bytes[idx]
            ax1.annotate(f'{ratio:.1f}× less',
                        xy=(seq_lengths[idx], fa2_gb[idx]),
                        xytext=(seq_lengths[idx] * 0.4, fa2_gb[idx] * 0.3),
                        fontsize=10, color='#2a9d8f',
                        arrowprops=dict(arrowstyle='->', color='#2a9d8f', alpha=0.7))

        ax1.set_xlabel('Sequence Length (S)', fontsize=12)
        ax1.set_ylabel('HBM Bytes Transferred (GB)', fontsize=12)
        ax1.set_title(f'HBM Traffic Scaling\nd={d}, Br={Br}', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_xlim(256, 65536)

        # ============ RIGHT PANEL: Breakdown bar chart ============
        S_fixed = 4096  # Fixed sequence length for breakdown

        fa1_bd = fa1_breakdown(S_fixed, d, Br)
        fa2_bd = fa2_breakdown(S_fixed, d, Br)

        # Convert to MB for readability
        categories = ['Q read', 'K read', 'V read', 'O read', 'O write']
        fa1_values = np.array([fa1_bd[cat] for cat in categories]) / (1024**2)
        fa2_values = np.array([fa2_bd[cat] for cat in categories]) / (1024**2)

        x = np.arange(len(categories))
        width = 0.35

        # Colors for each category
        cat_colors = ['#457b9d', '#457b9d', '#457b9d', '#e63946', '#e63946']

        bars1 = ax2.bar(x - width/2, fa1_values, width, label='FA1',
                       color=[c if v > 0 else 'white' for c, v in zip(cat_colors, fa1_values)],
                       edgecolor='black', alpha=0.7, hatch='///')
        bars2 = ax2.bar(x + width/2, fa2_values, width, label='FA2',
                       color=[c if v > 0 else 'white' for c, v in zip(cat_colors, fa2_values)],
                       edgecolor='black', alpha=0.9)

        # Add value labels (show actual values, even small ones)
        for bar, val in zip(bars1, fa1_values):
            label = f'{val:.1f}' if val < 1 else f'{val:.0f}'
            y_pos = max(bar.get_height(), 0.3)  # minimum height for visibility
            ax2.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, y_pos),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='#e63946')

        for bar, val in zip(bars2, fa2_values):
            if val == 0:
                label = '0'
            elif val < 1:
                label = f'{val:.1f}'
            else:
                label = f'{val:.0f}'
            y_pos = max(bar.get_height(), 0.3)
            ax2.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, y_pos),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='#2a9d8f')

        # Highlight the key difference
        ax2.annotate('Key difference:\nFA2 never reads O back',
                    xy=(3.5, max(fa1_values[3], fa1_values[4]) * 0.5),
                    fontsize=9, ha='center', style='italic', color='gray',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylabel('HBM Bytes (MB)', fontsize=12)
        ax2.set_title(f'HBM Access Breakdown at S={S_fixed}\nFA1 vs FA2', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3)

        # Add total comparison
        fa1_total = sum(fa1_values)
        fa2_total = sum(fa2_values)
        ratio = fa1_total / fa2_total
        ax2.text(0.02, 0.98, f'Total: FA1={fa1_total:.0f}MB, FA2={fa2_total:.0f}MB\nFA2 is {ratio:.1f}× more efficient',
                transform=ax2.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.suptitle('Flash Attention v1 vs v2: Theoretical HBM Access Comparison\n(Both produce identical output - difference is memory access pattern)', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig('/home/olivier/ml/image/flash_attention/fa1_vs_fa2.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: fa1_vs_fa2.png")


if __name__ == '__main__':
    print("Generating Flash Attention visualizations...\n")

    plot_tile_size_vs_latency()
    plot_memory_scaling()
    plot_arithmetic_intensity()
    plot_performance_vs_block_size()
    plot_tile_size_heatmap()
    plot_fa1_vs_fa2()

    print("\nAll visualizations saved to /home/olivier/ml/image/flash_attention/")
