# Flash Attention v2 Visualizations

## Plots

### 1. `arithmetic_intensity.png` - Roofline Model
Shows Flash Attention workload on the roofline model with varying sequence length S.

- **X-axis**: Arithmetic Intensity (FLOPs/byte)
- **Y-axis**: Performance (FLOPs/s)
- **Color gradient**: Blue (short S, memory-bound) → Red (optimal) → Green (long S, compute-bound)
- **Red dot**: Optimal point where workload transitions from memory-bound to compute-bound

Parameters: A100 GPU (FP16), d=64, Br=256

### 2. `performance_vs_block_size.png` - Block Size Comparison
Shows how block size (Br) affects achievable performance, **including SRAM constraints**.

- **X-axis**: Sequence Length (S)
- **Y-axis**: % of Peak Performance
- **Gray dashed**: Standard Attention (must materialize S×S matrix in HBM)
- **Solid lines**: Flash Attention with Br values that fit in SRAM (legend shows SRAM usage)
- **Dotted line**: Br=512 exceeds SRAM - severe performance penalty
- **Dots**: Saturation point (95% of maximum)

#### SRAM constraint (key insight!)
Flash Attention tiles must fit in SRAM (~164KB on A100). For FP16 with d=64:
```
SRAM needed = (4 × Br × d + 2 × Br) × 2 bytes
```

| Br  | SRAM (KB) | Fits? | Max % of Peak |
|-----|-----------|-------|---------------|
| 64  | 32        | ✓     | ~42%          |
| 128 | 64        | ✓     | ~84%          |
| 256 | 129       | ✓     | 100%          |
| 512 | 258       | ✗     | ~20% (spills!) |

**Br=512 exceeds SRAM** and must spill to HBM, losing Flash Attention's benefit entirely.

#### Why larger Br (that fits) is better
Peak performance requires arithmetic intensity ≥ 153 FLOPs/byte (A100).
Since max AI ≈ Br, you need Br ≥ 153 to reach peak. But Br must also fit in SRAM.

**Optimal Br = largest value that fits in SRAM** (typically 128-256 for d=64).

### 3. `memory_scaling.png` - HBM vs SRAM
Two-panel plot showing the **two memory systems** in Flash Attention:

#### Left Panel: HBM (Global Memory)
- **Gray dashed**: Standard Attention O(S²) - stores full attention matrix
- **Teal solid**: Flash Attention O(S·d) - **same for all Br values**
- **Dotted line**: A100 HBM limit (80GB)

| Method | HBM Complexity | At S=100k |
|--------|----------------|-----------|
| Standard | O(S²) | ~20 GB |
| Flash | O(S·d) | ~0.05 GB |

**Key insight**: Flash Attention HBM usage is independent of Br - it only stores Q, K, V, O.

#### Right Panel: SRAM (On-chip Memory)
Bar chart showing SRAM required per block size:

| Br  | SRAM (KB) | Status |
|-----|-----------|--------|
| 16  | 8         | ✓ Safe (green) |
| 32  | 16        | ✓ Safe (green) |
| 64  | 32        | ✓ Safe (green) |
| 128 | 64        | ✓ Safe (green) |
| 256 | 129       | ⚠ Borderline (orange) |
| 512 | 258       | ✗ Exceeds (red, hatched) |

**Key insight**: SRAM is the binding constraint, not HBM. Tiles that exceed SRAM spill to HBM, destroying performance.

#### Why this matters
Flash Attention's speedup comes from keeping tiles in **fast SRAM** instead of slow HBM:
- SRAM bandwidth: ~19 TB/s (A100)
- HBM bandwidth: ~2 TB/s (A100)

If tiles don't fit in SRAM, you lose this 10x bandwidth advantage.

### 4. `tile_size_vs_latency.png` - Tile Size Trade-offs
U-shaped curve showing the latency trade-off when choosing tile size.

- **X-axis**: Block Size (Br), log scale
- **Y-axis**: Relative Latency (lower is better)
- **Red line**: Total latency
- **Orange dashed**: Kernel overhead component (decreases with larger tiles)
- **Gray dashed**: SRAM spill cost component (zero until exceeds, then explodes)
- **Red shaded region**: Exceeds SRAM - severe performance penalty
- **Star**: Optimal tile size (Br=256)

#### The trade-off
| Region | Dominant Factor | Latency |
|--------|-----------------|---------|
| Small tiles (Br < 64) | Kernel overhead | High (too many launches) |
| Medium tiles (64-256) | Balanced | Low (sweet spot) |
| Large tiles (Br > 325) | SRAM spill | Very high (10x slower HBM) |

The SRAM boundary is at Br≈325 (where SRAM usage = 164KB). The plot shows this with a vertical dotted line and red shading for the exceeded region.

#### SRAM usage by tile size
Labels below each point show SRAM usage:
| Br | SRAM | Status |
|----|------|--------|
| 64 | 32KB | ✓ Safe |
| 128 | 64KB | ✓ Safe |
| 256 | 129KB | ✓ Fits (optimal!) |
| 512 | 258KB | ✗ Exceeds |

#### Key insight
**Optimal Br = largest tile that fits in SRAM**

Going smaller wastes compute (kernel overhead). Going larger spills to HBM (10x bandwidth penalty).

### 5. `tile_size_heatmap.png` - Br × Bc Performance
2D heatmap showing relative performance across different tile size combinations.

- **X-axis**: Block Size Bc (K, V dimension) - tile size for key/value blocks
- **Y-axis**: Block Size Br (Q dimension) - tile size for query blocks
- **Color**: Green = high performance, Red = low performance
- **Black box**: Optimal tile combination
- **Dashed line**: SRAM limit boundary

#### What the heatmap shows
The heatmap reveals the performance landscape when choosing Br and Bc independently:

| Region | Performance | Reason |
|--------|-------------|--------|
| Top-left (small Br, Bc) | Moderate (~0.50-0.65) | Fits in SRAM, but kernel overhead |
| Center (Br=64, Bc=64) | Good (0.75) | Good balance |
| Diagonal (Br=128, Bc=128) | Better (0.87) | Larger tiles, still fits |
| (Br=256, Bc=256) | Optimal (1.00) | Largest balanced that fits SRAM |
| Bottom-right (large Br, Bc) | Poor (0.00-0.03) | Exceeds SRAM, hatched |

#### SRAM constraint
The A100 has ~164KB of usable shared memory per thread block. For FP16 with d=64:
```
SRAM needed = (2×Br×d + 2×Bc×d + 2×Br) × 2 bytes  (for Q, K, V, O tiles + stats)
```

Cells that exceed SRAM are hatched and show very low performance (0.00-0.03).

| Br | Bc | SRAM (KB) | Exceeds? |
|----|-----|-----------|----------|
| 256 | 256 | 129 | No ✓ |
| 256 | 512 | 193 | Yes ✗ (hatched) |
| 512 | 256 | 194 | Yes ✗ (hatched) |
| 512 | 512 | 258 | Yes ✗ (hatched) |

#### Key insights
1. **Optimal is (256, 256)**: Largest balanced tile that fits in SRAM
2. **Exceeded cells are clearly worst**: 0.00-0.03 vs 0.50+ for fitting tiles
3. **Larger tiles (that fit) are better**: Higher arithmetic intensity
4. **Balanced tiles preferred**: Imbalanced tiles have ~15% penalty

#### Relationship to other plots
- `performance_vs_block_size.png` shows Br effect at fixed Bc (assumes Bc scales with Br)
- This heatmap shows the full 2D space when Br and Bc are chosen independently

## Hardware Parameters (A100 SXM4)
- Peak FLOPs: 312 TFLOPS (FP16)
- HBM Bandwidth: 2039 GB/s
- SRAM per SM: ~164 KB (configurable shared memory)
- SRAM Bandwidth: ~19 TB/s
- Critical Intensity: ~153 FLOPs/byte

## Key Insight: SRAM is the Constraint

Flash Attention's performance comes from **keeping tiles in SRAM** (10x faster than HBM).
All plots consistently show:

| Plot | What it shows about SRAM |
|------|--------------------------|
| `performance_vs_block_size` | Br=512 exceeds SRAM → poor performance (dotted line) |
| `memory_scaling` | Bar chart shows Br=512 exceeds 164KB limit (hatched) |
| `tile_size_vs_latency` | SRAM boundary at Br≈325, red shading for exceeded region |
| `tile_size_heatmap` | Only 3 cells exceed SRAM → red, hatched (256×512, 512×256, 512×512) |

**Optimal strategy**: Choose the largest Br that fits in SRAM (typically 128-256 for d=64).
