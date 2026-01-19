import numpy as np

def pe_sinusoidal(x):
    """
    Encodes the absolute position to each token vector in dimension d_model.
    Implementation for transformers with dimension S sequence length, H d_model.
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    :param x: Input [B, S, N, H]
    :return: x + PE [B, S, N, H]
    """
    B, S, N, H = x.shape

    # evolution over the H axis (depth)
    # we take steps of 2 because sin/cos share the same values
    theta = 1.0 / 10000.0 ** (np.arange(0, H, 2) / H)

    # evolution over the S axis (time)
    pos = np.arange(S)

    # compute the frequencies in 2D, outer product pos x theta
    freqs = pos[:, None] * theta[None, :] # [S, H/2]
    cos_values = np.cos(freqs)
    sin_values = np.sin(freqs)

    # assemble cos and sin values into final PE matrix
    pe = np.zeros((S, H))
    pe[..., 0::2] = sin_values
    pe[..., 1::2] = cos_values

    # additive PE to the input
    # NumPy handles broadcasting automatically
    # [B, S, N, H] + [S, N, H] -> [B, S, N, H]
    return x + pe[:, None, :]

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Set the backend to non-interactive (headless)
    import matplotlib.pyplot as plt

    # SINUSOIDAL PE
    B, S, N, H = 1, 100, 2, 128
    # x = np.random.normal(size=(B, S, N, H))
    x = np.zeros((B, S, N, H)) # we want to show only the PE
    # Apply PE on the input X
    pe_sinusoidal = pe_sinusoidal(x) 

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(pe_sinusoidal[0, :, 0, :], aspect='auto', cmap='RdBu', origin='lower')
    plt.colorbar()
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Depth (Embedding Dim)")
    plt.ylabel("Position (Time)")
    # plt.show()
    plt.savefig("image/embedding/positional_encoding_sinusoidal.png")