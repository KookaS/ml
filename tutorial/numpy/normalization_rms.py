import numpy as np

def norm_rms(x, gamma=1.0):
    """
    RMS normalization is used in Layer Normalization for Transformers.
    
    Formula:
        norm(x) = x * gamma / RMS(x)
        RMS(x) = sqrt( sum(x**2) / len(x))

    :param x: Input [B, S, N, H]
    """
    # keep B, S, N separated from the layer norm
    rms = np.sqrt(np.sum(x**2, axis=-1, keepdims=True) / x.shape[-1])
    return x * gamma / rms

if __name__ == "__main__":
    B, S, N, H = 1, 10, 2, 8
    x = np.ones((B, S, N, H))