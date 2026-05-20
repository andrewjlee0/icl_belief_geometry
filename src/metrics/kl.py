"""KL divergence computations."""
import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """KL(p || q) per row."""
    return np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)
