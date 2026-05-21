import numpy as np

def leaky_relu(x, alpha=0.01):
    x=np.asarray(x, dtype=float)
    return np.maximum(alpha*x, x)
    pass