import numpy as np


def expand_dim(x: "np.ndarray(N)") -> "np.ndarray(1, N)":
    """
    Expand the dimension of the input array by 1.
    >>> x = np.array([1, 2, 3])
    >>> expand_dim(x)
    array([[1],
           [2],
           [3]])
    >>> expand_dim(x).shape
    (3, 1)
    """
    return x.reshape(-1, 1)
