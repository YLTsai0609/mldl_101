import numpy as np


def softmax(x: "np.ndarray(M, N)") -> "np.ndarray(M, N)":
    """
    Softmax normalization for each row, usually used to amplify the difference between the componenets.
    >>> x = np.array([1, 2, 3])
    >>> softmax(x)
    array([[0.09003057, 0.24472847, 0.66524096]])

    >>> x = np.array([[1, 2, 3], [1,2,5]])
    >>> softmax(x)
    array([[0.09003057, 0.24472847, 0.66524096],
           [0.01714783, 0.04661262, 0.93623955]])
    """

    # NOTE: num.sum(axis=-1), sum at the last axis
    return np.exp(x) / np.sum(np.exp(x), axis=-1).reshape(-1, 1)


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
