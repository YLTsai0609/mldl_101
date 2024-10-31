"""
This is a self-attention calculation with numpy.
https://github.com/p208p2002/Self-Attention-cacultate-with-numpy/blob/master/Self-Attention-Caculate.ipynb

A review of self-attention, cross-attention, and other attention mechanisms.

https://github.com/YLTsai0609/DataScience101/blob/master/sa.md?__readwiseLocation=
"""

import numpy as np
import scipy.special


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


# n_samples (n_token) = 3, (in nlp problem, would be the sequence length)
# n_input_feature = 4

input_1 = np.array([1, 0, 1, 0], dtype="float32")
input_2 = np.array([0, 2, 0, 2], dtype="float32")
input_3 = np.array([1, 1, 1, 1], dtype="float32")
inputs = np.vstack([input_1, input_2, input_3])  # (n_samples, n_input_feature) = (3,4)
# Positional Encoder usually add in input here


wk = np.array(
    [[0, 1], [1, 0], [0, 0], [1, 1]], dtype="float32"
)  # (n_input_feature, n_hidden) = (2,3) dimension reduction
wq = np.array([[1, 0], [1, 0], [0, 1], [0, 0.2]], dtype="float32")
wv = np.array([[0, 2], [2, 3], [1, 3], [1, 0]], dtype="float32")

# generate query, key, value from np.random.randn(n_input_feature, n_hidden)
# test_q = np.random.randn(4,5)
# test_q.shape


q = (
    inputs @ wq
)  # (n_samples, n_input_feature) dot (n_input_feature, n_hidden) = (n_samples, n_hidden)
k = inputs @ wk
v = inputs @ wv

d = np.sqrt(q.shape[-1])  # numbers of features normalization

A = softmax(
    q @ k.T / d
)  # (n_samples, n_hidden) dot (n_hidden, n_samples) = (n_samples, n_samples)

A2 = scipy.special.softmax(q @ k.T / d, axis=1)

O = A @ v  # (n_samples, n_samples) dot (n_samples, n_hidden) = (n_samples, n_hidden)

# attention matrix (n_samples, n_samples) and softmax normalization is the bottleneck of computation
