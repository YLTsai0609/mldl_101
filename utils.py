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

if __name__ == "__main__":
    # numpy broadcasting 
    a = np.array([1, 2, 3])
    b = 2
    print(a * b) # [2, 4, 6]

    # numpy broadcasting 2
    # https://xiao2macf.blog.csdn.net/article/details/128210631?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-128210631-blog-120618706.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-1-128210631-blog-120618706.235%5Ev43%5Epc_blog_bottom_relevance_base8&utm_relevant_index=1&ydreferer=aHR0cHM6Ly9ibG9nLmNzZG4ubmV0Lw%3D%3D
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([100, 200, 300])
    print(a.shape, b.shape, a + b) # (2, 3) (3,) [[101, 202, 303], [104, 205, 306]]
    # 比對最長的維度，短得必須為 1 ，否則就會 error，是一個省記憶體的設計
