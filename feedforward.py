import numpy as np

from utils import softmax

# 模擬輸入資料: 3 筆樣本，每筆有 4 個特徵 (n_samples, n_input_features) = (3, 4)
input_1 = np.array([1, 0, 1, 0], dtype="float32")
input_2 = np.array([0, 2, 0, 2], dtype="float32")
input_3 = np.array([1, 1, 1, 1], dtype="float32")
inputs = np.vstack([input_1, input_2, input_3])  # (3, 4)

# 定義 dense layer 的權重矩陣和偏置
n_input_features = 4  # 輸入特徵數
n_output_features_l1 = 5  # 輸出特徵數 (dense layer 的單元數)
n_output_features_l2 = 3  # 輸出特徵數 (dense layer 的單元數)
# 隨機初始化權重矩陣和偏置項
w1 = np.random.randn(n_input_features, n_output_features_l1)  # (4, 3)
b1 = np.random.randn(n_output_features_l1)  # (3,)

w2 = np.random.randn(n_output_features_l1, n_output_features_l2)  # (3, 1)
b2 = np.random.randn(n_output_features_l2)  # (1,)

# 計算 fully connected layer 的輸出
o1 = inputs @ w1 + b1  # (3, 4) dot (4, 3) + (3,) -> (3, 3)
o2 = o1 @ w2 + b2  # (3, 3) dot (3, 1) + (1,) -> (3, 1)
print("Inputs:\n", inputs)
print("Weights:\n", w1, w2)
print("Bias:\n", b1, b1)
print("Output:\n", o1, o2)

print(softmax(o2))  # normalization to each class to perform cross-entropy
