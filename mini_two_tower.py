import numpy as np

# 超參數
n_features = 4  # 原始特徵維度
embedding_dim = 3  # 嵌入維度

# 模擬使用者特徵和項目特徵的輸入
user_features = np.array([1, 0, 1, 0], dtype="float32")  # 使用者特徵 (1, 4)
item_features = np.array([0, 1, 0, 1], dtype="float32")  # 項目特徵 (1, 4)

# 隨機初始化兩個塔的權重
user_weights = np.random.randn(n_features, embedding_dim)  # 使用者塔的權重 (4, 3)
item_weights = np.random.randn(n_features, embedding_dim)  # 項目塔的權重 (4, 3)

# 前向傳播計算嵌入
user_embedding = user_features @ user_weights  # (1, 4) dot (4, 3) = (1, 3)
item_embedding = item_features @ item_weights  # (1, 4) dot (4, 3) = (1, 3)

# 計算使用者和項目嵌入的相似度分數（內積）# Not normalized
similarity_score = np.dot(
    user_embedding, item_embedding.T
)  # (1, 3) dot (3, 1) = (1, 1)

print("User embedding:\n", user_embedding)
print("Item embedding:\n", item_embedding)
print("Similarity score:\n", similarity_score)
