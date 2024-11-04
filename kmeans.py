'''
K-Means Algorithm implementation
'''
import numpy as np
import numpy.linalg as LA

input_1 = np.array([1, 0], dtype="float32")
input_2 = np.array([0, 2], dtype="float32")
input_3 = np.array([0, 2.1], dtype="float32")
input_4 = np.array([1, 0.1], dtype="float32")
inputs = np.vstack([input_1, input_2, input_3,input_4])  # (n_samples, n_input_feature) = (4,2)

num_clusters = 2
max_iter = 10

# centroids_index = np.random.choice(inputs.shape[0], num_clusters, replace=False)
# centroids = inputs[centroids_index] #(n_clusters, n_input_feature)

centroids = np.array([[1, 0], [0, 0]]) #(n_clusters, n_input_feature)
loss = []
for i in range(max_iter):
    distances_per_cluster = []
    for c in range(num_clusters):
        distances_per_cluster.append(LA.norm(inputs - centroids[c],axis=1)) # (n_samples, n_input_features) - (c_i, n_input_features) = (n_samples, n_input_features)
    distances_per_cluster = np.vstack(distances_per_cluster) # (n_clusters, n_samples)
    # L2 norm 把 sample distaince 平方回來，在加到 loss
    loss.append(np.sum(distances_per_cluster ** 2))

    cluster_assignments = np.argmin(distances_per_cluster, axis=0) # (n_samples,) 將每個 sample 分配到對應的 cluster_id

    # 有了新的中心點後，重新計算中心點，將每個 cluster 的樣本取平均位置
    for c in range(num_clusters):
        centroids[c] = inputs[cluster_assignments == c].mean(axis=0)

    # Stop condition
    # 1. 如果 loss 不再下降，則停止
    # 2. 如果迭代次數達到最大迭代次數，則停止


def predict(new_samples : "np.ndarray(n_samples, n_input_features)", centroids : "np.ndarray(n_clusters, n_input_features)"):
    """
    generate predictions for new samples
    time complexity: O(n_clusters * n_samples)
    if non-clustered samples, time complexity: O(n_samples ** 2), where n_clusters << n_samples
    """
    distances_per_cluster = []
    for c in range(num_clusters):
        distances_per_cluster.append(LA.norm(new_samples - centroids[c],axis=1))
    distances_per_cluster = np.vstack(distances_per_cluster)
    prediction = np.argmin(distances_per_cluster, axis=0) 
    print('new_samples:',new_samples, 'centroids:', centroids, 'prediction:', prediction)
    return prediction


for s in [
    np.array([[1, 0.5]]),
    np.array([[0, 2.5]]),
]:
    predict(s, centroids)



#NOTE: 開發時，先以定義好的 inputs, centroids 來測試，再換成 random initialization 

# inputs[cluster_assignments == 0].mean(axis=0)
# LA.norm(inputs - centroids[:, np.newaxis], axis=2)

# a = LA.norm(inputs - centroids[0,:], axis=1) # (4, 2) - (2, 2) = (4, 2) # numpy 的 axis=1，是表示要把 axis=1 (行) 的值 aggregate 掉
# b = LA.norm(inputs - centroids[1,:], axis=1)

# distances_per_cluster.append(a)
# distances_per_cluster.append(b)
# distances_per_cluster = np.vstack(distances_per_cluster)
# distances_per_cluster.shape
# cluster_assignments # [0, 1, 0, 0]