'''
Recap KMeans
for i in range(max_iter):
    # E-step (Expectation): 計算每個點屬於各個類別的機率（在 K-means 中是硬分配）
    distances = np.zeros((num_clusters, inputs.shape[0]))
    for c in range(num_clusters):
        distances[c] = LA.norm(inputs - centroids[c], axis=1)
    cluster_assignments = np.argmin(distances, axis=0)

    # M-step (Maximization): 更新模型參數（在 K-means 中是更新中心點）
    for c in range(num_clusters):
        centroids[c] = inputs[cluster_assignments == c].mean(axis=0)

A brief introduction to EM algorithm
https://zh.wikipedia.org/zh-tw/%E6%9C%80%E5%A4%A7%E6%9C%9F%E6%9C%9B%E7%AE%97%E6%B3%95
'''
import numpy as np
from numpy import linalg as LA

class EMAlgorithm:
    def __init__(self):
        self.parameters = None
        self.latent_variables = None
    
    def e_step(self, data):
        """計算隱變量的期望"""
        raise NotImplementedError
    
    def m_step(self, data):
        """更新模型參數"""
        raise NotImplementedError
    
    def fit(self, data, max_iter=100):
        for i in range(max_iter):
            # E-step
            self.latent_variables = self.e_step(data)
            
            # M-step
            self.parameters = self.m_step(data)
            
            # 收斂檢查
            if self.converged():
                break

class KMeans(EMAlgorithm):
    def __init__(self, n_clusters : int, data : "np.ndarray(n_samples, n_features)"):
        self.n_clusters = n_clusters
        self.centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)] # (n_clusters, n_features)
        self.cluster_assignments = None # (n_samples,)
        self.loss = []    

    def e_step(self, data : "np.ndarray(n_samples, n_features)"):
        """實現 K-means 的分配步驟"""
        distances = np.zeros((self.n_clusters, data.shape[0])) # (n_clusters, n_samples)
        for c in range(self.n_clusters):
            distances[c] = LA.norm(data - self.centroids[c], axis=1)
        self.loss.append(self.compute_loss(distances))
        return np.argmin(distances, axis=0)
    
    def m_step(self, data : "np.ndarray(n_samples, n_features)"):
        """實現 K-means 的中心點更新"""
        for c in range(self.n_clusters):
            self.new_centroids[c] = data[self.latent_variables == c].mean(axis=0)
    
    def compute_loss(self, distances : "np.ndarray(n_clusters, n_samples)"):
        '''
        replace this if need other loss function
        '''
        return np.sum(distances ** 2)
    
    def fit(self, data, max_iter=100):
        for i in range(max_iter):
            self.cluster_assignments = self.e_step(data) # (n_samples,)
            self.m_step(data)
            self.loss.append(self.compute_loss(data))

    def predict(self, data : "np.ndarray(n_samples, n_features)"):
        return self.e_step(data)
