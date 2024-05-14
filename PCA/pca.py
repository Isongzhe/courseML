import pandas as pd
import numpy as np

# 讀取 CSV 檔案
df = pd.read_csv('./PCA/ORL_dataset/train_images.csv', header=None)

# 將 DataFrame 轉換為 numpy 數組
X = df.values

# 確認 X 的形狀
print('Shape of X:', X.shape)

def pca(X, n_components):
    # 1. 中心化數據（將數據的平均值減為 0）
    X_centered = X - np.mean(X, axis=0)

    # 2. 計算協方差矩陣
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # 3. 計算協方差矩陣的特徵值和特徵向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 4. 對特徵值進行排序（並保留索引）
    sorted_index = np.argsort(eigenvalues)[::-1]

    # 5. 選擇前 n_components 個特徵向量
    sorted_eigenvectors = eigenvectors[:, sorted_index[:n_components]]

    # 6. 將數據投影到選擇的主成分上
    X_pca = X_centered @ sorted_eigenvectors

    return X_pca

X_pca = pca(X, 65)

# 確認 X 的形狀
print('Shape of X_pca:', X_pca.shape)