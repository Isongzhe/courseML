import pandas as pd
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X):
        # 1. 中心化數據
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 計算協方差矩陣
        covariance_matrix = np.cov(X_centered.T)

        # 3. 計算協方差矩陣的特徵值和特徵向量
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 4. 對特徵值進行排序（並保留索引）
        sorted_index = np.argsort(eigenvalues)[::-1]

        # 5. 選擇前 n_components 個特徵向量
        self.eigenvectors = eigenvectors[:, sorted_index[:self.n_components]]

    def transform(self, X):
        # 6. 將數據投影到選擇的主成分上
        X_centered = X - self.mean
        X_pca = X_centered @ self.eigenvectors
        return X_pca
    
def main():
    # 讀取 CSV 檔案，並轉為 numpy array
    X_train = pd.read_csv('./PCA/ORL_dataset/train_images.csv', header=None).values
    # 讀取 CSV 檔案，並轉為 numpy array
    X_test = pd.read_csv('./PCA/ORL_dataset/train_images.csv', header=None).values


    # 確認 X_train 的形狀
    print('Shape of X_train before PCA:', X_train.shape)
    # 確認 X_test 的形狀
    print('Shape of X_test  before PCA:', X_test.shape)
    # 使用PCA方法，降維到65維：
    pca = PCA(65)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 確認 X_train 的形狀
    print('Shape of X_train after PCA:', X_train_pca.shape)
    # 確認 X_test 的形狀
    print('Shape of X_test  after PCA:', X_test_pca.shape)

if __name__ == '__main__':
    main()