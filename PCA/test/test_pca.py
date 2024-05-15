import pytest
import numpy as np
import sys
sys.path.append('D:\\GitHub\\courseML\\PCA')
from pca import PCA

def test_pca():
    # 創建一個簡單的數據集
    X = np.array([[1, 2], [3, 4], [5, 6]])

    # 創建PCA對象並擬合數據
    pca = PCA(1)
    pca.fit(X)

    # 驗證平均值和特徵向量是否正確
    np.testing.assert_array_almost_equal(pca.mean, np.array([3, 4]))
    np.testing.assert_array_almost_equal(np.abs(pca.eigenvectors), np.abs(np.array([[-0.70710678], [-0.70710678]])))

    # 轉換數據並驗證結果
    X_pca = pca.transform(X)
    expected_result = np.array([[-2.82842712], [0.], [2.82842712]])
    np.testing.assert_array_almost_equal(X_pca, expected_result)