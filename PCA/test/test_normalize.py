import pytest
import numpy as np
from normalize import Normalization

def test_normalization():
    # 建立 Normalization 物件
    norm = Normalization()

    # 建立一個用於測試的數據集
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 應用 fit_transform 方法
    X_norm = norm.fit_transform(X)

    # 檢查結果是否正確
    assert np.allclose(X_norm, (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)))

    # 檢查 min 和 range 屬性是否正確設置
    assert np.allclose(norm.min, np.min(X, axis=0))
    assert np.allclose(norm.range, np.max(X, axis=0) - np.min(X, axis=0))

    # 建立一個新的數據集
    X_new = np.array([[10, 11, 12], [13, 14, 15]])

    # 應用 transform 方法
    X_new_norm = norm.transform(X_new)

    # 檢查結果是否正確
    assert np.allclose(X_new_norm, (X_new - norm.min) / norm.range)