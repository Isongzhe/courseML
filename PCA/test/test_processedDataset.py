import numpy as np
import pandas as pd
from ProcessedDataset import ProcessedDataset, params
import pytest

class TestProcessedDataset:
    @pytest.fixture
    def setup(self):
        model = ProcessedDataset(params)
        return model

    def test_one_hot_encoding(self, setup):
        setup.one_hot_encoding()
        assert setup.train_y.ndim == 2, "One-hot encoding failed, output should be 2D"
        assert setup.test_y.ndim == 2, "One-hot encoding failed, output should be 2D"

    def test_apply_pca(self, setup):
        train_x = np.random.rand(100, 65)
        test_x = np.random.rand(50, 65)
        X_train_pca, X_test_pca = setup.apply_pca(train_x, test_x)
        assert X_train_pca.shape == train_x.shape, "PCA failed, output shape should be the same as input"
        assert X_test_pca.shape == test_x.shape, "PCA failed, output shape should be the same as input"

    def test_apply_normalization(self, setup):
        np.random.seed(0)  # 確保每次生成的隨機數據都是一樣的
        train_x = np.random.rand(100, 65)
        test_x = np.random.rand(50, 65)
        X_train_normalized, X_test_normalized = setup.apply_normalization(train_x, test_x)
        assert np.max(X_train_normalized) <= 1, "Normalization failed, max should be <= 1"
        assert np.min(X_train_normalized) >= 0, "Normalization failed, min should be >= 0"

if __name__ == '__main__':
    pytest.main()