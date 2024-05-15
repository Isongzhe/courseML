import numpy as np
import pytest
from normalize import MinMaxScaler

def test_min_max_scaler_init():
    scaler = MinMaxScaler()
    assert scaler.min is None
    assert scaler.max is None

def test_min_max_scaler_fit():
    scaler = MinMaxScaler()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler.fit(X)
    assert np.array_equal(scaler.min, np.array([1, 2, 3]))
    assert np.array_equal(scaler.max, np.array([7, 8, 9]))

def test_min_max_scaler_transform():
    scaler = MinMaxScaler()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    X_expected = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    assert np.allclose(X_transformed, X_expected)

def test_min_max_scaler_fit_transform():
    scaler = MinMaxScaler()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_transformed = scaler.fit_transform(X)
    X_expected = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    assert np.allclose(X_transformed, X_expected)

def test_min_max_scaler_on_single_feature():
    scaler = MinMaxScaler()
    X = np.array([[1], [2], [3]])
    X_transformed = scaler.fit_transform(X)
    X_expected = np.array([[0], [0.5], [1]])
    assert np.allclose(X_transformed, X_expected)

def test_min_max_scaler_on_constant_feature():
    scaler = MinMaxScaler()
    X = np.array([[1], [1], [1]])
    X_transformed = scaler.fit_transform(X)
    assert np.isnan(X_transformed).all(), "All values should be NaN when dividing by zero in MinMaxScaler"