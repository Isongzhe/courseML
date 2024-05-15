import pytest
import numpy as np
import pandas as pd
from model import ActivationFunctions, ProcessedDataset, NeuralNetwork, calculate_accuracy

def test_apply_pca():
    params = {
    'n_components': 65,
    'input_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_images.csv',
    'output_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_images.csv',
    'input_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_labels.csv',
    'output_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_labels.csv'
    }
    model = ProcessedDataset('one_hot_encoding', params)
    X_train_pca, X_test_pca = model.apply_pca(model.input_train, model.input_test)
    assert X_train_pca.shape == (200, 65)
    assert X_test_pca.shape == (200, 65)

def test_apply_normalization():
    params = {
        'n_components': 65,
    'input_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_images.csv',
    'output_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_images.csv',
    'input_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_labels.csv',
    'output_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_labels.csv'
    }
    model = ProcessedDataset('one_hot_encoding', params)
    X_train_normalized, X_test_normalized = model.apply_normalization(model.input_train, model.input_test)
    
    # 檢查數據是否已經正規化
    assert np.min(X_train_normalized) == 0
    assert np.max(X_train_normalized) == 1
    assert np.min(X_test_normalized) == 0
    assert np.max(X_test_normalized) == 1


def test_NeuralNetwork():
    nn = NeuralNetwork(10, 100, 0.01, 'cross_entropy')
    assert nn.n == 10
    assert nn.epochs == 100
    assert nn.learning_rate == 0.01
    assert nn.loss_function == 'cross_entropy'

def test_activation_functions():
    x = np.array([0, 1, 2])
    assert np.all(ActivationFunctions.purelin(x) == x)
    assert np.all(ActivationFunctions.dpurelin(x) == np.ones_like(x))
    assert np.all(ActivationFunctions.logsig(x) == 1 / (1 + np.exp(-x)))
    assert np.all(ActivationFunctions.dlogsig(x) == ActivationFunctions.logsig(x) * (1 - ActivationFunctions.logsig(x)))
    assert np.allclose(ActivationFunctions.softmax(x), np.exp(x) / np.sum(np.exp(x)))

def test_processed_dataset():
    params = {
        'n_components': 2,
        'input_train_path': r'D:\GitHub\courseML\datasets\iris_in.csv',
        'output_train_path': r'D:\GitHub\courseML\datasets\iris_out.csv',
        'split_ratio': 0.8
    }
    pd_model = ProcessedDataset('one_hot_encoding', params)
    assert pd_model.mode == 'one_hot_encoding'
    assert pd_model.split_ratio == 0.8

def test_neural_network():
    hyperparameters = {
        'n': 2,
        'epochs': 10,
        'learning_rate': 0.01,
        'loss_function': 'cross_entropy',
    }
    nn = NeuralNetwork(**hyperparameters)
    assert nn.n == 2
    assert nn.epochs == 10
    assert nn.learning_rate == 0.01
    assert nn.loss_function == 'cross_entropy'

def test_calculate_accuracy():
    y_true = np.array([[0, 1], [1, 0], [0, 1]])
    y_pred = np.array([[0, 1], [1, 0], [1, 0]])
    assert calculate_accuracy(y_true, y_pred) == 2/3

