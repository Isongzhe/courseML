import pytest
import numpy as np
from model import ActivationFunctions, NeuralNetwork

def test_purelin():
    assert ActivationFunctions.purelin(5) == 5
    assert ActivationFunctions.purelin(-3) == -3

def test_dpurelin():
    assert np.all(ActivationFunctions.dpurelin(np.array([5, -3, 0])) == np.ones(3))

def test_logsig():
    assert np.allclose(ActivationFunctions.logsig(np.array([0, -1, 1])), np.array([0.5, 0.26894142, 0.73105858]), atol=1e-6)

def test_dlogsig():
    assert np.allclose(ActivationFunctions.dlogsig(np.array([0, -1, 1])), np.array([0.25, 0.19661193, 0.19661193]), atol=1e-6)

def test_softmax():
    assert np.allclose(ActivationFunctions.softmax(np.array([[1, 2, 3], [2, 3, 4]])), np.array([[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]), atol=1e-6)

def test_cross_entropy():
    nn = NeuralNetwork(3, 3, 0.1)
    true_vector = np.array([[1, 0, 0], [0, 1, 0]])
    pred_vector = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    expected_cross_entropy = -np.mean(np.sum(true_vector * np.log(pred_vector), axis=1))
    assert np.allclose(nn.cross_entropy(true_vector, pred_vector), expected_cross_entropy, atol=1e-6)

# def test_forward_propagation():
#     nn = NeuralNetwork(3, 3, 0.1)
#     a_hid, a_out = nn.forward_propagation(np.array([[1, 2, 3], [4, 5, 6]]))
#     assert a_hid.shape == (2, 3)
#     assert a_out.shape == (2, 3)

# def test_Back_propagation():
#     nn = NeuralNetwork(3, 3, 0.1)        # get input_size
#     error_vector = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
#     a_out = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
#     a_hid = np.array([[0.26894142, 0.73105858, 0.5], [0.26894142, 0.73105858, 0.5]])
#     input_vector = np.array([[1, 2, 3], [4, 5, 6]])
#     learning_rate = 0.1
#     w_out, w_hid = nn.Back_propagation(error_vector, a_out, a_hid, input_vector, learning_rate)
#     assert w_out.shape == (3, 3)
#     assert w_hid.shape == (3, 3)

def test_train_step():
    nn = NeuralNetwork(3, 3, 0.1)
    loss = nn.train_step(0)
    assert isinstance(loss, float)

def test_predict():
    nn = NeuralNetwork(3, 3, 0.1)
    true, predict = nn.predict()
    assert true.shape == predict.shape