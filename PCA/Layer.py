import numpy as np
class ActivationFunctions:
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def dsigmoid(z):
        sz = ActivationFunctions.sigmoid(z)
        return sz * (1 - sz)
    
    @staticmethod
    def logsig(x):
        return -np.log(1 + np.exp(-x))
    
    @staticmethod
    def dlogsig(x):
        sx = ActivationFunctions.sigmoid(x)
        return sx * (1 - sx)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        # Implement the forward computation for the layer
        raise NotImplementedError
        
    def backward(self, output_error, learning_rate):
        # Implement the backward computation for the layer
        raise NotImplementedError

class InputLayer(Layer):
    def __init__(self, input_size, output_size):
        super(InputLayer, self).__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
        
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error.mean(axis=0) * self.input.shape[0]
        
        return input_error

class SigmoidLayer(Layer):
    # 补全遗失的forward方法
    def forward(self, input_data):
        self.input = input_data
        self.output = ActivationFunctions.sigmoid(np.dot(input_data, self.weights) + self.bias)
        return self.output
    
    # 补全遗失的backward方法
    def backward(self, output_gradient, learning_rate):
        sigmoid_gradient = output_gradient * ActivationFunctions.dsigmoid(self.output)
        input_error = np.dot(sigmoid_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, sigmoid_gradient)
        
        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * sigmoid_gradient.mean(axis=0)
        
        return input_error
    
class LogsigLayer(Layer):
    # 补全遗失的forward方法
    def forward(self, input_data):
        self.input = input_data
        self.output = ActivationFunctions.logsig(np.dot(input_data, self.weights) + self.bias)
        return self.output
    
    # 补全遗失的backward方法
    def backward(self, output_gradient, learning_rate):
        sigmoid_gradient = output_gradient * ActivationFunctions.dlogsig(self.output)
        input_error = np.dot(sigmoid_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, sigmoid_gradient)
        
        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * sigmoid_gradient.mean(axis=0)
        
        return input_error
    
class SoftmaxOutputLayer(Layer):
    def __init__(self, input_size, output_size):
        super(SoftmaxOutputLayer, self).__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        linear_output = np.dot(input_data, self.weights) + self.bias
        self.output = ActivationFunctions.softmax(linear_output)
        return self.output

    def backward(self, expected_output, learning_rate):
        output_error = self.output - expected_output
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error.mean(axis=0)

        return input_error
