import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from normalize import MinMaxScaler
from pca import PCA
import pickle


def check_numpy_array(name, array):
    print(f"Shape of {name}: {array.shape}")


class ActivationFunctions:
    @staticmethod
    def purelin(x):
        return x

    @staticmethod
    def dpurelin(x):
        return np.ones_like(x)

    @staticmethod
    def logsig(x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dlogsig(x):
        fx = ActivationFunctions.logsig(x)
        return fx * (1 - fx)
    
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
class NeuralNetwork:
    # Constructor initialize the hyperparameters & weights & rmse_list
    def __init__(self, n, epochs, learning_rate):
        # Initialize all activation Functions 
        # from ActivationFunctions class to get activation functions
        # Help me to pass function_variable to other functions
        self.purelin = ActivationFunctions.purelin
        self.dpurelin = ActivationFunctions.dpurelin
        self.logsig = ActivationFunctions.logsig
        self.dlogsig = ActivationFunctions.dlogsig
        self.softmax = ActivationFunctions.softmax

        # get Hyperparameters
        self.n = n
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Initialize loss_list
        self.total_loss_list = []

        # train dataset
        self.train_x = pd.read_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\train_x.csv', header=None)
        self.train_y = pd.read_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\train_y.csv', header=None)
        # test dataset
        self.test_x = pd.read_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\test_x.csv', header=None)
        self.test_y = pd.read_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\test_y.csv', header=None)

        # get input_size
        input_size = len(self.train_x.columns)
        # get input_size
        output_size = len(self.train_y.columns)

        # Initialize weights
        self.w_out = np.random.rand(self.n, output_size)*2-1
        self.w_hid = np.random.rand(input_size, self.n)*2-1

    def cross_entropy(self, true_vector, pred_vector):
        return -np.mean(np.sum(true_vector * np.log(pred_vector), axis=1))

    def forward_propagation(self, input_vector):
        if isinstance(input_vector, pd.DataFrame):
            input_vector = input_vector.values  # Convert to numpy array if input is a DataFrame
        sum_hid = input_vector @ self.w_hid
        a_hid = self.logsig(sum_hid)
        sum_out = a_hid @ self.w_out
        a_out = self.softmax(sum_out)
        return a_hid, a_out

    def Back_propagation(self, error_vector, a_out, a_hid, input_vector, learning_rate):
        if isinstance(input_vector, pd.DataFrame):
            input_vector = input_vector.values  # Convert to numpy array if input is a DataFrame
        Delta_out = (error_vector)
        Delta_hid = Delta_out @ self.w_out.T
        self.w_out += learning_rate * a_hid.T @ Delta_out
        self.w_hid += learning_rate * input_vector.T @ (Delta_hid * self.dlogsig(a_hid))
        return self.w_out, self.w_hid
    
    def train(self):
        for epoch in range(self.epochs):
            loss_list = [self.train_step(i) for i in range(len(self.train_x))]
            self.total_loss_list.append(np.mean(loss_list))

            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}, Loss: {self.total_loss_list[-1]:.4f}')

    def train_step(self, i):
        input_vector = np.array(self.train_x.iloc[i]).reshape(1, -1)
        a_hid, a_out = self.forward_propagation(input_vector)
        true_vector = np.array(self.train_y.iloc[i]).reshape(1, -1)
        pred_vector = a_out

        loss = self.cross_entropy(true_vector, pred_vector)
        error_vector = (true_vector - pred_vector)

        # update weights
        self.w_out, self.w_hid = self.Back_propagation(error_vector, a_out, a_hid, input_vector, self.learning_rate)
        return loss
    
    # defined predict function and return predict_df and true_df
    def predict(self):
        _, predict_value = self.forward_propagation(self.train_x.values)  # Ensure X is a NumPy array here
        groundTrue_array = self.test_y.values
        predict_array = predict_value
        return groundTrue_array, predict_array
    
    def save_weights(self):
        np.save('./weights/w_out.npy', self.w_out)
        np.save('./weights/w_hid.npy', self.w_hid)

    def save_df(self):
        self.predict_df.to_csv('./result/predict.csv', index=False, header=None)
        self.true_df.to_csv('./result/true.csv', index=False, header=None)

    # use Plotting class to plot loss and predict
    def loss_plot(self):
        plt.plot(self.total_loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()


def calculate_accuracy(y_true, y_pred):
    # 找出每個標籤和預測值中最大值的索引
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    # 比較這些索引是否相等，並計算準確度
    accuracy = np.mean(y_true_indices == y_pred_indices)

    return accuracy

def train():
    hyperparameters = {
    'n': 80,                 # n: number of hidden neurons
    'epochs': 300,          # epochs: number of training epochs
    'learning_rate': 0.1,   # learning_rate: learning rate   
    }
    nn = NeuralNetwork(**hyperparameters)
    nn.train()
    save_model(nn)

def save_model(nn):   
    # 保存模型參數
    with open('model_parameters.pkl', 'wb') as f:
        pickle.dump(nn, f)

def load_model():
    # 載入模型參數
    with open('model_parameters.pkl', 'rb') as f:
        nn = pickle.load(f)
    return nn

def predict():
    nn = load_model()
    true, predict = nn.predict()
    accuracy = calculate_accuracy(true, predict)
    print(f'Accuracy: {accuracy:.4f}')
    nn.loss_plot()


if __name__ == "__main__":
    # train()
    predict()
    
