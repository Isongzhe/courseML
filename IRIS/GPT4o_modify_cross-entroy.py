import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import warnings

# warnings.filterwarnings('ignore', category=FutureWarning)

class Plotting:
    def __init__(self, model):
        self.model = model

    def trainLoss_plot(self):
        plt.plot(self.model.total_loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()

    def predict_plot(self, mode):
        predict = self.model.predict_df.values
        true = self.model.true_df.values

        if mode == 'one_hot_encoding':
            for i in range(true.shape[1]):
                plt.figure()
                plt.scatter(range(len(true)), true[:, i], label='true', alpha=0.5)
                plt.scatter(range(len(predict)), predict[:, i], label='predict', alpha=0.5)
                plt.xlabel('Number of Samples')
                plt.ylabel('Class Probability')
                plt.title(f'{mode}: Predict vs True for Class {i+1}')
                plt.legend()
                plt.show()
        elif mode == 'regression':
            colors = {1:'red', 2:'green', 3:'blue'}
            plt.scatter(true, predict, c=[colors[i[0]] for i in true])
            plt.xlabel('True_class')
            plt.ylabel('Predict_class')
            plt.title(f'{mode}: Predict vs True for Class')
            plt.show()
        else:
            print('Invalid mode. Please enter one_hot_encoding or regression')
            return

class ActivationFunctions:
    @staticmethod
    def purelin(x):
        return x

    @staticmethod
    def dpurelin(x):
        return np.ones_like(x)

    @staticmethod
    def logsig(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dlogsig(x):
        fx = ActivationFunctions.logsig(x)
        return fx * (1 - fx)
    
    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

class ProcessedDataset:
    def __init__(self, mode):
        self.mode = mode
        self.input_dataset = pd.read_csv('./datasets/iris_in.csv', header=None)
        self.output_dataset = pd.read_csv('./datasets/iris_out.csv', header=None)
        if self.mode == 'one_hot_encoding': 
            self.one_hot_encoding()
            print(f'現在是{self.mode}模式')
        elif mode == 'regression':
            print(f'現在是{self.mode}模式')
        else:
            raise ValueError(f'Invalid mode: {self.mode}, 請重新輸入正確的mode')

    def one_hot_encoding(self):
        self.output_dataset = pd.get_dummies(self.output_dataset[0]).astype(int)
        
    def get_processedDatasets(self):
        return self.input_dataset, self.output_dataset

class NeuralNetwork:
    def __init__(self, n, split_ratio, epochs, learning_rate, loss_function='mse', mode='regression'):
        self.plotting = Plotting(self)
        self.purelin = ActivationFunctions.purelin
        self.dpurelin = ActivationFunctions.dpurelin
        self.logsig = ActivationFunctions.logsig
        self.dlogsig = ActivationFunctions.dlogsig
        self.softmax = ActivationFunctions.softmax

        self.mode = mode
        processedDataset_model = ProcessedDataset(mode)
        self.input_data, self.output_data = processedDataset_model.get_processedDatasets()

        self.n = n
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.total_loss_list = []

    def load_dataset(self):
        self.idx = int(len(self.input_data) * self.split_ratio)
        self.train_x = self.input_data.iloc[:self.idx, :]
        self.train_y = self.output_data.iloc[:self.idx, :]
        self.test_x = self.input_data.iloc[self.idx:, :]
        self.test_y = self.output_data.iloc[self.idx:, :]

        input_size = len(self.train_x.columns)
        output_size = len(self.train_y.columns)

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

        if self.loss_function == 'mse':
            error_vector = true_vector - pred_vector
            loss = np.mean(np.square(error_vector))
        elif self.loss_function == 'cross_entropy':
            loss = self.cross_entropy(true_vector, pred_vector)
            error_vector = -(pred_vector - true_vector)
        else:
            raise ValueError(f'Invalid loss function: {self.loss_function}, please input correct loss function')
        
        self.w_out, self.w_hid = self.Back_propagation(error_vector, a_out, a_hid, input_vector, self.learning_rate)
        return loss

    def predict(self):
        _, predict_value = self.forward_propagation(self.test_x.values)  # Ensure test_x is a NumPy array here
        self.predict_df = pd.DataFrame(predict_value)
        self.true_df = self.test_y
        return self.predict_df, self.true_df

    def save_df(self, mode):
        if mode == 'one_hot_encoding':
            self.predict_df.to_csv(f'./result/predict({mode}).csv', index=False, header=None)
            self.true_df.to_csv(f'./result/true({mode}).csv', index=False, header=None)
        elif mode == 'regression':
            self.predict_df.to_csv(f'./result/predict({mode}).csv', index=False, header=None)
            self.true_df.to_csv(f'./result/true({mode}).csv', index=False, header=None)
        else:
            raise ValueError(f'Invalid mode: {mode}, 請重新輸入正確的mode')

    def loss_plot(self):
        self.plotting.trainLoss_plot()
    
    def result_plot(self):
        self.plotting.predict_plot(self.mode)

def calculate_accuracy(y_true, y_pred):
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_true_indices == y_pred_indices)
    return accuracy

if __name__ == "__main__":
    hyperparameters = {
        'n': 15,
        'split_ratio': 0.5,
        'epochs': 5000,
        'learning_rate': 0.0003,
        'loss_function': 'cross_entropy',
        'mode': 'one_hot_encoding'
    }
    nn = NeuralNetwork(**hyperparameters)
    nn.load_dataset()
    nn.train()
    accuracy = calculate_accuracy(*nn.predict())
    print(f'Accuracy: {accuracy:.4f}')
    nn.loss_plot()