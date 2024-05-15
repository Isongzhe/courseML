import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Plotting Class
class Plotting:
    # Constructor initialize the model(get model from NeuralNetwork class)
    def __init__(self, model):
        self.model = model

    def trainLoss_plot(self):
        plt.plot(self.model.total_loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()

    # defined regression classify function
    def predict_plot(self, mode):
        # get model's df and turn (predict_df, true_df) to ndarray (by values.ravel())
        predict = self.model.predict_df.values
        true = self.model.true_df.values

        # If the mode is one-hot encoding, create a binary scatter plot for each class
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
            # Create a color map
            colors = {1:'red', 2:'green', 3:'blue'}
            # Create a scatter plot with different colors for each class
            plt.scatter(true, predict, c=[colors[i[0]] for i in true])
            plt.xlabel('True_class')
            plt.ylabel('Predict_class')
            plt.title(f'{mode}: Predict vs True for Class')
            plt.show()
        else:
            print('Invalid mode. Please enter one_hot_encoding or regression')
            return
# Activation Functions Class
class ActivationFunctions:
    # Constructor static methods
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
        # 計算所有元素的指數
        exps = np.exp(x - np.max(x))
        # 計算 softmax
        return exps / np.sum(exps)
    
class ProcessedDataset:
    def __init__(self, mode):
        self.mode = mode
        # load dataset
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
        # 將資料轉換為 one-hot 編碼
        self.output_dataset = pd.get_dummies(self.output_dataset[0]).astype(int)
        
    def get_processedDatasets(self):
        return self.input_dataset, self.output_dataset

# Neural Network Class
class NeuralNetwork:
    # Constructor initialize the hyperparameters & weights & rmse_list
    def __init__(self, n, split_ratio, epochs, learning_rate, loss_function = 'mse', mode = 'regression'):

        # Initialize Plotting class (Delegation plot function to Plotting class)
        # Help me to change plot's ways in the future
        self.plotting = Plotting(self)
        
        # Initialize all activation Functions 
        # from ActivationFunctions class to get activation functions
        # Help me to pass function_variable to other functions
        self.purelin = ActivationFunctions.purelin
        self.dpurelin = ActivationFunctions.dpurelin
        self.logsig = ActivationFunctions.logsig
        self.dlogsig = ActivationFunctions.dlogsig
        self.softmax = ActivationFunctions.softmax

        self.mode = mode

        # Initialize load_dataset function
        processedDataset_model = ProcessedDataset(mode)
        self.input_data, self.output_data = processedDataset_model.get_processedDatasets()

        # get Hyperparameters
        self.n = n
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        # Initialize loss_list
        self.total_loss_list = []

    def load_dataset(self):
        # # load dataset
        # self.input_data = pd.read_csv('./datasets/iris_in.csv', header=None)
        # self.output_data = pd.read_csv('./datasets/iris_out.csv', header=None)

        # get split dataset ratio(eq: train: 50%, test: 50%)
        self.idx = int(len(self.input_data) * self.split_ratio)
        # train dataset
        self.train_x = self.input_data.iloc[:self.idx, :]
        self.train_y = self.output_data.iloc[:self.idx, :]
        # test dataset
        self.test_x = self.input_data.iloc[self.idx:, :]
        self.test_y = self.output_data.iloc[self.idx:, :]

        # get input_size
        input_size = len(self.train_x.columns)
        # get input_size
        output_size = len(self.train_y.columns)

        # Initialize weights
        self.w_out = np.random.rand(self.n, output_size)*2-1
        self.w_hid = np.random.rand(input_size, self.n)*2-1

    def cross_entropy(self, true_vector, pred_vector):
        # print(f'這是每筆的np.log {-np.log10(pred_vector)}')
        return np.sum(true_vector * (-np.log10(pred_vector)))

    # defined forward propagation function
    def forward_propagation(self, input_vector):

        # pls check the shape of input_vector shape needs(1,4), ex: 4 = features(input_sizes)
        sum_hid = input_vector @ self.w_hid #(1,4)@(4,n) = (1,n)
        a_hid = self.logsig(sum_hid) #(1,n) n=hidden layer node number

        sum_out = a_hid @ self.w_out #(1,n) @ (n,output_size) = (1, output_size)
        # print(f"sum_out: {sum_out}")
        a_out = self.softmax(sum_out) # (1, output_size)
        # print(f"a_out: {a_out}")

        return a_hid, a_out
    
    # defined back propagation function
    def Back_propagation(self, error_vector, a_out, a_hid, input_vector, learning_rate):
        # for example output_size = 3
        # error shape = (1,output_size), =(1,3)
        # get Delta_out and Delta_hid 
        # Delta_out = (error_vector) * self.dpurelin(a_out) #(1,output_size)*(1,output_size) = (1,output_size)
        Delta_out = (error_vector) 
        Delta_hid = Delta_out @ self.w_out.T #(1,output_size)@ (output_size,n) = (1,n)

        # update weights
        # 1.weight_out
        self.w_out += learning_rate * a_hid.T @ Delta_out #(n,output_size) + lr* (n,1) @ (1,output_size) = (n,output_size)

        # 2.weight_hid
        self.w_hid += learning_rate * input_vector.T @ (Delta_hid * self.dlogsig(a_hid)) #(input_sizes,n) + lr* (input_sizes,1) @ [(1,n)*(1,n)] = (input_sizes,n)

        # return updated weights
        return self.w_out, self.w_hid
    
    # defined train function
    def train(self):
        # iterate epochs times(and then we will run down all training process)
        for epoch in range(self.epochs):
            # iterate over each training data and update weights
            loss_list = [self.train_step(i) for i in range(len(self.train_x))]

            # append the rmse of each epoch to rmse_list
            self.total_loss_list.append(np.mean(loss_list))
            if (epoch+1) % 100 == 0:
                # print the epoch and rmse
                print(f'Epoch: {epoch+1}, Loss: {self.total_loss_list[-1]:.4f}')
            else:
                pass
    
    # defined train_step function (run over each training data and update weights is one epoch)
    def train_step(self,i):
        # get input vector(也就是input_data的每一筆數據, reshape成(1,features=input_sizes))
        input_vector = np.array(self.train_x.iloc[i]).reshape(1, -1) # (1, input_sizes)
        # do forward propagation(得到a_hid, a_out)
        a_hid, a_out = self.forward_propagation(input_vector)

        # get true_value and pred_value
        true_vector = np.array(self.train_y.iloc[i]).reshape(1, -1) # (1, output_size)
        pred_vector = a_out   # (1, output_size)

        if self.loss_function == 'mse':
            error_vector = true_vector - pred_vector
            loss = np.square(error_vector).sum()

        elif self.loss_function == 'cross_entropy':
            loss = self.cross_entropy(true_vector, pred_vector)
            error_vector = -(pred_vector - true_vector)

        else:
            raise ValueError(f'Invalid loss function: {self.loss_function}, please input correct loss function')
        
        self.w_out, self.w_hid = self.Back_propagation(error_vector, a_out, a_hid, input_vector, self.learning_rate)
        return loss
    
    # defined predict function and return predict_df and true_df
    def predict(self):
        _, predict_value = self.forward_propagation(self.test_x) #return a_out就好
        self.predict_df = pd.DataFrame(predict_value)  # pred true (turn to dataframe from tuple)
        self.true_df = self.test_y # ground true (test_y)
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

    # use Plotting class to plot loss and predict
    def loss_plot(self):
        self.plotting.trainLoss_plot()
    
    def result_plot(self):
        self.plotting.predict_plot(self.mode)


def calculate_accuracy(y_true, y_pred):
    # 找出每個標籤和預測值中最大值的索引
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    # 比較這些索引是否相等，並計算準確度
    accuracy = np.mean(y_true_indices == y_pred_indices)

    return accuracy



if __name__ == "__main__":
    hyperparameters = {
    'n': 15,                 # n: number of hidden neurons
    'split_ratio': 0.5,      # split_ratio: train/test split ratio
    'epochs': 1000,          # epochs: number of training epochs
    'learning_rate': 0.0003,   # learning_rate: learning rate
    'loss_function':'cross_entropy', # loss_function: mse or cross_entropy
    'mode': 'one_hot_encoding' # mode: one_hot_encoding or normal(default)    
    }
    nn = NeuralNetwork(**hyperparameters)
    nn.load_dataset()
    nn.train()
    accuracy = calculate_accuracy(*nn.predict())
    print(f'Accuracy: {accuracy:.4f}')
    
    nn.loss_plot()