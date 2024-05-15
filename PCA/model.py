import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from normalize import MinMaxScaler
from pca import PCA
from Layer import Layer

def check_numpy_array(name, array):
    print(f"Shape of {name}: {array.shape}")

class ProcessedDataset:
    def __init__(self, params):
        self.pca = PCA(params['n_components'])
        self.normalization = MinMaxScaler()

        self.train_x = pd.read_csv(params['input_train_path'], header=None).values
        self.test_x = pd.read_csv(params['output_train_path'], header=None).values
        self.train_y = pd.read_csv(params['input_test_path'], header=None).values
        self.test_y = pd.read_csv(params['output_test_path'], header=None).values

    def one_hot_encoding(self):
        # 先將數據轉換為一維數組
        self.train_y = np.squeeze(self.train_y)
        self.test_y = np.squeeze(self.test_y)
        # 獲取所有不同的類別
        classes = np.unique(np.concatenate((self.train_y, self.test_y)))
        # 將類別轉換為 one-hot 編碼
        self.train_y = (self.train_y[:, None] == classes).astype(int)
        self.test_y = (self.test_y[:, None] == classes).astype(int)

    # 應用PCA，並返回處理後的數據集(只有訓練集需要fit，測試集不需要fit)
    def apply_pca(self):
        self.pca.fit(self.train_x)
        self.train_x = self.pca.transform(self.train_x)
        self.test_x = self.pca.transform(self.test_x)

    # 應用標準化，並返回處理後的數據集(只有訓練集需要fit，測試集不需要fit)
    def apply_normalization(self):
        # 使用訓練數據來 fit normalizer
        self.normalization.fit(self.train_x)
        # 使用 normalizer 來轉換訓練數據
        self.train_x = self.normalization.transform(self.train_x)
        self.test_x = self.normalization.transform(self.test_x)
    
    def get_processedDatasets(self):
        # 應用PCA
        self.apply_pca()
        # 應用標準化(先PCA再標準化)
        self.apply_normalization()
        # 轉成onehot encoding
        self.one_hot_encoding()

        return np.real(self.train_x), self.train_y, np.real(self.test_x), self.test_y

# class NeuralNetwork:
#     def __init__(self, layers):
#         self.layers = layers
#         self.losses = []

#     def train(self, trainx, trainy, epochs, learning_rate):
#         for epoch in range(epochs):
#             epoch_loss = 0
#             for i in range(len(trainx)):
#                 # 前向傳播
#                 output = trainx[i]
#                 for layer in self.layers:
#                     output = layer.forward(output)

#                 # 反向傳播
#                 expected_output = trainy[i]
#                 error = output - expected_output
#                 for layer in reversed(self.layers):
#                     error = layer.backward(error, learning_rate)

#                 # 計算損失
#                 epoch_loss += np.sum(error**2) / 2

#             # 記錄每個 epoch 的平均損失
#             self.losses.append(epoch_loss / len(trainx))

#     def plot_loss(self):
#         plt.plot(self.losses)
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.show()

#     def pred(self, testx, testy):
#         correct = 0
#         for i in range(len(testx)):
#             output = testx[i]
#             for layer in self.layers:
#                 output = layer.forward(output)

#             if np.argmax(output) == np.argmax(testy[i]):
#                 correct += 1

#         return correct / len(testx)
    

# # 主程序，创建和训练神经网络
# if __name__ == "__main__":
#     # load dataset and preprocess using ProcessedDataset, etc.
#     params = {
#     'n_components': 65,
#     'input_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_images.csv',
#     'output_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_images.csv',
#     'input_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_labels.csv',
#     'output_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_labels.csv'
#     }
#     processedDataset_model = ProcessedDataset(params)
#     train_x, train_y, test_x, test_y = processedDataset_model.get_processedDatasets()
#     input_size = train_x.shape[1]
#     output_size = train_y.shape[1]

#     # 創建神經網路的層
#     input_layer = Layer.InputLayer(input_size, 50)
#     hidden_layer = Layer.LogsigLayer(50, 50)
#     output_layer = Layer.SoftmaxOutputLayer(50, output_size)
#     nn = NeuralNetwork([input_layer, hidden_layer, output_layer])

#     # 訓練神經網路
#     nn.train(train_x, train_y, epochs=1000, learning_rate=0.01)

#     # 測試神經網路
#     accuracy = nn.pred(test_x, test_y)
#     print("Test accuracy:", accuracy)

#     # Plot the loss over epochs
#     nn.plot_loss()