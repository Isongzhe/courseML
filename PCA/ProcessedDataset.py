import pandas as pd
import numpy as np
from normalize import MinMaxScaler
from pca import PCA

params = {
    'n_components': 65,
    'input_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_images.csv',
    'output_train_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_images.csv',
    'input_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\train_labels.csv',
    'output_test_path': r'D:\GitHub\courseML\PCA\ORL_dataset\test_labels.csv'
}

class ProcessedDataset:
    def __init__(self, params):
        self.pca = PCA(params['n_components'])
        self.normalization = MinMaxScaler()

        self.train_x = pd.read_csv(params['input_train_path'], header=None).values
        self.test_x = pd.read_csv(params['output_train_path'], header=None).values
        self.train_y = pd.read_csv(params['input_test_path'], header=None).values
        self.test_y = pd.read_csv(params['output_test_path'], header=None).values
        check_numpy_array(self.train_x, "train_x before PCA and normalization")
        check_numpy_array(self.train_y, "train_y before one-hot encoding")
        check_numpy_array(self.test_x, "test_x before PCA and normalization")
        check_numpy_array(self.test_y, "test_y before one-hot encoding")

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
    def apply_pca(self, train_X, test_X):
        self.pca.fit(train_X)
        X_train_pca = self.pca.transform(train_X)
        X_test_pca = self.pca.transform(test_X)
        return X_train_pca,X_test_pca
    
    # 應用標準化，並返回處理後的數據集(只有訓練集需要fit，測試集不需要fit)
    def apply_normalization(self, train_X, test_X):
        # 使用訓練數據來 fit normalizer
        self.normalization.fit(train_X)
        # 使用 normalizer 來轉換訓練數據
        X_train_normalized = self.normalization.transform(train_X)
        X_test_normalized = self.normalization.transform(test_X)
        return X_train_normalized, X_test_normalized
    
    def get_processedDatasets(self):
        # 應用PCA
        X_train_pca, X_test_pca = self.apply_pca(self.train_x, self.test_x)
        # 應用標準化(先PCA再標準化)
        self.train_x, self.test_x = self.apply_normalization(X_train_pca, X_test_pca)

        return np.real(self.train_x), self.train_y, np.real(self.test_x), self.test_y

def check_numpy_array(file, file_name):
    print(f"Checking file: {file_name}")
    print(f"Number of images: {len(file)}")
    print(f"Shape of images: {file.shape}")

def main():
    model = ProcessedDataset(params)
    model.one_hot_encoding()
    train_x,train_y,test_x,test_y = model.get_processedDatasets()
    print('************************')

    check_numpy_array(train_x, "train_x after PCA and normalization")
    check_numpy_array(train_y, "train_y after one-hot encoding")
    check_numpy_array(test_x, "test_x after PCA and normalization")
    check_numpy_array(test_y, "test_y after one-hot encoding")


    # 保存為 CSV (index=False不要保存row索引；header=False不要保存column名)
    pd.DataFrame(train_x).to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\train_x.csv', index=False,  header=False)
    pd.DataFrame(train_y).to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\train_y.csv', index=False,  header=False)
    pd.DataFrame(test_x).to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\test_x.csv', index=False,  header=False)
    pd.DataFrame(test_y).to_csv(r'D:\GitHub\courseML\PCA\ORL_dataset\processedDataset\test_y.csv', index=False,  header=False)


if __name__ == '__main__':
    main()