import pandas as pd

class OneHotEncoder:
    def __init__(self, output_file):
        self.output_dataset = pd.read_csv(output_file, header=None)

    def one_hot_encoding(self):
        self.output_dataset = pd.get_dummies(self.output_dataset[0]).astype(int)

    def get_encoded_data(self):
        return self.output_dataset