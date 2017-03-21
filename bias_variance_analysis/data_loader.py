# from bias_variance_analysis import pd
import pandas as pd

class DataLoader(object):
    """A Blue print of the data loader which loads the data and has methods to print
    dimensions of the data"""

    def __init__(self, data, target):
        """Assumes the data is an pandas data frame with the feature names as first row
        Attributes:
            data = A Pandas Data Frame
            target = A string indicating the target feature name"""
        self.data = data
        self.features = data.columns.drop(target)
        self.target = target

    def display_parameters(self):
        print("Features:\n", self.features)
        print("\nTarget variable:", self.target)
        print(self.data.describe())
        # print("\nSample data:\n",self.data.head())


if __name__ == "__main__":
    input_path = "/media/gowtham/D6DC3802DC37DC05/Datasets/Bank marketing"
    input_data = pd.read_csv(input_path+"/bank-full.csv", delimiter=";")

    bank_data = DataLoader(input_data, 'y')
    bank_data.display_parameters()
