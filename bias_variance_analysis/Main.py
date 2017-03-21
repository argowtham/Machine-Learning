# from bias_variance_analysis import pd
# from bias_variance_analysis import DataLoader
# from bias_variance_analysis import LogisticRegression
from bias_variance_analysis.data_loader import DataLoader
import pandas as pd
from bias_variance_analysis.Logistic_Regression import LogisticRegression


if __name__ == "__main__":
    input_path = "/media/gowtham/D6DC3802DC37DC05/Datasets/Bank marketing"
    input_data = pd.read_csv(input_path + "/bank.csv", delimiter=";")
    test_data = pd.read_csv(input_path + "/bank-full.csv", delimiter=";")

    bank_data = DataLoader(input_data, 'y')
    bank_test = DataLoader(test_data, 'y')
    # bank_data.display_parameters()

    lr_model = LogisticRegression(bank_data, bank_test)
    lr_model.learn()
    lr_model.predict()
    lr_model.find_accuracy()
    lr_model.print_parameters()
#     TODO Scale data and check how this performs
