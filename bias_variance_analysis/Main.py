import pandas as pd
from bias_variance_analysis.naive_bayes import naive_bayes
from sklearn.model_selection import train_test_split


def find_feature_type(data):
    feature_type = {}
    for feature in data.columns:
        if len(data[feature].value_counts()) > 25:
            feature_type[feature] = "continuous"
        else:
            feature_type[feature] = "discrete"
    return feature_type


def down_sample(data):
    good_data = data[data["y"] == "yes"]
    bad_data = data[data["y"] == "no"]
    good_test, good_train = train_test_split(good_data, test_size=1000)
    bad_test, bad_train = train_test_split(bad_data, test_size=1000)
    x_train = pd.concat([good_train, bad_train])
    x_test = pd.concat([good_test, bad_test])
    y_train = x_train.pop("y")
    y_test = x_test.pop("y")
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    input_path = "/media/gowtham/D6DC3802DC37DC05/Datasets/Bank marketing"
    data = pd.read_csv(input_path + "/bank-full.csv", delimiter=";")

    # bank_data = DataLoader(input_data, 'y')
    # bank_test = DataLoader(test_data, 'y')
    # # bank_data.display_parameters()
    #
    # lr_model = LogisticRegression(bank_data, bank_test)
    # lr_model.learn()
    # lr_model.predict()
    # lr_model.find_accuracy()
    # lr_model.print_parameters()

    nb_model = naive_bayes(data, find_feature_type(data))
    x_train, x_test, y_train, y_test = down_sample(data)
    nb_model.train(x_train, y_train)
