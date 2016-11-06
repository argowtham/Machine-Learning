import pandas as pd
from sklearn import tree
import numpy as np


def evaluate(array1, array2):
    correct, wrong = 0, 0
    for i, j in zip(array1, array2):
        if i == j:
            correct += 1
        else:
            wrong += 1
    print("Accuracy is {}%".format(round(correct / len(array1), 2)*100))
    print("Error rate is {}%".format(round(wrong / len(array1), 2)*100))


if __name__ == "__main__":
    data = pd.read_csv(
        "C:\/Users\Gowtham\Box Sync\IU\Fall'16\Applied Machine Learning\Homeworks\PA2\mushrooms\/agaricuslepiotatrain1.csv",
        sep=',', header='infer')
    test_data = pd.read_csv(
        "C:\/Users\Gowtham\Box Sync\IU\Fall'16\Applied Machine Learning\Homeworks\PA2\mushrooms\/agaricuslepiotatest1.csv",
        sep=',', header='infer')
    Y = data['bruises?_bruises']
    X = data.drop(data.columns[20:22], axis=1)
    X_test = test_data.drop(test_data.columns[20:22], axis=1)

    model = tree.DecisionTreeClassifier(criterion='entropy')
    model.fit(X, Y)
    predicted = model.predict(X_test)
    actual = np.array(test_data['bruises?_bruises'])
    evaluate(predicted, actual)
