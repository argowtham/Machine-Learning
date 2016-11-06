import pandas as pd
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.tree import DecisionTreeClassifier as dc
import numpy as np
from Decision_trees import Decision_tree_classifier as dt


if __name__ == "__main__":
    data = pd.read_csv("C:\/Users\Gowtham\Box Sync\IU\Fall'16\Applied Machine Learning\Homeworks\PA2\mushrooms\/agaricuslepiotatrain1.csv", sep=',', header = 'infer')
    test_data = pd.read_csv("C:\/Users\Gowtham\Box Sync\IU\Fall'16\Applied Machine Learning\Homeworks\PA2\mushrooms\/agaricuslepiotatest1.csv", sep=',', header = 'infer')
    Y = data['bruises?_bruises']
    X = data.drop(data.columns[20:22], axis=1)
    X_test = test_data.drop(test_data.columns[20:22], axis=1)

    model = abc(dc(max_depth=1, criterion='entropy'), algorithm="SAMME", n_estimators=10)
    model.fit(X, Y)

    predicted = model.predict(X_test)
    actual = np.array(test_data['bruises?_bruises'])
    dt.evaluate(predicted, actual)
