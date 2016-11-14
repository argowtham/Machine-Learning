import pandas as pd
from sklearn.ensemble import BaggingClassifier as bc
import numpy as np
from Decision_trees import Decision_tree_classifier as dc

if __name__ == "__main__":
    data = pd.read_csv(input("Enter the path of the train file"), sep=',', header='infer')
    test_data = pd.read_csv(input("Enter the path of the test file"), sep=',', header='infer')
    target_feature = input("Enter the target feature that has to be classified")
    Y = data[target_feature]
    X = data.drop(data.columns[20:22], axis=1)
    X_test = test_data.drop(test_data.columns[20:22], axis=1)

    model = bc()
    model.fit(X, Y)
    predicted = model.predict(X_test)
    actual = np.array(test_data['bruises?_bruises'])
    dc.evaluate(predicted, actual)
