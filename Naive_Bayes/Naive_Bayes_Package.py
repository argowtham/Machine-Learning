from Naive_Bayes import NaiveBayes as nb
import sklearn.naive_bayes as naivebayes
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
    print("=============================================")
    print("\t\t\tNaive Bayes Algorithm using package")
    print("=============================================")
    file_path = input("Enter the path of the file\n")
    data = nb.process_data(file_path, sep=',')
    train_data, test_data = nb.split_data(data, fraction=0.67)
    n = len(data.columns) - 1
    X = train_data.drop(train_data.columns[n], axis=1)
    Y = train_data[n]
    model = naivebayes.GaussianNB()
    model.fit(X, Y)
    test_data.drop(test_data.columns[n], axis=1)
    predicted = model.predict(test_data)
    actual = np.array(test_data[n])
    evaluate(predicted, actual)
