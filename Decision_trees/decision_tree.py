import math
import numpy


def build_tree(S):
    pass


def calculate_entropy(dataset, target):
    pos, neg, total = 0, 0, len(dataset)
    for i in range(0, total):
        if dataset[i][target] == 1:
            pos += 1
        else:
            neg += 1
    p_pos = (pos/total)
    p_neg = (neg/total)
    entropy = - p_pos*math.log(p_pos, 2) - p_neg*math.log(p_neg, 2)
    return entropy


def calculate_gain(dataset, names):
    s = calculate_entropy(dataset, 0)
    for feature in names[1:7]:
        s_v = []
        unique_value = list(set(dataset[feature]))
        print(unique_value)
        for row in dataset:
            if row[feature] ==



if __name__ == '__main__':
    print("Program to implement the decision tree without using package")
    path = "D:\OneDrive\IU\Fall'16\Applied Machine Learning\Homeworks\PA1\monks"
    train_files = ["monks-1.train", "monks-2.train", "monks-3.train"]
    features = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    # data_type = [('class', 'i8'), ('a1', 'i8'), ('a2', 'i8'), ('a3', 'i8'),
    #              ('a4', 'i8'), ('a5', 'i8'), ('a6', 'i8'), ('Id', 'S5')]
    for file in train_files:
        data = numpy.genfromtxt(path + '\/' + file, names=features, usecols=(range(0, 7)), dtype=None)
        gain = calculate_gain(data, features)
        break

