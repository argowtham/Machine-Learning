import pandas as pd
import math
from Decision_trees import binary_node as bn


class BinaryDTree:
    def __init__(self, target, depth):
        self.root = None
        self.target = target
        self.depth = depth

    def buildtree(self, data, features):
        self.root = find_node(data, features)
        self.root.depth = 1
        find_child(self.root, data, features)
        pass

    def calculate_entropy(self, data):
        total = len(data)
        if total == 0:
            return 0
        else:
            classifier = self.target
            p_pos = list(data[classifier]).count(1)/total
            p_neg = list(data[classifier]).count(0)/total

            if p_pos == 0:
                x = 0
            else:
                x = - p_pos*math.log(p_pos, 2)

            if p_neg == 0:
                y = 0
            else:
                y = - p_neg * math.log(p_neg, 2)

            entropy = x + y
            return entropy

    def find_leaf(self, row):
        node_list = [self.root]
        for node in node_list:
            if not node.feature == 'leaf' and node.depth <= dt.depth:
                if row[node.feature] == node.value:
                    node_list.append(node.true_child)
                else:
                    node_list.append(node.false_child)
        return node_list.pop()

    def predict(self, data):
        data['predicted'] = -1
        for i in range(len(data)):
            leaf_node = self.find_leaf(data.loc[i])
            data.set_value(i, 'predicted', leaf_node.label)
        return data


def process_data(data, features):
    return pd.read_table(data, delim_whitespace=True, names=features, header=None)


def find_node(data, features):
    min_entropy = math.inf
    entropy = dt.calculate_entropy(data)
    if entropy == 0:
        best_node = bn.BinaryNode(name='leaf', label=setlabel(data))
    else:
        for feature in features:
            unique_values = list(set(data[feature]))
            for value in unique_values:
                sub_data_true = data.query(str(feature) + ' == ' + str(value))
                sub_data_false = data.query(str(feature) + ' != ' + str(value))
                entropy_sum = (len(sub_data_true)/len(data)) * dt.calculate_entropy(sub_data_true) + \
                              (len(sub_data_false)/len(data)) * dt.calculate_entropy(sub_data_false)
                if min_entropy > entropy_sum:
                    min_entropy = entropy_sum
                    best_feature = feature
                    best_value = value
        best_node = bn.BinaryNode(name=best_feature,
                                  label=setlabel(data.query(str(best_feature) + ' == ' + str(best_value))))
        best_node.value = best_value
    # best_node.display()
    return best_node


def find_child(node, data, features):
    sub_data_true = data.query(str(node.feature) + " == " + str(node.value))
    sub_data_false = data.query(str(node.feature) + " != " + str(node.value))
    node.true_child = find_node(sub_data_true, features)
    node.false_child = find_node(sub_data_false, features)
    node.true_child.parent = node
    node.false_child.parent = node
    node.true_child.depth = node.depth + 1
    node.false_child.depth = node.depth + 1
    if not node.true_child.feature == 'leaf':
        find_child(node.true_child, sub_data_true, features)
    if not node.false_child.feature == 'leaf':
        find_child(node.false_child, sub_data_false, features)


def setlabel(data):
    classifier = dt.target  # Indicates the name of the column which has to be classified
    p_pos = list(data[classifier]).count(1)
    p_neg = list(data[classifier]).count(0)
    if p_pos > p_neg:
        return 1
    else:
        return 0


def find_error(data, a, b):
    actual = list(data[a])
    predicted = list(data[b])
    error = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    for i in range(len(actual)):
        if not actual[i] == predicted[i]:
            error += 1
            if actual[i] == 1:
                fn += 1
            else:
                fp += 1
        else:
            if actual[i] == 1:
                tp += 1
            else:
                tn += 1
    confusion_matrix = pd.DataFrame([[tn, fp], [fn, tp]], columns=['Yes', 'No'], index=['Yes', 'No'])
    print(confusion_matrix)
    print("Error rate: ", error/len(actual))
    return round((1-(error/len(actual)))*100, 2)


if __name__ == "__main__":
    print("------------------------------------------------")
    print("Program to implement binary split decision tree")
    print("------------------------------------------------")
    depth = eval(input("Enter the depth of the decision tree\n"))
    input_file = input("Enter the filename with the full path\n")
    attributes = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    dt = BinaryDTree(attributes[0], depth)
    dt.buildtree(data=process_data(input_file, attributes), features=attributes[1:7])
    test_file = input_file.replace("train", "test")
    predicted_data = dt.predict(process_data(test_file, attributes))
    print("Accuracy: ", find_error(predicted_data, 'class', 'predicted'))
    dt.root.tree_display()