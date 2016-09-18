import pandas as pd
import math
import operator
from Decision_trees import node


class DecisionTree:
    """Class for specifying the properties of decision tree"""
    def __init__(self, a, b, features):
        self.depth = a
        self.features = features
        self.input = b
        self.root = None

    def display(self):
        print("Depth: ", self.depth)
        print("Input file: ", self.input)
        print("Features: ", self.features)

    def build_tree(self):
        input_features = self.features[1:7]
        data = self.input
        self.root = find_node(data, input_features)
        self.root.depth = 1
        node_list = [self.root]
        for current_node in node_list:
            if current_node.depth <= self.depth:
                node_list.extend(find_child(data, current_node, current_node.child, input_features))
            #     print(current_node.name, current_node.depth)
            #     for kid in current_node.child:
            #         print(current_node.child[kid].parent.name, kid, current_node.child[kid].name)
            # else:
            #     print(current_node.name, current_node.depth, current_node.label)

    def predict(self, data):
        data['predicted'] = '-1'
        for i in range(len(data)):
            node_list = [self.root]
            for item in node_list:
                x = data.get_value(i, item.name)
                node_list.append(item.child[x])
                if item.label is not None:
                    data.set_value(i, 'predicted', item.label)
                    break
        return data

        # input_features.remove(root.name)
        # for key in current_node.child:
        #     sub_data = data[data[current_node.name].isin([key])]
        #     if self.depth == current_node.depth:
        #         current_node.label = set_label(sub_data)
        #         print(current_node.name, key, current_node.label)
        #         continue
        #     input_features.remove(current_node.name)
        #     current_node.child[key] = self.find_node(sub_data, input_features)
        #     child_node = current_node.child[key]
        #     child_node.parent = current_node
        #     child_node.depth = current_node.depth + 1
        #     del data[current_node.name]
        #     print(current_node.child[key].parent.name, key, current_node.child[key].name)
        #     current_node = child_node


def find_node(data, predictor):
        gain, unique_values = {}, []
        for feature in predictor:
            entropy, entropy_sum = calculate_entropy(data), 0
            unique_values = list(set(data[feature]))
            for value in unique_values:
                sub_data = data[data[feature].isin([value])]  # Splitting the data into subsets
                # based on unique values by attribute 'feature'
                entropy_sum += (len(sub_data)/len(data)) * calculate_entropy(sub_data)
            gain[feature] = calculate_gain(entropy, entropy_sum)
        sorted_gain = sorted(gain.items(), key=operator.itemgetter(1))
        sorted_gain.reverse()
        print(sorted_gain)
        best_attribute = sorted_gain[0][0]
        attribute_values = list(set(data[best_attribute]))
        current_node = node.Node(best_attribute, attribute_values)
        return current_node


def find_child(data, current_node, possible_values, features):
    edited_features = list(features)
    parents = [current_node]
    # edited_features.remove(current_node.name)
    for i in parents:
        edited_features.remove(i.name)
        if i.parent is None:
            break
        parents.append(i.parent)
    child_list = []
    for value in possible_values:
        sub_data = data[data[current_node.name].isin([value])]
        for attribute in parents:
            del sub_data[attribute.name]
        current_node.child[value] = find_node(sub_data, edited_features)
        child_node = current_node.child[value]
        child_list.append(child_node)
        child_node.parent = current_node
        child_node.depth = current_node.depth + 1
        if child_node.depth > dt.depth:
            child_node.label = set_label(sub_data)
    return child_list


def set_label(data):
    classifier = 'class'  # Indicates the name of the column which has to be classified
    p_pos = list(data[classifier]).count(1)
    p_neg = list(data[classifier]).count(0)
    if p_pos > p_neg:
        return 1
    else:
        return 0


def calculate_entropy(data):
    total = len(data)
    classifier = 'class'  # Indicates the name of the column which has to be classified
    p_pos = list(data[classifier]).count(1)/total
    p_neg = list(data[classifier]).count(0)/total

    if p_pos == 0:
        x = 1
    else:
        x = - p_pos*math.log(p_pos, 2)

    if p_neg == 0:
        y = 1
    else:
        y = - p_neg * math.log(p_neg, 2)

    entropy = x + y
    return entropy


def calculate_gain(entropy1, entropy2):
    return entropy1 - entropy2


def process_data(file_name, attr):
    return pd.read_table(file_name, delim_whitespace=True, names=attr, header=None)


def find_error(data, a, b):
    actual = list(data[a])
    predicted = list(data[b])
    error = 0
    for i in range(len(actual)):
        if not actual[i] == predicted[i]:
            error += 1
    return error

if __name__ == '__main__':
    print("Program to implement the decision tree without using package")
    depth = eval(input("Enter the desired depth of the decision tree\n"))
    path = input("Enter the filename with the full path\n")
    attributes = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    dt = DecisionTree(depth, process_data(path, attributes), attributes)
    # dt.display()
    dt.build_tree()
    test_data = input("Enter the test file path")
    predicted_data = dt.predict(process_data(test_data, attributes))
    print("Number of misclassified data points:", find_error(predicted_data, 'class', 'predicted'))

    # for file in train_files:
    #     data = pd.read_table(path+'\/'+file, delim_whitespace=True, names=features, header=None)
    #     # print(len(data))
    #     gain = {}
    #     for feature in input_features:
    #         entropy, entropy_sum = calculate_entropy(data), 0
    #         unique_values = list(set(data[feature]))
    #         for value in unique_values:
    #             sub_data = data[data[feature].isin([value])]  # Splitting the data into subsets
    #             # based on unique values by attribute 'feature'
    #             entropy_sum += (len(sub_data)/len(data)) * calculate_entropy(sub_data)
    #         gain[feature] = calculate_gain(entropy, entropy_sum)
    #     sorted_gain = sorted(gain.items(), key=operator.itemgetter(1))
    #     sorted_gain.reverse()
    #     print(sorted_gain[0][0])
    #     break



