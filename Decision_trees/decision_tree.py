import pandas as pd
import math
import operator
from Decision_trees import node


class DecisionTree:
    """Class for specifying the properties of decision tree"""
    def __init__(self, a, b):
        self.depth = a
        self.features = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
        self.input = pd.read_table(b, delim_whitespace=True, names=self.features, header=None)

    def display(self):
        print("Depth: ", self.depth)
        print("Input file: ", self.input)
        print("Features: ", self.features)

    def find_node(self, data, predictor):
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
        best_attribute = sorted_gain[0][0]
        attribute_values = list(set(data[best_attribute]))
        current_node = node.Node(best_attribute, attribute_values)
        return current_node

    def build_tree(self):
        input_features = self.features[1:7]
        data = self.input
        root = self.find_node(data, input_features)
        root.depth = 1
        current_node = root
        print(root.name)
        # input_features.remove(root.name)
        for key in current_node.child:
            sub_data = data[data[current_node.name].isin([key])]
            if self.depth == current_node.depth:
                current_node.label = set_label(sub_data)
                print(key, current_node.label)
                continue
            input_features.remove(current_node.name)
            current_node.child[key] = self.find_node(sub_data, input_features)
            child_node = current_node.child[key]
            child_node.parent = current_node
            child_node.depth = current_node.depth + 1
            del data[current_node.name]
            print(current_node.child[key].parent.name, key, current_node.child[key].name)
            current_node = child_node


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


if __name__ == '__main__':
    print("Program to implement the decision tree without using package")
    depth = eval(input("Enter the desired depth of the decision tree\n"))
    path = input("Enter the filename with the full path\n")
    dt = DecisionTree(depth, path)
    # dt.display()
    dt.build_tree()


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



