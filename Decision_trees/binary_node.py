class BinaryNode:
    def __init__(self, name, label):
        self.feature = name
        self.label = label
        self.value = None
        self.depth = None
        self.parent = None
        self.true_child = None
        self.false_child = None

    def display(self):
        print("Feature associated with the node:", self.feature)
        print("Label:", self.label)
        print("Depth:", self.depth)
        print("Value:", self.value)
