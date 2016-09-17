class Node:
    def __init__(self, name, child):
        self.name = name
        self.child = {}
        for kid in child:
            self.child[kid] = None
        self.parent = None
        self.label = None
        self.depth = None
