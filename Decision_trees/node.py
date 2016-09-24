class Node:
    def __init__(self, name, child):
        self.name = name
        self.child = {}
        if child is not None:
            for kid in child:
                self.child[kid] = None
        self.parent = None
        self.label = None
        self.depth = None
