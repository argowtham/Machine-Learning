class LogisticRegression(object):
    """A blue print of the Logistic Regression model. It has the following parameters

    Attributes:
        data: An object of the DataLoader class"""
    def __init__(self, data):
        self.weight = [1 for i in range(len(data.features))]

    def print_parameters(self):
        print("Number of weights learnded: ", len(self.weight))
        print("Learned weights: ", self.weight)
