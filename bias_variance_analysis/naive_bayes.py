class naive_bayes:
    def __init__(self, data, feature_type):
        self.class_prior = None
        self.feature_prob = None
        self.feature_type = feature_type
        self.target = list(self.feature_type)[-1]

    def train(self, x_train, y_train):
        """updates the model parameters from the training data
        x_train: A pandas data frame, feature predictors
        y_train: A pandas Series, class labels of the training data"""

        # Finding the prior class probabilities

        if self.feature_type[self.target] is "continuous":
            print("Naive Bayes is a classification model. Cannot work with continuous data")
        else:
            self.class_prior = y_train.value_counts(normalize=True)
        # print(self.class_prior)

        self.feature_type.pop("y")

        for feature in self.feature_type:
            print(feature)