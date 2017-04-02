import math
# from bias_variance_analysis import pd
import pandas as pd


class LogisticRegression(object):
    """A blue print of the Logistic Regression model. It has the following parameters

    Attributes:
        data_loader: An object of the DataLoader class
        x: Contains the modified input data
        """
    def __init__(self, train_data, test_data):
        dim = len(train_data.features)
        self.y = train_data.data.iloc[:, dim]
        self.x = train_data.data.iloc[:, :dim]
        self.x_t = test_data.data.iloc[:, :dim]
        self.y_t = test_data.data.iloc[:, dim]
        self.n_t = len(self.x_t)
        self.n = len(self.x)
        self.modify_data()
        self.d = len(self.x.columns)
        self.weight = [1 / self.d for _ in range(self.d)]
        self.theta = 0.005
        self.slope = None
        self.probability = None
        self.error = None
        self.mean_error = 0
        self.num_iterations = 100
        self.prediction = []
        self.accuracy = None

    def print_parameters(self):
        print("\nNumber of weights learned: ", self.d)
        print("Number of input data points: ", self.n)
        print("Learned weights:\n{}".format(self.weight))
        print("Current slope value:", self.slope)
        # print(self.x.iloc[0])

    def logistic(self, data_point):
        """Calculates the probability P(X=0) by calculating the value of the logistic function
        It requires the current data point as its parameter to find the logistic value.
        """
        data_point = list(data_point)
        tot = 0
        for i in range(self.d):
            tot += self.weight[i] * data_point[i]

        p = 1/(1+math.exp(-tot))

        return p

    def learn(self):
        """Learns the weight vector from the data"""
        # self.print_parameters()
        self.find_probability()
        self.find_error()
        iteration = 0
        while self.mean_error > 0.1 and iteration < self.num_iterations:
            self.find_slope()
            self.update_weights()
            # self.print_parameters()
            self.find_probability()
            self.find_error()
            iteration += 1

    def predict(self):
        self.find_probability("test")
        for _ in range(self.n_t):
            # print(_)
            if self.probability[_] >= 0.5:
                self.prediction.append(1)
            else:
                self.prediction.append(0)
        print(self.prediction)
        print(list(self.y_t))

    def find_probability(self, indicator=None):
        probability = []
        if indicator == "test":
            for _ in list(self.x_t.index):
                probability.append(self.logistic(self.x_t.loc[_, :]))
        else:
            for _ in list(self.x.index):
                probability.append(self.logistic(self.x.loc[_, :]))

        self.probability = probability

    def find_accuracy(self):
        correct = 0
        for _ in range(self.n_t):
            if self.prediction[_] == self.y_t[_]:
                correct += 1
        self.accuracy = correct / self.n_t
        print("Accuracy: {}".format(self.accuracy))

    def find_error(self):
        error = []
        for i in range(self.n):
            error.append(self.y[i] - self.probability[i])
        self.error = error
        for item in error:
            self.mean_error += abs(item)
        self.mean_error /= self.n
        print("Error value:", self.mean_error)

    def find_slope(self):
        slope = []
        for i in range(self.d):
            feature_slope = 0
            feature = self.x.iloc[:, i]
            for j in range(self.n):
                feature_slope += feature[j]*self.error[j]
            slope.append(feature_slope)
        self.slope = slope

    def modify_data(self):
        """Modifies the data to have only numbers by binarization of the data if the
        data set contains categorical variables"""

        self.x = pd.get_dummies(self.x)
        self.x_t = pd.get_dummies(self.x_t)

        # Adding a column of ones to the data
        # self.x['ones'] = pd.Series([1 for _ in range(self.n)], index=list(self.x.index))
        # self.x_t['ones'] = pd.Series([1 for _ in range(self.n_t)], index=list(self.x_t.index))

        # Converting "yes" & "no" values to 1 and 0 respectively
        temp_y = replace(list(self.y), 'no', 0)
        self.y = replace(temp_y, 'yes', 1)

        temp_y_t = replace(list(self.y_t), 'no', 0)
        self.y_t = replace(temp_y_t, 'yes', 1)

        self.x = (self.x - self.x.min())/(self.x.max() - self.x.min())
        self.x_t = (self.x_t - self.x_t.min())/(self.x_t.max() - self.x_t.min())

    def update_weights(self):
        for _ in range(self.d):
            self.weight[_] += self.theta * self.slope[_]

    def plot_auc_roc_curve(self):
        pass


def replace(array, x, y):
    for i in range(len(array)):
        if array[i] == x:
            array[i] = y
    return array
