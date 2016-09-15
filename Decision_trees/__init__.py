from sklearn import tree
import numpy as np

if __name__ == '__main__':
    file_name = "D:\OneDrive\IU\Fall'16\Applied Machine Learning\Homeworks\PA1\monks\monks-1.train"
    X = np.genfromtxt(file_name, usecols=(1, 2, 3, 4, 5, 6))
    Y = np.genfromtxt(file_name, usecols=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    with open("decision_tree.dot", 'w') as file:
        tree.export_graphviz(clf, out_file=file)
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data)

