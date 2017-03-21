import sklearn.linear_model as lm
import pandas as pd

model = lm.LogisticRegression(class_weight='balanced')

input_path = "/media/gowtham/D6DC3802DC37DC05/Datasets/Bank marketing"
input_data = pd.read_csv(input_path + "/bank-full.csv", delimiter=";")

x = input_data.iloc[:, :16]
y = input_data.y

print("Feature variables are:\n", x.head())
# print("Target variable is:\n", y)

model.fit(x,y)
print(model.get_params())

test_data = pd.read_csv(input_path + "/bank.csv", delimiter=";")

test_x = test_data.iloc[:, :16]
test_y = input_data.y

model.predict(test_x)

model.score(test_x, test_y)