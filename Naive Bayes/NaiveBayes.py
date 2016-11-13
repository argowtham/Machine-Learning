import pandas as pd
import math


def process_data(file, sep=None):
    return pd.read_csv(file, sep=sep, header=None)


def split_data(frame, fraction=None):
    if fraction is None:
        fraction = 0.67
    train = frame.sample(frac=fraction, replace=False)
    test = frame.drop(train.index)
    return train, test


def separate_by_class(frame):
    """Returns the dataset seperated as per the attribute values of class variable"""
    """Assuming the last column in dataframe as class variable"""
    target = list(frame.columns)[-1]
    unique_values = list(set(list(frame[target])))
    split_frame = []
    for value in unique_values:
        split_frame.append(frame[frame[target] == value])
    return split_frame


def find_mean(array):
    return sum(array)/float(len(array))


def find_sd(array):
    mean = find_mean(array)
    variance = 0
    for number in array:
        variance += (mean - number) ** 2
    variance /= float(len(array)-1)
    sd = math.sqrt(variance)
    return mean, sd


def find_parameters(frame):
    features = list(frame.columns)
    del features[-1]
    params = {}
    for feature in features:
        params[feature] = {}
        values = list(frame[feature])
        params[feature]['mean'], params[feature]['sd'] = find_sd(values)
    return params


def calculate_probability(value, mean, sd):
    exponent = math.exp(-(math.pow(value-mean, 2))/(2*math.pow(sd, 2)))
    probability = exponent / (math.sqrt(2 * math.pi * sd))
    return probability


def predict(frame, params):
    frame['predicted'] = -1
    for i in range(len(frame)):
        index = frame.iloc[i].name
        record = list(frame.iloc[i])
        del record[-2]
        probability = {}
        for key in params:
            probability[key] = 1.0
            for j in range(len(record)-1):
                probability[key] *= calculate_probability(record[j], params[key][j]['mean'], params[key][j]['sd'])
        max_prob = 0
        for k in probability:
            if probability[k] > max_prob:
                max_prob = probability[k]
                prediction = k
        frame.set_value(index, 'predicted', prediction)
    return frame


def find_error(frame, a, b, return_param=None):
    actual = list(frame[a])
    predicted = list(frame[b])
    error = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    for i in range(len(actual)):
        if not actual[i] == predicted[i]:
            error += 1
            if actual[i] == 1:
                fn += 1
            else:
                fp += 1
        else:
            if actual[i] == 1:
                tp += 1
            else:
                tn += 1
    confusion_matrix = pd.DataFrame([[tp, fn], [fp, tn]], columns=['Yes', 'No'], index=['Yes', 'No'])
    print(confusion_matrix)
    print("Error rate: ", error/len(actual))
    if return_param == "Error":
        return error/len(actual)
    else:
        return round((1-(error/len(actual)))*100, 2)


if __name__ == "__main__":
    print("=============================================")
    print("\t\t\tNaive Bayes Algorithm")
    print("=============================================")
    file_path = input("Enter the path of the file\n")
    data = process_data(file_path, sep=',')
    train_data, test_data = split_data(data, fraction=0.67)
    split_data = separate_by_class(train_data)
    parameters = {}
    target = list(data.columns)[-1]
    for chunk in split_data:
        parameters[chunk.iloc[0][target]] = find_parameters(chunk)
    predicted_data = predict(test_data, parameters)
    print("Accuracy: ", find_error(predicted_data, predicted_data[target], 'predicted'))