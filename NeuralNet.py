#####################################################################################################################
#   CS 6375 - Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delat12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class NeuralNet:
    def __init__(self, train_dataset, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 4))








    # sigmoid activation derivative and sigmoid function

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)





    # tanh activation derivative and tanh funtion

    def __activation(self, x, activation="tanh"):
        if activation == "tanh":
            self.__tanh(self, x)

    def __activation_derivative(self, x, activation="tanh"):
        if activation == "tanh":
            self.__tanh_derivative(self, x)

    def __tanh(self, x):
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return -1 + (2 * self.__sigmoid(2 * x))

    def __tanh_derivative(self, x):
        # return 1 - (np.square(x))
        return 1 - ((self.__tanh(x)) * self.__tanh(x))





    # relu activation, relu and derivative function

    def __activation(self, x, activation="relu"):
        if activation == "relu":
            self.__relu(self, x)

    def __activation_derivative(self, x, activation="relu"):
        if activation == "relu":
            self.__relu_derivative(self, x)

    def __relu(self, x):
        return np.maximum(x, 0)

    def __relu_derivative(self, x):
        if x.all() > 0:
            return 1
        else:
            return 0





    # Below is the training function

    def train(self, max_iterations = 1000, learning_rate = 0.05, activation = "sigmoid"):
        print ("The activation function used is: " + activation)
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation):
        # pass our inputs through our neural network

        # for input layer
        in1 = np.dot(self.X, self.w01)
        if activation == "sigmoid":
            self.X12 = self.__sigmoid(in1)
        elif activation == "tanh":
            self.X12 = self.__tanh(in1)
        elif activation == "relu":
            self.X12 = self.__relu(in1)

        # for hidden layer 1
        in2 = np.dot(self.X12, self.w12)
        if activation == "sigmoid":
            self.X23 = self.__sigmoid(in2)
        elif activation == "tanh":
            self.X23 = self.__tanh(in2)
        elif activation == "relu":
            self.X23 = self.__relu(in2)

        # for hidden layer 2
        in3 = np.dot(self.X23, self.w23)
        out = 0
        if activation == "sigmoid":
            out = self.__sigmoid(in3)
        elif activation == "tanh":
            out = self.__tanh(in3)
        elif activation == "relu":
            out = self.__relu(in3)

        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
            self.deltaOut = delta_output
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
            self.deltaOut = delta_output
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
            self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
            self.delta23 = delta_hidden_layer2
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
            self.delta23 = delta_hidden_layer2
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
            self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation):

        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
            self.delta12 = delta_hidden_layer1
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
            self.delta12 = delta_hidden_layer1
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
            self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self, activation="sigmoid"):

        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer
        elif activation == "relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))
            self.delta01 = delta_input_layer

    def predict(self, test, activation, header = True):

        self.X = test.iloc[:, 0: 14].values
        self.y = test.iloc[:, 14].values
        out = self.forward_pass(activation)
        test_error = (0.5 * (np.power((out - self.y), 2)))
        test_error = np.sum(test_error)
        return test_error


def preprocess(in_data):
    # Some data strings have whitespaces in front of them so delimiter used is 0 or more whitespaces followed by a ','.
    df = pd.read_csv(in_data, delimiter=' *, *', engine='python')
    # Detecting the missing values which are marked as '?' and replacing with 'nan'
    df = df.where(df != '?', np.NaN)
    df = df.where(df != '<=50K.', '<=50K')
    df = df.where(df != '>50K.', '>50K')
    # Dropping the 'nan' i.e. missing values
    df = df.dropna()
    # using Label Encoder to convert categorical data into numerical data fit for the Regression model
    lb_make = LabelEncoder()
    # Selecting the variables which have categorical values
    columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    for i in columns:
        df.iloc[:, i] = lb_make.fit_transform(df.iloc[:, i])
    # All the variables are of d-type initially; converting to numeric values
    for i in range(0, 15):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i])
    # Converting the panda DataFrame to a numpy MultiDimensional Array
    return df


if __name__ == "__main__":

    dataset = 'Adult_census_income.csv'
    data = preprocess(dataset)
    data = data.sample(frac=1).reset_index(drop=True)
    split = int(len(data) * .8)
    train_data = data.iloc[0:split, :]
    test_data = data.iloc[split:, :]
    neural_network = NeuralNet(train_data)
    activation = "relu"
    neural_network.train(2000, 0.05, activation)
    testError = neural_network.predict(test_data, activation)
    print ("\n")
    print ("The test error for the split is: " + str(testError))
