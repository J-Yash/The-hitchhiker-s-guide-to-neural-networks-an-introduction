import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Activation function(Sigmoid)


def sigmoid(x):
    return (1/(1 + np.exp(-x)))

# Reading data and preprocessing it so it is in the form that can be used in our neural network


def preprocess():
    # Reading data from the CSV file
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train.csv'),
                       header=0, delimiter=",", quoting=3)
    data_np = data.values[:, [0, 1, 4, 5, 6, 7, 11]]
    X = data_np[:, 1:]
    Y = data_np[:, [0]]

    # Converting string values into numeric values
    for i in range(X.shape[0]):
        if X[i][1] == 'male':
            X[i][1] = 1
        else:
            X[i][1] = 2

        if X[i][5] == 'C':
            X[i][5] = 1
        elif X[i][5] == 'Q':
            X[i][5] = 2
        else:
            X[i][5] = 3

        if math.isnan(X[i][2]):
            X[i][2] = 0
        else:
            X[i][2] = int(X[i][2])

    # Creating training and test sets
    X_train = np.array(X[:624, :].T, dtype=np.float64)
    X_train[2, :] = X_train[2, :]/max(X_train[2, :])  # Normalizing Age
    Y_train = np.array(Y[:624, :].T, dtype=np.float64)

    X_test = np.array(X[624:891, :].T, dtype=np.float64)
    X_test[2, :] = X_test[2, :]/max(X_test[2, :])  # Normalizing Age
    Y_test = np.array(Y[624:891, :].T, dtype=np.float64)
    return X_train, Y_train, X_test, Y_test

# Initializing Weights and Biases


def weight_initialization():
    W1 = np.random.randn(4, 6)
    b1 = np.zeros([4, 1])

    W2 = np.random.randn(1, 4)
    b2 = np.zeros([1, 1])

    return W1, b1, W2, b2

# Forward Propagation Step


def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return A2, A1

# Computing the loss/cost


def compute_cost(A2, Y):
    cost = -(np.sum(np.multiply(Y, np.log(A2)) +
                    np.multiply(1-Y, np.log(1-A2)))/Y.shape[1])
    return cost

# Backpropagation step and updating weights and biases(Gradient Descent)


def back_propagation_and_weight_updation(A2, A1, X_train, W2, W1, b2, b1, learning_rate=0.01):
    dZ2 = A2 - Y_train
    dW2 = np.dot(dZ2, A1.T)/X_train.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True)/X_train.shape[1]
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X_train.T)/X_train.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True)/X_train.shape[1]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, W2, b1, b2


if __name__ == "__main__":

    # STEP 1: LOADING AND PREPROCESSING DATA
    X_train, Y_train, X_test, Y_test = preprocess()

    # STEP 2: INITIALIZING WEIGHTS AND BIASES
    W1, b1, W2, b2 = weight_initialization()

    # Setting the number of iterations for gradient descent
    num_of_iterations = 50000

    all_costs = []

    for i in range(0, num_of_iterations):

        # STEP 3: FORWARD PROPAGATION
        A2, A1 = forward_propagation(W1, b1, W2, b2, X_train)

        # STEP 4: COMPUTING COST
        cost = compute_cost(A2, Y_train)
        all_costs.append(cost)

        # STEP 5: BACKPROPAGATION AND PARAMETER UPDATTION
        W1, W2, b1, b2 = back_propagation_and_weight_updation(
            A2, A1, X_train, W2, W1, b2, b1)

        if i % 1000 == 0:
            print("Cost after iteration "+str(i)+": "+str(cost))

    # STEP 6: EVALUATION METRICS

    # To Show accuracy of our training set
    A2, _ = forward_propagation(W1, b1, W2, b2, X_train)
    pred = (A2 > 0.5)
    print('Accuracy for training set: %d' % float((np.dot(Y_train, pred.T) +
                                                   np.dot(1-Y_train, 1-pred.T))/float(Y_train.size)*100) + '%')

    # To show accuracy of our test set
    A2, _ = forward_propagation(W1, b1, W2, b2, X_test)
    pred = (A2 > 0.5)
    print('Accuracy for test set: %d' % float((np.dot(Y_test, pred.T) +
                                               np.dot(1-Y_test, 1-pred.T))/float(Y_test.size)*100) + '%')

    # STEP 7: VISUALIZING EVALUATION METRICS

    # Plot graph for gradient descent
    plt.plot(np.squeeze(all_costs))
    plt.ylabel('Cost')
    plt.xlabel('Number of Iterations')
    plt.show()
