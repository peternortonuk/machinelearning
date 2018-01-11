import numpy as np
from assignment_utils import load_cat_dataset, build_cat_dataset
import utils

path = r'C:\dev\code\machinelearning\coursera\neural_networks\datasets'
file_train = r'train_catvnoncat.h5'
file_test = r'test_catvnoncat.h5'


def load_data():

    train_dataset, test_dataset = load_cat_dataset(path, file_train, file_test)
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = build_cat_dataset(train_dataset, test_dataset)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def prepare_data(train_set_x_orig, test_set_x_orig):
    train_set_x_flatten = utils.images2matrix(train_set_x_orig)
    test_set_x_flatten = utils.images2matrix(test_set_x_orig)

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, test_set_x


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # initialize parameters with zeros
    dim = X_train.shape[0]
    w, b = utils.initialize_with_zeros(dim)

    # Gradient descent
    parameters, grads, costs = utils.optimize(w, b, X_train, Y_train, num_iterations,
                                        learning_rate, print_cost=False)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = utils.predict(w, b, X_test)
    Y_prediction_train = utils.predict(w, b, X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
train_set_x, test_set_x = prepare_data()

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)