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


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
train_set_x, test_set_x = prepare_data()

d = utils.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)