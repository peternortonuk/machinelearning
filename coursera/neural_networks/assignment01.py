import numpy as np
from assignment_utils import load_cat_dataset, build_cat_dataset
import utils
import assignment_utils
import matplotlib.pyplot as plt

path = r'C:\dev\code\machinelearning\coursera\neural_networks\datasets'
file_train = r'train_catvnoncat.h5'
file_test = r'test_catvnoncat.h5'

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = assignment_utils.load_data(path, file_train, file_test)
train_set_x, test_set_x = assignment_utils.prepare_data(train_set_x_orig, test_set_x_orig)
d = utils.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


for index in range(test_set_x.shape[1]):
    num_px = test_set_x_orig.shape[1]
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    print (classes[d["Y_prediction_test"][0,index].astype(int)].decode("utf-8"))
    plt.show()
