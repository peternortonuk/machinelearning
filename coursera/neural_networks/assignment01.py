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


# extract the results
activation = d["Activation_test"][0]
y_prediction = d["Y_prediction_test"]

# get order of activation value
sorted_index = activation.argsort()

# sort the data
activation = activation[sorted_index]
test_set_x = test_set_x[:, sorted_index]
y_prediction = y_prediction[:,sorted_index]

# plot images
num_px = test_set_x_orig.shape[1]
for index in range(test_set_x.shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    text_Y = classes[y_prediction[0,index].astype(int)].decode("utf-8")
    text_A = str(activation[index]*100)
    text = text_Y + " at " + text_A[:5] + " percent"
    ax.annotate(text, fontsize=20, xy=(2, 1), xytext=(3, -1.5))
    plt.show()
