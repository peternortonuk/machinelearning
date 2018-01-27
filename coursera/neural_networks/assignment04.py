import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import os

from assignment01_utils import load_data
from assignment02_utils import predict
from assignment04_utils import two_layer_model, L_layer_model
from dnn_app_utils_v2 import print_mislabeled_images

# ==========================================
# run which model?

# set constants
TWO_LAYER_MODEL = 'two_layer_model'
L_LAYER_MODEL = 'L_layer_model'

# choose here
model_selection = L_LAYER_MODEL

# ==========================================
# configure images

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# ==========================================
# image file definitions
path = r'C:\dev\code\machinelearning\coursera\neural_networks\datasets'
file_train = r'train_catvnoncat.h5'
file_test = r'test_catvnoncat.h5'
train_x_orig, train_y, test_x_orig, test_y, classes = load_data(path, file_train, file_test)


# ==========================================
# explore dataset

# example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# shape of the data
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# ==========================================
# prepare the data

# reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# ==========================================
# two layer model


if model_selection == TWO_LAYER_MODEL:
    n_x = 12288     # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    # train the model
    parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

    # predict based on parameters
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)

print_mislabeled_images(classes, test_x, test_y, predictions_test)


# ==========================================
# five layer model


if model_selection == L_LAYER_MODEL:
    layers_dims = [12288, 20, 7, 5, 1]

    # train the model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

    # predict based on parameters
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)

print_mislabeled_images(classes, test_x, test_y, predictions_test)


# ==========================================
# my own images


my_image = "cat_picture.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

myfilepath = os.path.join(path, my_image)
image = np.array(ndimage.imread(myfilepath, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")



