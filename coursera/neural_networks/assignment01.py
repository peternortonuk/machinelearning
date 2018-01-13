import assignment01_utils
import matplotlib.pyplot as plt

# image file definitions
path = r'C:\dev\code\machinelearning\coursera\neural_networks\datasets'
file_train = r'train_catvnoncat.h5'
file_test = r'test_catvnoncat.h5'

# load data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = assignment01_utils.load_data(path,
                                                                                                   file_train,
                                                                                                   file_test)

# reshape the image data into feature vectors
train_set_x, test_set_x = assignment01_utils.prepare_data(train_set_x_orig,
                                                          test_set_x_orig)

# run the model
d = assignment01_utils.model(train_set_x,
                             train_set_y,
                             test_set_x,
                             test_set_y,
                             num_iterations = 2000,
                             learning_rate = 0.005,
                             print_cost = True)

# extract the results
activation = d["Activation_test"][0]
y_prediction = d["Y_prediction_test"]

# get order of activation value
sorted_index = activation.argsort()

# sort the data
activation = activation[sorted_index]
test_set_x_orig = test_set_x_orig[sorted_index]
y_prediction = y_prediction[:,sorted_index]

# plot images
for index in range(len(test_set_x_orig)):
    # plot the image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(test_set_x_orig[index])

    # annotate
    text_Y = classes[y_prediction[0,index].astype(int)].decode("utf-8")
    text_A = str(activation[index])
    text = text_Y + ": cat at P(" + text_A[:5] + ")"
    ax.annotate(text, fontsize=20, xy=(2, 1), xytext=(0, -1.5))

    plt.show()
