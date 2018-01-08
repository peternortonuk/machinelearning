import load_cat_dataset

path = r'C:\dev\code\machinelearning\coursera\neural_networks\datasets'
file_train = r'train_catvnoncat.h5'
file_test = r'test_catvnoncat.h5'


train_dataset, test_dataset = load_cat_dataset(path, file_train, file_test)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = build_cat_dataset(train_dataset, test_dataset)

