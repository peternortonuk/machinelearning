# ==========================================================================
# ==========================================================================
# http://scikit-learn.org/stable/tutorial/statistical_inference/index.html


# ==========================================================================
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html



# Nearest neighbor and the curse of dimensionality

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)


# split iris data in train and test data
# a random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)


print(knn.predict(iris_X_test))

print(iris_y_test)

print('break')


# ==========================================================================
# Linear model: from regression to sparsity

from sklearn import linear_model

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)

# The mean square error
mse = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
print(mse)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
score = regr.score(diabetes_X_test, diabetes_y_test)
print(score)


print('break')

# ==========================================================================
# SVM

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)

print('break')



# ==========================================================================
# http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

# Cross-validation generators

from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


# Kfold on a simple demo dataset
X = ["a", "a", "b", "c", "c", "c"]
# split the dataset into three equal groups
k_fold = KFold(n_splits=3)
# returns the indices of the dataset for the required train and test subset
for train_indices, test_indices in k_fold.split(X):
     print('Train: %s | test: %s' % (train_indices, test_indices))


# Kfold on a more realistic demo dataset with two factors
# define the splits on the x array and apply the result to both x and y
# this works because the indices apply to the rows of the np.array
# and if we don't specify the other dimensions, we get all of them
# ie it doesnt matter that x is 2D and y is 1D
X1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
X2 = ["z", "y", "x", "w", "v", "u", "t", "s", "r"]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
x_array = np.asarray([X1, X2]).T
y_array = np.asarray(y)
print('x_array: %s | y_array: %s' % (np.shape(x_array), np.shape(y_array)))

k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(x_array):
     print('Train indices: %s | test indices: %s' % (train_indices, test_indices))
     print('Train x: %s | Train y: %s' % (x_array[train_indices], y_array[train_indices]))
     print('Test x: %s | test y: %s' % (x_array[test_indices], y_array[test_indices]))

# And now the real thing
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print('X_digits: %s | y_digits: %s' % (np.shape(X_digits), np.shape(y_digits)))


k_fold = KFold(n_splits=3)
for train_indices, test_indices in k_fold.split(X_digits):
     print('Train: %s | test: %s' % (train_indices, test_indices))

svc = svm.SVC(C=1, kernel='linear')
kfold = KFold(n_splits=3)
result = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
          for train, test in k_fold.split(X_digits)]
print result