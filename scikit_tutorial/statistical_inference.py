# ==========================================================================
# http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html


# ==========================================================================
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
