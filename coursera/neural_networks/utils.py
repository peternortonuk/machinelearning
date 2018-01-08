import numpy as np
from unittest import TestCase


def sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """

    s = sigmoid(x)
    ds = s * (1 - s)

    return ds


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)

    return v

def images2matrix(images):
    """
    Argument:
    X -- a numpy array of shape (image_number, length, height, depth)
        eg X.shape = (209, 64, 64, 3)
        means there are 209 images of 64 x 64 pixels one for each of RGB

    Returns:
    X -- a numpy array with the same number of images
        the -1 argument means flatten and infer the length
        so returns a matrix of shape (image_number, length x height x depth)
    """
    X = images.reshape(images.shape[0], -1).T

    return X

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """

    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    # Divide x by its norm.
    x = x / x_norm

    return x


def softmax(x):
    """Calculates the softmax by row of the input x.
    softmax is a normalizing function that returns a matrix of the same shape

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum

    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function
    """

    loss = np.sum(np.abs(y - yhat))

    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function 
    np.dot(x,x) is the sum of the squares
    """

    loss = np.dot(y - yhat, y - yhat)

    return loss


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    return w, b


def calc_activation(w, b, X):
    """
    Compute activation layer

    Arguments:
    w -- A vector of weights
    b -- A constant
    X -- A matrix of features

    Return:
    A -- A vector of predictions
    """

    A = sigmoid(np.dot(w.T, X) + b)

    return A

def calc_cost_function(X, A, Y):
    """
    Compute cost function

    Arguments:
    X -- A matrix of features
    A -- A vector of predictions
    Y -- A vector of truth

    Return:
    J -- cost function
    """
    m = X.shape[1]
    J = (-1.0 / m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

    return J

def calc_dJdw(X, A, Y):
    m = X.shape[1]
    dJdw = (1.0 / m) * np.dot(X, (A-Y).T)
    return dJdw

def calc_dJdb(X, A, Y):
    m = X.shape[1]
    dJdb = (1.0 / m) * np.sum(A - Y)
    return dJdb


def forward_propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    A = calc_activation(w, b, X)

    J = calc_cost_function(X, A, Y)

    return A, J

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
A, cost = forward_propagate(w, b, X, Y)
print ("cost = " + str(cost))
#TestCase.assertAlmostEqual(cost, 5.80154531939, places=10, msg=None, delta=None)


def backward_propagate(X, A, Y):
    dJdw = calc_dJdw(X, A, Y)
    dJdb = calc_dJdb(X, A, Y)

    return dJdw, dJdb

dJdw, dJdb = backward_propagate(X, A, Y)
print dJdw
print dJdb
import pdb; pdb.set_trace()
pass