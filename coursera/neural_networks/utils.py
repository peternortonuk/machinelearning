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
    A -- predicted output
    J -- cost of prediction
    """

    A = calc_activation(w, b, X)
    J = calc_cost_function(X, A, Y)

    return A, J


def backward_propagate(X, A, Y):
    dJdw = calc_dJdw(X, A, Y)
    dJdb = calc_dJdb(X, A, Y)

    return dJdw, dJdb


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    for i in range(num_iterations):

        A, cost = forward_propagate(w, b, X, Y)
        dJdw, dJdb = backward_propagate(X, A, Y)

        w = w - learning_rate * dJdw
        b = b - learning_rate * dJdb

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dJdw,
             "db": dJdb}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = calc_activation(w, b, X)

    for i in range(A.shape[1]):
        # if modelled probability is greater than 0.5 then it's a cat
        idx = np.where(A >= 0.5)
        Y_prediction[idx] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000,
          learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###

    # initialize parameters with zeros (1 line of code)
    dim = X_train.shape[0] + X_test.shape[0]
    import pdb; pdb.set_trace()
    w, b = initialize_with_zeros(dim)

    # Gradient descent (1 line of code)
    parameters, grads, costs = None

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (2 lines of code)
    Y_prediction_test = None
    Y_prediction_train = None

    ### END CODE HERE ###

    # Print train/test Errors
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

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
