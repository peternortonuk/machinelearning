from testCases_v2 import forward_propagation_test_case
from assignment01_utils import calc_activation, sigmoid
import numpy as np


A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)

print B.shape
pass




