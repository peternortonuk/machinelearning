import numpy as np

a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b


'''
"np.dot(a,b)" performs a matrix multiplication on a and b, whereas "a*b" performs an element-wise multiplication
'''



# ================================
# question 4

# this demonstrates broadcasting to achieve the same shape
# so shape of results is (2, 3)

a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
print c.shape


# ================================
# question 5

a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)

# this doesnt work its an elementwise operator
#c = a*b


# ================================
# question 7

a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = np.dot(a, b)
print c.shape


# ================================
# question 8

a = np.random.randn(3, 4)
b = np.random.randn(4, 1)
c = np.zeros((3, 4))

for i in range(3):
    for j in range(4):
        c[i][j] = a[i][j] + b[j]
print c
d = a + b.T
print d
print c == d

# ================================
# question 9

# this demonstrates broadcasting and elementwise multiplication
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b

print c

pass