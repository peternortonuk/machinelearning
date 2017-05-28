from sklearn import datasets, svm
import matplotlib.pyplot as plt


# ==================================================================
# http://scikit-learn.org/stable/tutorial/basic/tutorial.html

# ==================================================================
# ==================================================================
# digits dataset

'''
images, 1797: 8x8 array of pixels
data, 1797: 64x1 vector of pixels
target, 1797: digit values
target_names, 10: digit values
DESCR' 2014: the help text
'''

digits = datasets.load_digits()
print(digits.keys())
print(digits.DESCR)

def summarise_dict(d):
    for k, v in d.items():
        print(k, len(v))
summarise_dict(digits)

select_keys = ['images', 'data', 'target']
for i in select_keys:
    print(i, digits[i][0].shape, digits[i][0])


# =============================
# display the digits


select_images = digits.images[-1:]
select_target = digits.target[-1:]

for image, target in zip(select_images, select_target):
    plt.figure()
    # clear axes and display updated title
    plt.cla()
    image_text = 'Digit: ' + str(target)
    plt.suptitle(image_text)
    # plot the image of the digit
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # show the plot and don't block execution
    plt.show(block=False)
    # wait
    plt.pause(0.8)
# and then close all figures
# plt.close('all')


# ==================================================================
# ==================================================================
# iris dataset

iris = datasets.load_iris()


# ==================================================================
# ==================================================================
# learning and predicting

from sklearn import svm

print('start')

# instantiate classifier
clf = svm.SVC(gamma=0.001, C=100.)

# train on all but the last one
clf.fit(digits.data[:-1], digits.target[:-1])

# predict; just the last one
print clf.predict(digits.data[-1:])

# the truth
print digits.target[-1]

print('end')

