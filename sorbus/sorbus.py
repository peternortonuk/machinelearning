
import pandas as pd
import numpy as np
import os
from sklearn import neighbors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

pathname = r'C:\dev\code\machinelearning\data'
filename = r'Soraria measurements for Petra Guy - updated for knn.xlsx'
sheetnames = [r'intermedia - upload - leaf',
              r'hibernica - upload - leaf',
              r'minima - upload - leaf',
              ]
train_factor = 0.9
pathfile = os.path.join(pathname, filename)

# import and concatenate into one df
list_of_df = []
for sheetname in sheetnames:
    df = pd.read_excel(pathfile, sheetname=sheetname, )
    list_of_df.append(df)
df_all = pd.concat(list_of_df, axis=0, join='inner', copy=True)

# calculate the train and test row count and slicer
df_count = df_all.groupby(by='Species').count().iloc[:,0]
total_row_count = df_count.min()
train_row_count = int(total_row_count*train_factor)
test_row_count = total_row_count - train_row_count

# add a column with the usage type and label as test versus train
unique_species = df_all['Species'].unique()
for species in unique_species:
    species_mask = df_all['Species'] == species
    # monumental faff due to assignment fail when using chained indexing
    # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    # train
    ix = df_all.index[species_mask][:train_row_count]
    df_all.loc[ix, 'Usage'] = 'train'
    # test
    ix = df_all.index[species_mask][train_row_count: train_row_count+test_row_count]
    df_all.loc[ix, 'Usage'] = 'test'

# create train and test data for the model
train_mask = df_all['Usage'] == 'train'
test_mask = df_all['Usage'] == 'test'
X_columns = ['leaf length', 'leaf width', 'widest point', 'total veins']
y_columns = ['Species']
X = df_all[train_mask][X_columns].values
y = df_all[train_mask][y_columns].values
Z = df_all[test_mask][X_columns].values
ZT = df_all[test_mask][y_columns].values

# fit the training data
n_neighbors = 15
weights = ['uniform', 'distance']
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights[0])
clf.fit(X, y.ravel())

# predict the test data
output = clf.predict(Z)
result = output == ZT.ravel()
print np.unique(result, return_counts=True)
print('cheese')


# create train and test data for the model
X_columns = ['leaf length', 'leaf width']
y_columns = ['Species']
X = df_all[train_mask][X_columns].values
y = df_all[train_mask][y_columns].values
Z = df_all[test_mask][X_columns].values
ZT = df_all[test_mask][y_columns].values

h = 1.0  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform']: #, 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
