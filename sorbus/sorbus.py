
import pandas as pd
import numpy as np
import os
from sklearn import neighbors
from utilities import plot_knn

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
np.ndarray.sort(unique_species)
for i, species in enumerate(unique_species):
    species_mask = df_all['Species'] == species
    # add integer code for use in charting later
    df_all.loc[species_mask,'Species_code'] = i
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
model_mask = df_all['Usage'].isin(['train', 'test'])
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

# charting with 2D only; use all the model data
# also requires numeric chart labels (instead of species name)
X_columns = ['leaf length', 'leaf width']
y_columns = ['Species_code']
X = df_all[model_mask][X_columns].values
y = df_all[model_mask][y_columns].values
plot_knn(X, y, n_neighbors)

print('cheese')