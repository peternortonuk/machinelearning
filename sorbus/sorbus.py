
import pandas as pd
import numpy as np
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split
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
df_all.reset_index(drop=True, inplace=True)

# calculate the minimum row count to ensure equal group sizes
df_count = df_all.groupby(by='Species').count().iloc[:, 0]
group_row_count = df_count.min()

# create a new df with equal sized groups
unique_species = df_all['Species'].unique()
df_equal = pd.DataFrame()
for i, species in enumerate(unique_species):
    species_mask = df_all['Species'] == species
    # add integer code for use in charting later
    df_all.loc[species_mask,'Species_code'] = i
    # monumental faff due to assignment fail when using chained indexing
    # http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    ix = df_all.index[species_mask][:group_row_count]
    df_subset = df_all.loc[ix, :]
    df_equal = pd.concat([df_equal, df_subset], axis=0)
species_key_df = df_all[['Species', 'Species_code']].drop_duplicates()

# create arrays of required data
X_columns = ['leaf length', 'leaf width', 'widest point', 'total veins']
y_columns = ['Species']
X = df_equal[X_columns].values
y = df_equal[y_columns].values

# use helper function to split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_factor, random_state=0)

# fit the training data
n_neighbors = 10
weights = ['uniform', 'distance']
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights[0])
clf.fit(X_train, y_train.ravel())

# predict the test data
output = clf.predict(X_test)
result = output == y_test
print np.unique(result, return_counts=True)

# charting with 2D only; use all the model data
# also requires numeric chart labels (instead of species name)
# create new arrays of required data
X_columns = ['leaf length', 'leaf width']
y_columns = ['Species_code']
X = df_equal[X_columns].values
y = df_equal[y_columns].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_factor, random_state=0)
print(species_key_df)
plot_knn(X_test, y_test, n_neighbors)

