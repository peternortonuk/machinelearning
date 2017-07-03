from __future__ import division
import pandas as pd
import numpy as np
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from utilities import plot_knn
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


# parameters of the model
n_neighbors = 10
weights = ['uniform', 'distance']
weight = weights[0]

ss = ShuffleSplit(n_splits=10, test_size=0.1)
for train_index, test_index in ss.split(X):
    # generate data from indices
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    # fit the training data
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X_train, y_train.ravel())

    # predict the test data
    output = clf.predict(X_test)

    # report results
    score = clf.score(X_test, y_test)
    print("Score: {:.2%}".format(score))


cv = 10
k_range = range(1, 20)
k_scores = []
k_vars = []
for k in k_range:
    knn = neighbors.KNeighborsClassifier(k, weights=weight)
    scores = cross_val_score(knn, X, y.ravel(), cv=cv, scoring='accuracy')
    k_scores.append(scores.mean())
    k_vars.append(scores.var())
plt.plot(k_range, k_scores)
plt.show()

plt.plot(k_range, k_vars)
plt.show()

# using GridSearch
cv = 10
parameters = {'weights': weights, 'n_neighbors': range(1, 10)}
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
clf = GridSearchCV(estimator=clf, cv=cv, param_grid=parameters)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_factor, random_state=0)
clf.fit(X_train, y_train)

print('cheese')

chart_it = None
if chart_it:
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

