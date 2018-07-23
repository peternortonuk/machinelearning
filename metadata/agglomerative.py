import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm

np.set_printoptions(precision=4, suppress = True)
plt.figure(figsize=(10,3))
#%matplotlibinline plt.style.use('seaborn-whitegrid.)

address = r'C:\dev\code\machinelearning\data\mtcars.csv'
cars = pd.read_csv(address)

cars.columns = ['car_names' ,'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
X = cars.ix[:, (1,3,4,6)].values
y = cars.ix[:, (9)].values

Z = linkage(X, 'ward')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.show()
pass

