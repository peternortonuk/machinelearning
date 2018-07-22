import numpy as np
import pandas as pd

import scipy
from scipy.cluster.hierarchy import dendogram, linkage
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

