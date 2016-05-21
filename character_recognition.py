import pandas as pd
import numpy as np
import os
import sklearn as sk


# import some data into dataframes
path_data = r'C:\dev\code\machinelearning\data'

# image data
file_X = r'digits_pixels_X.csv'
pathfile = os.path.join(path_data, file_X)
X = pd.read_csv(filepath_or_buffer=pathfile, header=None)
columns = np.arange(len(X.columns))
X.columns = columns

# results data
file_y = r'digits_values_y.csv'
pathfile = os.path.join(path_data, file_y)
y = pd.read_csv(filepath_or_buffer=pathfile, header=None)
y.columns = ['value']

print 'cheese'