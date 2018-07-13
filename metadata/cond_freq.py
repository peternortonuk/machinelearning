from load_data import get_df, select_columns
from itertools import combinations
from nltk import ConditionalFreqDist

# get the data as a dataframe
df = get_df()
mask_data_source = df['DataSource'] == 'APX'
df_select = df[mask_data_source]

# choose subset of columns and cast all values as string
df_select = df_select[select_columns].astype(str)

# choose a smaller subset of columns to analyse
report_columns = select_columns[1:5]

# a list of all pairwise combinations
combo_count = 2
groupby_columns = list(combinations(report_columns, combo_count))

# create a list of tuples
groupby_column = list(groupby_columns[0])
arr = df_select[list(groupby_column)].values
pairs = list(tuple(map(tuple, arr)))

# and now for the good stuff
cfd = ConditionalFreqDist(pairs)
conditions = cfd.conditions()

import pdb; pdb.set_trace()
