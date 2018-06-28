import pandas as pd
import os
from  itertools import combinations

file = r'arc_metadata.csv'
path = r'C:\dev\code\machinelearning\data'
filepath = os.path.join(path, file)
df = pd.read_csv(filepath)

mask_data_source = df['DataSource'] == 'APX'
df_select = df[mask_data_source]

rowcount = len(df_select.index)
column_count = len(df_select.columns)
combo_select = 2

# a list of all combinations
groupby_columns = list(combinations(df_select.columns, combo_select))

# every combination
df_analysis = pd.DataFrame()
for groupby_column in groupby_columns:
    # count distinct
    df_counts = df_select.groupby(groupby_column).nunique()

    # include the combo definition as a new column; populate with a list of tuples
    # note that we wont get a groupby where one element is NaN; so count again
    count_of_groups = len(df_counts.index)
    df_counts['groupby_column'] = [groupby_column] * count_of_groups

    # build up the list of results
    df_analysis = pd.concat([df_analysis, df_counts], axis=0)

import pdb; pdb.set_trace()

pass