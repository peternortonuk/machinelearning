import pandas as pd
import numpy as np
from itertools import combinations
#import matplotlib.pyplot as plt
from load_data import get_df, select_columns
#import mplcursors


df = get_df(shortname='clean_apx')
mask_data_source = df['DataSource'] == 'APX'
df_select = df[mask_data_source]

rowcount = len(df_select.index)
column_count = len(df_select.columns)
combo_count = 2

report_columns = select_columns
select_columns = select_columns[1:5]

# a list of all combinations
groupby_columns = list(combinations(select_columns, combo_count))

# every combination
dict_of_df = {}
dict_of_norm_df = {}
for groupby_column in groupby_columns:
    # count distinct
    df_unique_count = df_select[report_columns].groupby(groupby_column).nunique()

    # simple row count
    df_simple_count = df_select.groupby(groupby_column).size().to_frame()
    df_simple_count.columns = ['RowCount']

    # combine them
    df_count = pd.concat([df_simple_count, df_unique_count], axis=1)

    # count includes the groupby as an index; we want to combine to a single index
    a0 = df_count.index.get_level_values(0).values.tolist()
    a1 = df_count.index.get_level_values(1).values.tolist()
    l = list(zip(a0, a1))
    l = [' || '.join((str(i0), str(i1))) for i0, i1 in l]
    index = pd.Index(l)
    df_count['Index'] = index
    df_count.set_index('Index', drop=True, inplace=True)

    # build up the list of results
    dict_of_df[groupby_column] = df_count

    # now test for data quality; take a copy because we will overwrite values here
    df_norm = df_count.copy(deep=True)
    for i, row in df_norm.iterrows():
        for column in report_columns:
            if (row[column] == 1) | (row[column] == row['RowCount']):
                df_norm.loc[i, column] = np.NaN
            else:
                print('groupby fields: {}; groupby values: {}; column: {}.... count: {}: '.format(groupby_column, i, column, row[column]))

    # unpivot the data and remove NaN's
    df_norm.reset_index(inplace=True)  # this puts the index into a column; called Index
    df_norm = pd.melt(df_norm, id_vars=['Index'], value_vars=report_columns)
    df_norm.dropna(axis=0, how='any', inplace=True)
    dict_of_norm_df[groupby_column] = df_norm

df_chart = pd.concat(dict_of_norm_df.values())
df_chart.sort_values(['variable', 'Index'], inplace=True)

import pdb; pdb.set_trace()
number_of_subplots = len(groupby_columns)
fig, axes = plt.subplots(number_of_subplots, 1,
                         figsize=(6, number_of_subplots * 6))
for i, key in enumerate(dict_of_df.keys()):
    print(key)
    try:
        dict_of_df[key].transpose().plot(ax=axes[i], kind='bar', legend=False)
        axes[i].set_title(str(key))
        plt.subplots_adjust(top=0.1,
                            bottom=0.1,
                            left=0.1,
                            right=0.1,
                            hspace=1.0,
                            wspace=0.5)

    except:
        continue
plt.show()

pass
