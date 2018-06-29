import pandas as pd
import os
from itertools import combinations
import matplotlib.pyplot as plt
#import mplcursors

file = r'arc_metadata.csv'
path = r'C:\dev\code\machinelearning\data'
filepath = os.path.join(path, file)
df = pd.read_csv(filepath)

mask_data_source = df['DataSource'] == 'APX'
df_select = df[mask_data_source]

rowcount = len(df_select.index)
column_count = len(df_select.columns)
combo_count = 2

select_columns = [#'DataSource',
                  'Commodity',
                  'ProductOrMarket',
                  'Location',
                  'ContractType',
                  'PriceType',
                  #'Granularity',
                  #'PublicationFrequency',
                  #'PublicationCalendar',
                  'Currency',
                  'CurveUnit',
                  'CurveType',
                  'UsageType',
                  'OwnerRole',
                  'Destination',
                  'CurveStatus',
                  'TimeOfDay',
                  'Timezone',
                  ]
report_columns = select_columns
select_columns = select_columns[1:5]

# a list of all combinations
groupby_columns = list(combinations(select_columns, combo_count))

# every combination
dict_of_df = {}
for groupby_column in groupby_columns:
    # count distinct
    df_counts = df_select[report_columns].groupby(groupby_column).nunique()

    # count includes the groupby as an index; we want to combine to a single index
    a0 = df_counts.index.get_level_values(0).values.tolist()
    a1 = df_counts.index.get_level_values(1).values.tolist()
    l = list(zip(a0, a1))
    l = [' || '.join((str(i0), str(i1))) for i0, i1 in l]
    index = pd.Index(l)
    df_counts['Index'] = index
    df_counts.set_index('Index', drop=True, inplace=True)

    # build up the list of results
    dict_of_df[groupby_column] = df_counts

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
