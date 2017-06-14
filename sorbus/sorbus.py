
import pandas as pd
import os

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

# calculate the train and test row count
df_count = df_all.groupby(by='Species').count().iloc[:,0]
total_row_count = df_count.min()
train_row_count = int(total_row_count*train_factor)
test_row_count = total_row_count - train_row_count

print('cheese')

