import pandas as pd
from load_data import get_df, select_columns
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# get the data as a dataframe
df = get_df()
mask_data_source = df['DataSource'] == 'APX'
df_select = df[mask_data_source]
df_select = df_select[select_columns]


# create a list of features where each member is one row of the df
features = []
for index, row in df_select.iterrows():
    list_ = row.values.tolist()
    list_ = [str(i).lower() for i in list_]
    features.insert(-1, list_)


# the set of all words encountered
all_words = set(word.lower() for row in features for word in row)


# dictionary of true/false status for each word encountered
t = [({word: (word in row) for word in all_words}, 1) for row in features]

# start work
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_select)

import pdb; pdb.set_trace()
