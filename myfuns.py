import numpy as np
import pandas as pd
import requests
from io import StringIO

def readDat(this_url, col_names):
    this_string_raw = requests.get(this_url).text.split('\n')[:-1]
    this_data = [line.split('::') for line in this_string_raw]
    return pd.DataFrame(this_data, columns=col_names)

def myIBCF(w):
    w_pred = pd.DataFrame(np.zeros(S_matrix.shape[1]), index=S_matrix.columns, columns=['estimate'])
    for l in S_top_30.index:
        l_top_30_ix = S_top_30.loc[l, :].values
        w_l_top_30 = w[l_top_30_ix]
        S_l_top_30 = S_matrix.loc[l, l_top_30_ix]
        target_ix = (~S_l_top_30.isnull()) & (~w_l_top_30.isnull())
        sub_w = w_l_top_30[target_ix.values]
        sub_s = S_l_top_30[target_ix.values]
        w_l_pred = (sub_w * sub_s).sum() / sub_s.sum()
        w_pred.loc[l] = w_l_pred
    w_merged = pd.DataFrame(w).merge(w_pred, left_index=True, right_index=True)
    w_sorted = w_merged[w_merged.iloc[:, 0].isnull()].sort_values('estimate', ascending=False)
    return w_sorted.head(10).index.values, w_sorted

target_url = "https://liangfgithub.github.io/MovieData/"
df_movies = readDat(target_url + 'movies.dat?raw=true', ['movie_id', 'title', 'genres'])

# df_users = readDat(target_url + 'users.dat?raw=true', ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
# df_users['Age'] = df_users.Age.astype(int)
# df_users['UserID'] = df_users.UserID.astype(int)

#read pre-built dataframes
df_selection = pd.read_csv("movies_top.csv", index_col=0)
S_matrix = pd.read_csv("s_matrix_sub.csv", index_col=0)
S_top_30 = pd.read_csv("s_top_30_sub.csv", index_col=0)

w_empty = pd.DataFrame(np.full(S_matrix.shape[0], None), index=S_matrix.columns).iloc[:,0]

#-------------------------------------

genres = list(
    sorted(set([genre for genres in df_selection.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return df_selection.head(100)

def get_recommended_movies(new_user_ratings):
    df_new_user_ratings = pd.DataFrame([new_user_ratings]).T
    df_new_user_ratings.index = ['m' + str(x) for x in df_new_user_ratings.index]
    new_w = w_empty.copy()
    new_w[df_new_user_ratings.index] = df_new_user_ratings.iloc[:, 0].values
    top_10_str, _ = myIBCF(new_w)
    top_10_no_only = [x[1:] for x in top_10_str]
    return df_movies.set_index('movie_id').loc[top_10_no_only].reset_index()

def get_popular_movies(genre: str):
    return df_selection[df_selection.genres.str.contains(genre)].head(10)
