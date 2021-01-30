# LUIZ HENRIQUE DE MELO SANTOS - 2017014464
# SISTEMAS DE RECOMENDACAO - 2020/2 - DIG DCC030 TO

# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
OUTPUT_DIR = 'out.csv'

# PARAMETERS
lmbda = 0.2  # regularization parameter
k = 3 # dimension of matrix
n_epochs = 25  # number of epochs
alpha=0.01  # learning rate

# data import
_ratings = pd.read_csv(sys.argv[1])
_ratings.columns = ['UserId:ItemId', 'Rating', 'Timestamp']
_targets = pd.read_csv(sys.argv[2])

# shuffle data
_ratings = _ratings.sample(frac=1).reset_index(drop=True)

# separate first column
sep = _ratings['UserId:ItemId'].str.split(':', expand=True)
sep.columns = ['UserId', 'ItemId']
_ratings = pd.concat([_ratings, sep], axis=1)
# selete columns
_ratings = _ratings.drop(['UserId:ItemId'], axis=1)

def encode_column(column):
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array([key_to_id[x] for x in column]), len(keys)

def encode_df(df):
    item_ids, df.loc[:,'ItemId'], num_item = encode_column(df['ItemId'].copy())
    user_ids, df.loc[:,'UserId'], num_users = encode_column(df['UserId'].copy())
    return df, num_users, num_item, user_ids, item_ids

df, num_users, num_item, user_ids, item_ids = encode_df(_ratings)

# create a sparse user-item matrix
ratings = df.pivot(index="UserId", columns="ItemId", values="Rating")
ratings = ratings.fillna(0).values

# creates a training set by selecting users that have 35 or more ratings, then randomly select
# 15 of those ratings for validation set, but set those values to 0 in the training set
def train_test_split(ratings):
    validation = np.zeros(ratings.shape)
    train = ratings.copy()
    
    for user in np.arange(ratings.shape[0]):
        if len(ratings[user,:].nonzero()[0]) >= 35:
            val_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=15, replace=False)
            train[user, val_ratings] = 0
            validation[user, val_ratings] = ratings[user, val_ratings]
    return train, validation

train, val = train_test_split(ratings)

m, n = train.shape  # number of users and items
P = 3 * np.random.rand(k,m) # user feature matrix
Q = 3 * np.random.rand(k,n) # movie feature matrix

# P is user feature matrix
# Q is item feature matrix
def prediction(P,Q):
    return np.dot(P.T,Q)

# training of the model
train_errors = []
val_errors = []
users,items = train.nonzero()      
for epoch in range(n_epochs):
    for u, i in zip(users,items):
        e = train[u, i] - prediction(P[:,u],Q[:,i])  # calculate error for gradient update
        P[:,u] += alpha * ( e * Q[:,i] - lmbda * P[:,u]) # update user feature matrix
        Q[:,i] += alpha * ( e * P[:,u] - lmbda * Q[:,i])  # update item feature matrix

# recover the matrix of predictions
SGD_predictions = prediction(P,Q)
mean = SGD_predictions.mean()

# create the output dataframe of predictions
_predictions = _targets.copy()
_predictions['Prediction'] = 0

# define ratings for input dataset
for i in range(_predictions.shape[0]):
    inf = _predictions['UserId:ItemId'][i].split(':')
    
    if inf[0] not in user_ids:  # if user not in data base
        if inf[1] not in item_ids:  # if theres no one in data base
            _predictions.loc[i, 'Prediction'] = mean
        else:  # calculate item means
            _predictions.loc[i, 'Prediction'] = np.mean(SGD_predictions[:][item_ids[inf[1]]])

    elif inf[1] not in item_ids:  # if item not in data base
        if inf[0] not in user_ids:  # if theres no one in data base
            _predictions.loc[i, 'Prediction'] = mean
        else:  # calculate user means
            _predictions.loc[i, 'Prediction'] = np.mean(SGD_predictions[user_ids[inf[0]]][:])

    else:  # recover the prediction previously made
        _predictions.loc[i, 'Prediction'] = SGD_predictions[user_ids[inf[0]]][item_ids[inf[1]]]

# generate the output archive
_predictions.to_csv(OUTPUT_DIR, index=False)
