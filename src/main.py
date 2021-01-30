import time
from datetime import timedelta
start = time.time()

import numpy as np
import pandas as pd
OUTPUT_DIR = 'out.csv'

lmbda = 0.2 # Regularization parameter
k = 3 # tweak this parameter
n_epochs = 45  # Number of epochs
alpha=0.01  # Learning rate

# Data import
_ratings = pd.read_csv('ratings.csv')
_ratings.columns = ['UserId:ItemId', 'Rating', 'Timestamp']
_targets = pd.read_csv('targets.csv')

# Shuffle data
_ratings = _ratings.sample(frac=1).reset_index(drop=True)

# Separate first column
sep = _ratings['UserId:ItemId'].str.split(':', expand=True)
sep.columns = ['UserId', 'ItemId']
_ratings = pd.concat([_ratings, sep], axis=1)

# Delete columns
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
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100

# This creats a validation dataset by selecting rows (user) that have 35 or more ratings, then randomly select 15 of those ratings
# For validation set, but set those values to 0 in the training set.
def train_test_split(ratings):
    validation = np.zeros(ratings.shape)
    train = ratings.copy() #don't do train=ratings, other wise, ratings becomes empty
    
    for user in np.arange(ratings.shape[0]):
        if len(ratings[user,:].nonzero()[0])>=35:  # 35 seems to be best, it depends on sparsity of your user-item matrix
            val_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=15, #tweak this, 15 seems to be optimal
                                        replace=False)
            train[user, val_ratings] = 0
            validation[user, val_ratings] = ratings[user, val_ratings]
    return train, validation

train, val = train_test_split(ratings)

m, n = train.shape  # Number of users and items
P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix

#P is latent user feature matrix
#Q is latent item feature matrix
def prediction(P,Q):
    return np.dot(P.T,Q)

train_errors = []
val_errors = []
#Only consider items with ratings 
users,items = train.nonzero()      
for epoch in range(n_epochs):
    for u, i in zip(users,items):
        e = train[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient update
        P[:,u] += alpha * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
        Q[:,i] += alpha * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent item feature matrix

# recover the matrix of predictions
SGD_predictions = prediction(P,Q)
mean = SGD_predictions.mean()

_predictions = _targets.copy()
_predictions['Prediction'] = 0

# calculate user/item means just if necessary
for i in range(_predictions.shape[0]):
    inf = _predictions['UserId:ItemId'][i].split(':')
    
    if inf[0] not in user_ids:
        if inf[1] not in item_ids:  # if theres no one in data base
            _predictions.loc[i, 'Prediction'] = mean
        else:  # if user not in data base - calculate item means
            _predictions.loc[i, 'Prediction'] = np.mean(SGD_predictions[:][item_ids[inf[1]]])
        
    elif inf[1] not in item_ids:
        if inf[0] not in user_ids:  # if theres no one in data base
            _predictions.loc[i, 'Prediction'] = mean
        else:  # if item not in data base - calculate user means
            _predictions.loc[i, 'Prediction'] = np.mean(SGD_predictions[user_ids[inf[0]]][:])
        
    else:
        _predictions.loc[i, 'Prediction'] = SGD_predictions[user_ids[inf[0]]][item_ids[inf[1]]]

_predictions.to_csv(OUTPUT_DIR, index=False)

elapsed = (time.time()-start)
print("Time elapsed:",str(timedelta(seconds=elapsed)), "\n")
