#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('cp ../input/rapids/rapids.0.17.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[2]:


import time
import torch
import random
import numpy as np
import pandas as pd

from IPython.display import display
from sklearn.metrics import mean_absolute_error

import cudf
import cupy as cp
import cupyx.scipy.sparse

from cuml import SVR, RandomForestRegressor, QN, Ridge, Lasso, KNeighborsRegressor


# In[3]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[4]:


PATH_TO_TRAIN = "../input/geolocationofgermantweetsdataset/train_embeddings.csv"
PATH_TO_VALID = "../input/geolocationofgermantweetsdataset/valid_embeddings.csv"
PATH_TO_TEST  = "../input/geolocationofgermantweetsdataset/test_embeddings.csv"

train_df = pd.read_csv(PATH_TO_TRAIN)
valid_df = pd.read_csv(PATH_TO_VALID)
test_df  = pd.read_csv(PATH_TO_TEST)


# In[ ]:





# In[ ]:


# # display(train_df)
# EMBEDDINGS_SIZE = 4396 + 768

# features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]


# tic = time.time()

# train_latitude, train_longitude, train_error = 0, 0, 0
# valid_latitude, valid_longitude, valid_error = 0, 0, 0

# for label in ["latitude", "longitude"]:
#     X_train, y_train = train_df[features].values, train_df[label].values
#     X_valid, y_valid = valid_df[features].values, valid_df[label].values

#     X_train = cp.asarray(X_train)
#     X_valid = cp.asarray(X_valid)

#     y_train = cp.asarray(y_train)
#     y_valid = cp.asarray(y_valid)

#     model = SVR()
#     print("Train input shape: {}".format(X_train.shape))
#     FEATURES = X_train.shape[1]
#     model.fit(X_train, y_train)

#     y_predict_train = model.predict(X_train)
#     y_predict_valid = model.predict(X_valid)

#     y_predict_train = cp.asnumpy(y_predict_train)
#     y_predict_valid = cp.asnumpy(y_predict_valid)
    
#     y_train = cp.asnumpy(y_train)
#     y_valid = cp.asnumpy(y_valid)
    
#     mae_train = mean_absolute_error(y_train, y_predict_train)
#     mae_valid = mean_absolute_error(y_valid, y_predict_valid)

#     print("MAE Train {}: {}".format(label, mae_train))
#     print("MAE Valid {}: {}".format(label, mae_valid))

#     if label == "latitude":
#         train_latitude, valid_latitude = mae_train, mae_valid
#     else:
#         train_longitude, valid_longitude = mae_train, mae_valid
        


# train_error = (train_latitude + train_longitude) / 2
# valid_error = (valid_latitude + valid_longitude) / 2

# print("Train Error: {}".format(train_error))
# print("Valid Error: {}".format(valid_error))

# toc = time.time()
# print("[training] -> time {}'s".format(toc - tic))


# In[5]:


from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, StratifiedKFold
import gc


PATH_TO_LEVEL_ONE_MODELS   = "../input/geolocationofgermantweetsdataset/level_one_models.csv"
PATH_TO_LEVEL_ONE_FEATURES = "../input/geolocationofgermantweetsdataset/level_one_features.csv"
PATH_TO_LEVEL_ONE_TEST_FEATURES = "../input/geolocationofgermantweetsdataset/level_one_test_features.csv"

level_one_features = pd.read_csv(PATH_TO_LEVEL_ONE_FEATURES)
level_one_test_features = pd.read_csv(PATH_TO_LEVEL_ONE_TEST_FEATURES)
level_one_models   = pd.read_csv(PATH_TO_LEVEL_ONE_MODELS)

display(level_one_models.tail(n = 30))

data = pd.concat([train_df, valid_df], axis = 0)
data = data.sample(frac = 1, random_state = SEED).reset_index(drop=True)

del train_df, valid_df
gc.collect()


# In[ ]:


for C in [1, 5, 10, 20]:
    N_FOLDS = 5
    N_REPEATS = 5

    # display(level_one_models)

    features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]

    ID = level_one_models.id.values[-1] + 1
    TEXT = "embeddings"
    MODEL = "SVR"
    PARAMETERS = "C = {}, kernel = 'rbf'".format(C)
    OBSERVATION = "stacked + transformers (version-7)"


    tic = time.time()

    rkf = RepeatedKFold(n_splits = N_FOLDS, n_repeats = N_REPEATS, random_state = SEED)

    for label in ["latitude", "longitude"]:
        print("Training for label: {}".format(label))
        y_oof  = np.zeros((data.shape[0], N_REPEATS))
        y_test = np.zeros((test_df.shape[0], N_FOLDS, N_REPEATS))
        for repet_idx, (train_idx, valid_idx) in enumerate(rkf.split(data)):
            print("FOLD {}, Repetition {}".format(repet_idx % N_FOLDS, repet_idx // N_FOLDS))
            X_train, y_train = data.iloc[train_idx][features].values, data.iloc[train_idx][label].values
            X_valid, y_valid = data.iloc[valid_idx][features].values, data.iloc[valid_idx][label].values
            X_test = test_df[features].values

            X_train = cp.asarray(X_train)
            X_valid = cp.asarray(X_valid)
            X_test = cp.asarray(X_test)

            y_train = cp.asarray(y_train)
            y_valid = cp.asarray(y_valid) 

            model = SVR(C = C)
            model.fit(X_train, y_train)

            y_predict = model.predict(X_valid)
            y_test_predict = model.predict(X_test)
            
            y_predict  = cp.asnumpy(y_predict)
            y_test_predict = cp.asnumpy(y_test_predict)

            y_oof[valid_idx, (repet_idx // N_FOLDS)] = y_predict
            y_test[:, (repet_idx % N_FOLDS), (repet_idx // N_FOLDS)] = y_test_predict

            del y_predict, y_test_predict, X_train, X_valid, X_test, y_train, y_valid, model
            gc.collect()


        y_oof = np.mean(y_oof, axis = 1)	
        mae_error = mean_absolute_error(y_oof, data[label].values)

        y_test = np.mean(y_test, axis = 2)
        y_test = np.mean(y_test, axis = 1)

        row = [ID] + [MODEL] + [PARAMETERS] + [round(mae_error, 4)] + [OBSERVATION] + [label] + [TEXT]
        level_one_models.loc[len(level_one_models)] = row
        level_one_features["feature_{}_{}".format(label, ID)] = y_oof
        level_one_test_features["feature_test_{}_{}".format(label, ID)] = y_test

        del y_oof, y_test
        gc.collect()

    toc = time.time()
    print("[training] -> time {}'s".format(toc - tic))


# In[ ]:


for (C, kernel, degree) in [(10, 'poly', 2), (20, 'poly', 2), (10, 'poly', 3), (20, 'poly', 3), (10, 'poly', 4), (20, 'poly', 4)]:
    N_FOLDS = 5
    N_REPEATS = 5

    # display(level_one_models)

    features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]

    ID = level_one_models.id.values[-1] + 1
    TEXT = "embeddings"
    MODEL = "SVR"
    PARAMETERS = "C = {}, kernel = {}, degree = {}".format(C, kernel, degree)
    OBSERVATION = "stacked + transformers (version-7)"

    tic = time.time()

    rkf = RepeatedKFold(n_splits = N_FOLDS, n_repeats = N_REPEATS, random_state = SEED)

    for label in ["latitude", "longitude"]:
        print("Training for label: {}".format(label))
        y_oof  = np.zeros((data.shape[0], N_REPEATS))
        y_test = np.zeros((test_df.shape[0], N_FOLDS, N_REPEATS))
        for repet_idx, (train_idx, valid_idx) in enumerate(rkf.split(data)):
            print("FOLD {}, Repetition {}".format(repet_idx % N_FOLDS, repet_idx // N_FOLDS))
            X_train, y_train = data.iloc[train_idx][features].values, data.iloc[train_idx][label].values
            X_valid, y_valid = data.iloc[valid_idx][features].values, data.iloc[valid_idx][label].values
            X_test = test_df[features].values

            X_train = cp.asarray(X_train)
            X_valid = cp.asarray(X_valid)
            X_test = cp.asarray(X_test)

            y_train = cp.asarray(y_train)
            y_valid = cp.asarray(y_valid) 

            model = SVR(C = C, kernel = kernel, degree = degree)
            model.fit(X_train, y_train)

            y_predict = model.predict(X_valid)
            y_test_predict = model.predict(X_test)
            
            y_predict  = cp.asnumpy(y_predict)
            y_test_predict = cp.asnumpy(y_test_predict)

            y_oof[valid_idx, (repet_idx // N_FOLDS)] = y_predict
            y_test[:, (repet_idx % N_FOLDS), (repet_idx // N_FOLDS)] = y_test_predict

            del y_predict, y_test_predict, X_train, X_valid, X_test, y_train, y_valid, model
            gc.collect()


        y_oof = np.mean(y_oof, axis = 1)	
        mae_error = mean_absolute_error(y_oof, data[label].values)

        y_test = np.mean(y_test, axis = 2)
        y_test = np.mean(y_test, axis = 1)

        row = [ID] + [MODEL] + [PARAMETERS] + [round(mae_error, 4)] + [OBSERVATION] + [label] + [TEXT]
        level_one_models.loc[len(level_one_models)] = row
        level_one_features["feature_{}_{}".format(label, ID)] = y_oof
        level_one_test_features["feature_test_{}_{}".format(label, ID)] = y_test

        del y_oof, y_test
        gc.collect()

    toc = time.time()
    print("[training] -> time {}'s".format(toc - tic))


# In[ ]:


for alpha in [1, 5, 10]:
    N_FOLDS = 5
    N_REPEATS = 5

    # display(level_one_models)

    features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]

    ID = level_one_models.id.values[-1] + 1
    TEXT = "embeddings"
    MODEL = "Ridge"
    PARAMETERS = "alpha = {}".format(alpha)
    OBSERVATION = "stacked + transformers (version-7)"

    tic = time.time()

    rkf = RepeatedKFold(n_splits = N_FOLDS, n_repeats = N_REPEATS, random_state = SEED)

    for label in ["latitude", "longitude"]:
        print("Training for label: {}".format(label))
        y_oof  = np.zeros((data.shape[0], N_REPEATS))
        y_test = np.zeros((test_df.shape[0], N_FOLDS, N_REPEATS))
        for repet_idx, (train_idx, valid_idx) in enumerate(rkf.split(data)):
            print("FOLD {}, Repetition {}".format(repet_idx % N_FOLDS, repet_idx // N_FOLDS))
            X_train, y_train = data.iloc[train_idx][features].values, data.iloc[train_idx][label].values
            X_valid, y_valid = data.iloc[valid_idx][features].values, data.iloc[valid_idx][label].values
            X_test = test_df[features].values

            X_train = cp.asarray(X_train)
            X_valid = cp.asarray(X_valid)
            X_test = cp.asarray(X_test)

            y_train = cp.asarray(y_train)
            y_valid = cp.asarray(y_valid) 

            model = Ridge(alpha = alpha)
            model.fit(X_train, y_train)

            y_predict = model.predict(X_valid)
            y_test_predict = model.predict(X_test)
            
            y_predict  = cp.asnumpy(y_predict)
            y_test_predict = cp.asnumpy(y_test_predict)

            y_oof[valid_idx, (repet_idx // N_FOLDS)] = y_predict
            y_test[:, (repet_idx % N_FOLDS), (repet_idx // N_FOLDS)] = y_test_predict

            del y_predict, y_test_predict, X_train, X_valid, X_test, y_train, y_valid, model
            gc.collect()


        y_oof = np.mean(y_oof, axis = 1)	
        mae_error = mean_absolute_error(y_oof, data[label].values)

        y_test = np.mean(y_test, axis = 2)
        y_test = np.mean(y_test, axis = 1)

        row = [ID] + [MODEL] + [PARAMETERS] + [round(mae_error, 4)] + [OBSERVATION] + [label] + [TEXT]
        level_one_models.loc[len(level_one_models)] = row
        level_one_features["feature_{}_{}".format(label, ID)] = y_oof
        level_one_test_features["feature_test_{}_{}".format(label, ID)] = y_test

        del y_oof, y_test
        gc.collect()

    toc = time.time()
    print("[training] -> time {}'s".format(toc - tic))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

for C in [1, 5, 10, 20]:
    N_FOLDS = 5
    N_REPEATS = 5

    # display(level_one_models)

    ID = level_one_models.id.values[-1] + 1
    TEXT = "original"
    MODEL = "SVR"
    PARAMETERS = "C = {}, kernel = 'rbf'".format(C)
    OBSERVATION = "ngram_range = (1, 10), analyzer = 'char_wb'"

    tic = time.time()

    rkf = RepeatedKFold(n_splits = N_FOLDS, n_repeats = N_REPEATS, random_state = SEED)

    for label in ["latitude", "longitude"]:
        print("Training for label: {}".format(label))
        y_oof  = np.zeros((data.shape[0], N_REPEATS))
        y_test = np.zeros((test_df.shape[0], N_FOLDS, N_REPEATS))
        for repet_idx, (train_idx, valid_idx) in enumerate(rkf.split(data)):
            print("FOLD {}, Repetition {}".format(repet_idx % N_FOLDS, repet_idx // N_FOLDS))
            X_train, y_train = data.iloc[train_idx].text.values, data.iloc[train_idx][label].values
            X_valid, y_valid = data.iloc[valid_idx].text.values, data.iloc[valid_idx][label].values
            X_test = test_df.text.values

            tfv = TfidfVectorizer(ngram_range = (1, 10), analyzer = 'char_wb')
            tfv.fit(X_train)

            X_train = tfv.transform(X_train)
            X_valid = tfv.transform(X_valid)
            X_test  = tfv.transform(X_test)
            
            svd = TruncatedSVD(n_components = 5000, random_state = SEED)
            svd.fit(X_train)

            X_train = svd.transform(X_train)
            X_valid = svd.transform(X_valid)
            X_test  = svd.transform(X_test)

            X_train = cp.asarray(X_train)
            X_valid = cp.asarray(X_valid)
            X_test = cp.asarray(X_test)

            y_train = cp.asarray(y_train)
            y_valid = cp.asarray(y_valid) 

            model = SVR(C = C)
            model.fit(X_train, y_train)
                    

            y_predict = model.predict(X_valid)
            y_test_predict = model.predict(X_test)
            
            y_predict  = cp.asnumpy(y_predict)
            y_test_predict = cp.asnumpy(y_test_predict)

            y_oof[valid_idx, (repet_idx // N_FOLDS)] = y_predict
            y_test[:, (repet_idx % N_FOLDS), (repet_idx // N_FOLDS)] = y_test_predict

            del y_predict, y_test_predict, X_valid, X_test, y_valid, X_train, y_train, model
            gc.collect()


        y_oof = np.mean(y_oof, axis = 1)	
        mae_error = mean_absolute_error(y_oof, data[label].values)

        y_test = np.mean(y_test, axis = 2)
        y_test = np.mean(y_test, axis = 1)

        row = [ID] + [MODEL] + [PARAMETERS] + [round(mae_error, 4)] + [OBSERVATION] + [label] + [TEXT]
        level_one_models.loc[len(level_one_models)] = row
        level_one_features["feature_{}_{}".format(label, ID)] = y_oof
        level_one_test_features["feature_test_{}_{}".format(label, ID)] = y_test

        del y_oof, y_test
        gc.collect()

    toc = time.time()
    print("[training] -> time {}'s".format(toc - tic))


# In[ ]:


display(level_one_models.tail(n = 30))


# In[ ]:


level_one_models.to_csv("level_one_models.csv", index = False)
level_one_features.to_csv("level_one_features.csv", index = False)
level_one_test_features.to_csv("level_one_test_features.csv", index = False)

