#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from IPython.display import display


# In[2]:


get_ipython().system(' pip uninstall -y typing')
get_ipython().system(' pip install flair')


# In[3]:


import time
def create_embeddings_flair(data: pd.DataFrame, column: str = "text", path: str = None, embeddings_type: str = "tranformer", typs: str = "train"):
    assert column in data.columns.tolist(), "[embeddings.py] -> [create_embedding_flair] -> Input column not in dataframe columns"
    assert embeddings_type in ["tranformer", "stacked"]

    from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
    from flair.data import Sentence

    fast_text_embedding      = WordEmbeddings('de')
    flair_embedding_forward  = FlairEmbeddings('de-forward')
    flair_embedding_backward = FlairEmbeddings('de-backward')

    stacked_embeddings = DocumentPoolEmbeddings([fast_text_embedding, flair_embedding_forward, flair_embedding_backward])

    transformer_embedding = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune = False)

    tic = time.time()

    embeddings = []

    for i, text in enumerate(data[column].values):
        print("sentence {}/{}".format(i, len(data)))
        sentence = Sentence(text)

        if embeddings_type == "stacked":
            stacked_embeddings.embed(sentence)
        elif embeddings_type == "tranformer":
            transformer_embedding.embed(sentence)

        embedding = sentence.embedding.detach().cpu().numpy()
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    columns = ["embedding_{}".format(feature) for feature in range(embeddings.shape[1])]

    csv = pd.DataFrame(embeddings, columns = columns)
    csv.to_csv(path + embeddings_type + "_" + typs + ".csv", index = False)

    toc = time.time()

    print("[create_embeddings_flair] -> [embeddings_type: {}, typs: {}] -> time {}'s".format(embeddings_type, typs, toc - tic))


# In[4]:


PATH_TO_TRAIN = "/kaggle/input/geolocationofgermantweetsdataset/preprocessed_train.csv"
PATH_TO_VALID = "/kaggle/input/geolocationofgermantweetsdataset/preprocessed_valid.csv"
PATH_TO_TEST  = "/kaggle/input/geolocationofgermantweetsdataset/preprocessed_test.csv"

train = pd.read_csv(PATH_TO_TRAIN)
valid = pd.read_csv(PATH_TO_VALID)
test  = pd.read_csv(PATH_TO_TEST)


# In[5]:


create_embeddings_flair(train, column = 'final_text', path = "", embeddings_type = "tranformer", typs = "train")
create_embeddings_flair(valid, column = 'final_text', path = "", embeddings_type = "tranformer", typs = "valid")
create_embeddings_flair(test,  column = 'final_text', path = "", embeddings_type = "tranformer", typs = "test")


# In[ ]:




