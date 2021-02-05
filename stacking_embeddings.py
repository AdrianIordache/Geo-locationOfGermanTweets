import numpy as np
import pandas as pd 

from IPython.display import display

PATH_TO_DATA = "data"

PATH_TO_TRAIN_STACK = PATH_TO_DATA + "/embeddings/version-5/train_embeddings.csv"
PATH_TO_VALID_STACK = PATH_TO_DATA + "/embeddings/version-5/valid_embeddings.csv"
PATH_TO_TEST_STACK  = PATH_TO_DATA + "/embeddings/version-5/test_embeddings.csv"


PATH_TO_TRAIN_TRANSFORMER = PATH_TO_DATA + "/embeddings/version-6/train_embeddings.csv"
PATH_TO_VALID_TRANSFORMER = PATH_TO_DATA + "/embeddings/version-6/valid_embeddings.csv"
PATH_TO_TEST_TRANSFORMER  = PATH_TO_DATA + "/embeddings/version-6/test_embeddings.csv"

train_stack = pd.read_csv(PATH_TO_TRAIN_STACK)
valid_stack = pd.read_csv(PATH_TO_VALID_STACK)
test_stack  = pd.read_csv(PATH_TO_TEST_STACK)

train_transform = pd.read_csv(PATH_TO_TRAIN_TRANSFORMER)
valid_transform = pd.read_csv(PATH_TO_VALID_TRANSFORMER)
test_transform  = pd.read_csv(PATH_TO_TEST_TRANSFORMER)

features = ["embedding_{}".format(emb) for emb in range(4396 + 768)]

train = np.concatenate((train_stack.values, train_transform.values), axis = 1)
valid = np.concatenate((valid_stack.values, valid_transform.values), axis = 1)
test  = np.concatenate((test_stack.values, test_transform.values), axis = 1)

train_df = pd.DataFrame(train, columns = features)
valid_df = pd.DataFrame(valid, columns = features)
test_df  = pd.DataFrame(test,  columns = features)

display(train_df)

train_df.to_csv("train_embeddings.csv", index = False)
valid_df.to_csv("valid_embeddings.csv", index = False)
test_df.to_csv("test_embeddings.csv", index = False)

