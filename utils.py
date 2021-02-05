import os
import re
import time
import copy
import string
import random
import pickle
import numpy as np
import pandas as pd
from IPython.display import display


import torch

import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub


import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF



from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, RANSACRegressor, PassiveAggressiveRegressor, LassoLars
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy 
nlp = spacy.load('de_core_news_lg') 

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import matplotlib.pyplot as plt
import seaborn as sns

from bayes_opt import BayesianOptimization
from sklearn.feature_selection import SelectFromModel

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")

DECIMALS = 4

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


EMBEDDINGS_SIZE = 4396 + 768
MERGE_WITH_EMBEDDINGS = False

PATH_TO_DATA = "data/"
COLUMNS = ["id", "latitude", "longitude", "text"]
TEST_COLUMNS = ["id", "text"]

PATH_TO_TRAIN = PATH_TO_DATA + "training.txt"
PATH_TO_VALID = PATH_TO_DATA + "validation.txt"
PATH_TO_TEST  = PATH_TO_DATA + "test.txt"

PATH_TO_PREPROCCESED_TRAIN = PATH_TO_DATA + "preprocessed/preprocessed_train.csv"
PATH_TO_PREPROCCESED_VALID = PATH_TO_DATA + "preprocessed/preprocessed_valid.csv"
PATH_TO_PREPROCCESED_TEST  = PATH_TO_DATA + "preprocessed/preprocessed_test.csv"

PATH_TO_TRAIN_EMBEDDINGS = PATH_TO_DATA + "/embeddings/version-7/train_embeddings.csv"
PATH_TO_VALID_EMBEDDINGS = PATH_TO_DATA + "/embeddings/version-7/valid_embeddings.csv"
PATH_TO_TEST_EMBEDDINGS  = PATH_TO_DATA + "/embeddings/version-7/test_embeddings.csv"

PATH_TO_LOGGER = "logs/logger.csv"
logger = pd.read_csv(PATH_TO_LOGGER)

PATH_TO_MODEL_ANALYSIS = PATH_TO_DATA + "analysis/model_analysis.csv"
model_analyzer = pd.read_csv(PATH_TO_MODEL_ANALYSIS)


train_df = pd.read_csv(PATH_TO_TRAIN, sep = ',', header = None)
train_df.columns = COLUMNS

valid_df = pd.read_csv(PATH_TO_VALID, sep = ',', header = None)
valid_df.columns = COLUMNS

test_df = pd.read_csv(PATH_TO_TEST, sep = ',', header = None)
test_df.columns = TEST_COLUMNS

preprocessed_train = pd.read_csv(PATH_TO_PREPROCCESED_TRAIN)
preprocessed_valid = pd.read_csv(PATH_TO_PREPROCCESED_VALID)
preprocessed_test  = pd.read_csv(PATH_TO_PREPROCCESED_TEST)

if MERGE_WITH_EMBEDDINGS:
	train_emb = pd.read_csv(PATH_TO_TRAIN_EMBEDDINGS)
	valid_emb = pd.read_csv(PATH_TO_VALID_EMBEDDINGS)
	test_emb  = pd.read_csv(PATH_TO_TEST_EMBEDDINGS)

	assert len(train_df) == len(train_emb), "[utils.py] -> [Merging Train] -> Shape[0] does not match"
	assert len(valid_df) == len(valid_emb), "[utils.py] -> [Merging Valid] -> Shape[0] does not match"
	assert len(test_df)  == len(test_emb), "[utils.py] -> [Merging Valid] -> Shape[0] does not match"

	train_emb["id"] = train_df["id"]
	train_df = train_df.merge(train_emb, on = "id")

	valid_emb["id"] = valid_df["id"]
	valid_df = valid_df.merge(valid_emb, on = "id")

	test_emb["id"] = test_df["id"]
	test_df = test_df.merge(test_emb, on = "id")


submission = pd.DataFrame(columns = ["id", "lat", "long"])
submission["id"] = test_df.id

def predict_to_submission_file(model, embeddings, label: str, path: str = "submission.txt", save_file: bool = False):
	assert label in ["lat", "long"], "[utils.py] -> [add_to_submission_file] -> Wrong label, must be in ['lat', 'long']"
	predictions = model.predict(embeddings)
	submission[label] = predictions

	if save_file:
		submission.to_csv(path, index = False)

def add_to_submission_file(predictions, label: str, path: str = "submission.txt", save_file: bool = False):
	assert label in ["lat", "long"], "[utils.py] -> [add_to_submission_file] -> Wrong label, must be in ['lat', 'long']"
	submission[label] = predictions

	if save_file:
		submission.to_csv(path, index = False)

def round(x: float, decimal: int = DECIMALS):
	return np.round(x, decimal)

def create_test_features(features: [str]) -> [str]:
	test_features = [feature.split("_") for feature in features]
	test_features = [split[0] + "_test_" + split[1] + "_" + split[2] for split in test_features]
	
	return test_features

print("All Modules and Data Imported")
