from utils import *

N_FOLDS = 5
N_REPEATS = 5

PATH_TO_LEVEL_ONE_MODELS   = "data/level_one/level_one_models.csv"
PATH_TO_LEVEL_ONE_FEATURES = "data/level_one/level_one_features.csv"
PATH_TO_LEVEL_ONE_TEST_FEATURES = "data/level_one/level_one_test_features.csv"

level_one_features = pd.read_csv(PATH_TO_LEVEL_ONE_FEATURES)
level_one_test_features = pd.read_csv(PATH_TO_LEVEL_ONE_TEST_FEATURES)
level_one_models   = pd.read_csv(PATH_TO_LEVEL_ONE_MODELS)

# display(level_one_models)

features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]

ID = 43
TEXT = "embeddings"
MODEL = "LGBMRegressor"
PARAMETERS = "n_estimators = 300"
OBSERVATION = "stacked + transformers (version-7)"

import gc

tic = time.time()

data = pd.concat([train_df, valid_df], axis = 0)
data = data.sample(frac = 1, random_state = SEED).reset_index(drop=True)

del train_df, valid_df
gc.collect()

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

		# tfv = TfidfVectorizer(ngram_range = (1, 10), analyzer = 'char_wb')
		# tfv.fit(X_train)

		# X_train = tfv.transform(X_train)
		# X_valid = tfv.transform(X_valid)
		# X_test  = tfv.transform(X_test)

		model = LGBMRegressor(n_estimators = 300, random_state = SEED, n_jobs = -1)
		model.fit(X_train, y_train)

		y_predict = model.predict(X_valid)
		y_test_predict = model.predict(X_test)

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
	level_one_models.to_csv(PATH_TO_LEVEL_ONE_MODELS, index = False)

	level_one_features["feature_{}_{}".format(label, ID)] = y_oof
	level_one_features.to_csv(PATH_TO_LEVEL_ONE_FEATURES, index = False)

	level_one_test_features["feature_test_{}_{}".format(label, ID)] = y_test
	level_one_test_features.to_csv(PATH_TO_LEVEL_ONE_TEST_FEATURES, index = False)

	del y_oof, y_test
	gc.collect()

toc = time.time()
print("[training] -> time {}'s".format(toc - tic))