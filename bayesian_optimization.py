from utils import *
from feature_selection import *

PATH_TO_OPTIMIZATION = "data/optimization/"
PATH_TO_LEVEL_TWO = "data/level_two/"

PATH_TO_MODELS = PATH_TO_LEVEL_TWO + "level_two_models.csv"
PATH_TO_FEATURES = PATH_TO_LEVEL_TWO + "level_two_features_skf.csv"
PATH_TO_TEST_FEATURES = PATH_TO_LEVEL_TWO + "level_two_test_features.csv"

PATH_TO_OPTIMIZATION_LOG  = PATH_TO_OPTIMIZATION + "optimization.csv"
PATH_TO_OPTIMIZATION_TEST = PATH_TO_OPTIMIZATION + "level_two_test.csv"

level_one_models = pd.read_csv(PATH_TO_MODELS)
level_one_features = pd.read_csv(PATH_TO_FEATURES)
level_one_test_features = pd.read_csv(PATH_TO_TEST_FEATURES)


def train(C, n_estimators, learning_rate_hist, learning_rate_cat, max_leaf_nodes, keep_percentage):
	r2 = lambda x: np.round(x, 2)
	r4 = lambda x: np.round(x, 4)
	
	C = r2(C)
	n_estimators       = r2(n_estimators) 
	learning_rate_hist = r2(learning_rate_hist)
	learning_rate_cat  = r2(learning_rate_cat)
	max_leaf_nodes     = r2(max_leaf_nodes)
	keep_percentage    = r2(keep_percentage)

	n_estimators   = int(n_estimators)
	max_leaf_nodes = int(max_leaf_nodes)

	optimization_log = pd.read_csv(PATH_TO_OPTIMIZATION_LOG)
	level_two_test_features = pd.read_csv(PATH_TO_OPTIMIZATION_TEST)

	valid_latitude, valid_longitude, valid_error = 0, 0, 0

	try:
		ID = optimization_log.id.values[-1] + 1
	except:
		ID = 0

	ID = int(ID)

	error = 0
	N_FOLDS = 5
	for label in ["latitude", "longitude"]:
		train_features = [column for column in level_one_features.columns.tolist() if label in column]
		test_features  = [column for column in level_one_test_features.columns.tolist() if label in column]

		train_df = level_one_features[train_features]
		test_df  = level_one_test_features[test_features]

		labels 	 = train_df[label]
		train_df = train_df.drop(label, axis = 1, inplace = False)

		y_skf = train_df[label + "_skf"]
		train_df = train_df.drop(label + "_skf", axis = 1, inplace = False)

		features = feature_selection(train_df, labels, keep_percentage = keep_percentage)
		train_df = train_df[features]

		test_features = create_test_features(features)
		test_df = test_df[test_features]

		y_oof  = np.zeros((train_df.shape[0],))
		y_test = np.zeros((test_df.shape[0], N_FOLDS)) 
		skf = StratifiedKFold(n_splits = N_FOLDS, random_state = SEED)
		for (fold, (train_idx, valid_idx)) in enumerate(skf.split(X = train_df, y = y_skf)):
			X_train, y_train = train_df.iloc[train_idx].values, labels.iloc[train_idx].values
			X_valid, y_valid = train_df.iloc[valid_idx].values, labels.iloc[valid_idx].values
			X_test = test_df.values

			scaler = MinMaxScaler()
			scaler.fit(X_train)

			X_train = scaler.transform(X_train)
			X_valid = scaler.transform(X_valid)
			X_test  = scaler.transform(X_test)

			estimators = [
				("SVR", SVR(C = C)),
				("Hist", HistGradientBoostingRegressor(random_state = SEED,  loss = "least_absolute_deviation", learning_rate = learning_rate_hist, max_leaf_nodes = max_leaf_nodes)),
				("Cat", CatBoostRegressor(n_estimators = n_estimators, learning_rate = learning_rate_cat, random_state = SEED, silent = True, loss_function = "MAPE"))
			]
        

			model = VotingRegressor(estimators, n_jobs = -1)
			model.fit(X_train, y_train)

			y_predict = model.predict(X_valid)
			y_test_predict = model.predict(X_test)

			y_test[:, fold] = y_test_predict
			y_oof[valid_idx, ] = y_predict

		oof_mae = mean_absolute_error(labels.values, y_oof)

		error += oof_mae * 0.5
		y_test = np.mean(y_test, axis = 1)

		if label == "latitude":
			valid_latitude = oof_mae
		else:
			valid_longitude = oof_mae

		level_two_test_features["optimized_{}_{}".format(label, ID)] = y_test

	row = [ID, C, n_estimators, learning_rate_hist, learning_rate_cat, max_leaf_nodes, keep_percentage, r4(valid_latitude), r4(valid_longitude), r4(error)]
	optimization_log.loc[len(optimization_log)] = row

	optimization_log.to_csv(PATH_TO_OPTIMIZATION_LOG, index = False)
	level_two_test_features.to_csv(PATH_TO_OPTIMIZATION_TEST, index = False)

	return error * (-1)

def Optimize(bounds, init_points = 32, iterations = 64):
	tic = time.time()
	optimizer = BayesianOptimization(train, bounds, random_state = SEED)
	optimizer.maximize(init_points = init_points, n_iter = iterations, acq = 'ucb', xi = 0.0, alpha = 1e-6)

	results = open("Optimized_Parameters.pickle","wb")
	pickle.dump(optimizer.max, results)
	results.close()

	toc = time.time()
	print("Time to optimize {}'s'".format(toc - tic))

if __name__ == "__main__":

	bounds = {
		"C": (0.1, 20),
		"n_estimators": (100, 2000),
		"learning_rate_hist": (0.01, 1),
		"learning_rate_cat": (0.01, 1),
		"max_leaf_nodes": (4, 64),
		"keep_percentage": (0.5, 1), 
	}

	Optimize(bounds, init_points = 128, iterations = 512)
