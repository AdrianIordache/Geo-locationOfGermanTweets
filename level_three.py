from utils import *
from feature_selection import *


PATH_TO_LEVEL_TWO = "data/level_two/"

PATH_TO_MODELS = PATH_TO_LEVEL_TWO + "level_two_models.csv"
PATH_TO_FEATURES = PATH_TO_LEVEL_TWO + "level_two_features_skf.csv"
PATH_TO_TEST_FEATURES = PATH_TO_LEVEL_TWO + "level_two_test_features.csv"

level_two_models = pd.read_csv(PATH_TO_MODELS)
level_two_features = pd.read_csv(PATH_TO_FEATURES)
level_two_test_features = pd.read_csv(PATH_TO_TEST_FEATURES)

PATH_TO_BAYESIAN_PARAMETERS = "data/optimization/stage-2/Optimized_Parameters.pickle"

file = open(PATH_TO_BAYESIAN_PARAMETERS, "rb")
bayes = pickle.load(file)

r2 = lambda x: np.round(x, 2)
r4 = lambda x: np.round(x, 4)

C = r2(bayes['params']['C'])
n_estimators       = r2(bayes['params']['n_estimators']) 
learning_rate_hist = r2(bayes['params']['learning_rate_hist'])
learning_rate_cat  = r2(bayes['params']['learning_rate_cat'])
max_leaf_nodes     = r2(bayes['params']['max_leaf_nodes'])
keep_percentage    = r2(bayes['params']['keep_percentage'])

n_estimators   = int(n_estimators)
max_leaf_nodes = int(max_leaf_nodes)

save_to_log = False
description = "Bayesian Optimization for parameters"
index = logger.id.values[-1] + 1
submission_name = "submission_id_{}.txt".format(index)

train_latitude, train_longitude, train_error = 0, 0, 0
valid_latitude, valid_longitude, valid_error = 0, 0, 0

error = 0
N_FOLDS = 5
for label in ["latitude", "longitude"]:
    train_features = [column for column in level_two_features.columns.tolist() if label in column]
    test_features  = [column for column in level_two_test_features.columns.tolist() if label in column]

    train_df = level_two_features[train_features]
    test_df  = level_two_test_features[test_features]

    labels   = train_df[label]
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
    oof_mse = mean_squared_error(labels.values, y_oof)
    print("MAE Valid {}: {}".format(label, oof_mae))
    print("MSE Valid {}: {}".format(label, oof_mse))

    error += oof_mae * 0.5
    y_test = np.mean(y_test, axis = 1)

    if label == "latitude":
        valid_latitude = oof_mae
        if save_to_log:
            add_to_submission_file(y_test, "lat")
    else:
        valid_longitude = oof_mae
        if save_to_log:
            add_to_submission_file(y_test, "long", path = submission_name, save_file = True)


print("OOF Error: {}".format(error))
valid_error = error

test_error = 0

if save_to_log:
    row = [train_latitude, valid_latitude, train_longitude, valid_longitude, train_error, valid_error, test_error]
    row = [round(row[idx]) for idx in range(len(row))]
    row = [index] + row + [description]
    logger.loc[len(logger)] = row
    logger.to_csv(PATH_TO_LOGGER, index = False)