from utils import *
from feature_selection import *

PATH_TO_LEVEL_ONE = "data/level_one/stage-2/"

PATH_TO_MODELS = PATH_TO_LEVEL_ONE + "level_one_models.csv"
PATH_TO_FEATURES = PATH_TO_LEVEL_ONE + "level_one_features_skf.csv"
PATH_TO_TEST_FEATURES = PATH_TO_LEVEL_ONE + "level_one_test_features.csv"

level_one_models = pd.read_csv(PATH_TO_MODELS)
level_one_features = pd.read_csv(PATH_TO_FEATURES)
level_one_test_features = pd.read_csv(PATH_TO_TEST_FEATURES)


PATH_TO_LEVEL_TWO_MODELS   = "data/level_two/level_two_models.csv"
PATH_TO_LEVEL_TWO_FEATURES = "data/level_two/level_two_features.csv"
PATH_TO_LEVEL_TWO_TEST_FEATURES = "data/level_two/level_two_test_features.csv"

level_two_features = pd.read_csv(PATH_TO_LEVEL_TWO_FEATURES)
level_two_test_features = pd.read_csv(PATH_TO_LEVEL_TWO_TEST_FEATURES)
level_two_models   = pd.read_csv(PATH_TO_LEVEL_TWO_MODELS)

SAVE_TO_LOG = True

ID = level_two_models.id.values[-1] + 1
MODEL = "AdaBoostRegressor(NuSVR(C = 10, nu = 0.8))"
PARAMETERS = " max_features = 0.8, n_estimators = 10"
FEATURES_SELECTION = "All Features -> 57 Features"

train_latitude, train_longitude, train_error = 0, 0, 0
valid_latitude, valid_longitude, valid_error = 0, 0, 0

error = 0
N_FOLDS = 5
for label in ["latitude", "longitude"]:
    train_features = [column for column in level_one_features.columns.tolist() if label in column]
    test_features  = [column for column in level_one_test_features.columns.tolist() if label in column]

    train_df = level_one_features[train_features]
    test_df  = level_one_test_features[test_features]

    labels   = train_df[label]
    train_df = train_df.drop(label, axis = 1, inplace = False)

    y_skf = train_df[label + "_skf"]
    train_df = train_df.drop(label + "_skf", axis = 1, inplace = False)

    # features = feature_selection(train_df, labels, keep_percentage = 1)
    # train_df = train_df[features]

    # test_features = create_test_features(features)
    # test_df = test_df[test_features]

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

        # selector = SelectFromModel(estimator = LinearSVR(C = 10)).fit(X_train, y_train)

        # X_train = selector.transform(X_train)
        # X_valid = selector.transform(X_valid)
        # X_test  = selector.transform(X_test)

        # print("Input features after selection: ", X_train.shape[1])

        model = AdaBoostRegressor(NuSVR(C = 10, nu = 0.8), n_estimators = 20, random_state = SEED)
        model.fit(X_train, y_train)

        y_predict = model.predict(X_valid)
        y_test_predict = model.predict(X_test)

        y_test[:, fold] = y_test_predict
        y_oof[valid_idx, ] = y_predict

    oof_mae = mean_absolute_error(labels.values, y_oof)
    print("MAE Valid {}: {}".format(label, oof_mae))

    error += oof_mae * 0.5
    y_test = np.mean(y_test, axis = 1)


    if SAVE_TO_LOG:
        row = [ID] + [MODEL] + [PARAMETERS] + [round(oof_mae, 4)] + [FEATURES_SELECTION] + [label]
        level_two_models.loc[len(level_two_models)] = row
        level_two_models.to_csv(PATH_TO_LEVEL_TWO_MODELS, index = False)

        level_two_features["feature_{}_{}".format(label, ID)] = y_oof
        level_two_features.to_csv(PATH_TO_LEVEL_TWO_FEATURES, index = False)

        level_two_test_features["feature_test_{}_{}".format(label, ID)] = y_test
        level_two_test_features.to_csv(PATH_TO_LEVEL_TWO_TEST_FEATURES, index = False)



print("OOF Error: {}".format(error))
valid_error = error
