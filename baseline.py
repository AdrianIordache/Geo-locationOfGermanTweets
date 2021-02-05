from utils import *
from translator import *
from text_cleaning import *

# train_df = TextCleaner(train_df).data
# valid_df = TextCleaner(valid_df).data
# test_df  = TextCleaner(test_df).data

train_df = copy.deepcopy(preprocessed_train)
valid_df = copy.deepcopy(preprocessed_valid)
test_df  = copy.deepcopy(preprocessed_test)

save_to_log = True
description = "removed all emojis, ngram_range -> (1, 7) and analyzer = 'char_wb'"
index = logger.id.values[-1] + 1
submission_name = "submission_id_{}.txt".format(index)

display(train_df.head(n = 5))

estimators = [
	("lr", LinearRegression(n_jobs = -1)),
	("svr", LinearSVR(random_state = SEED)),
	("ridge", Ridge(random_state = SEED)),
	("svr_2", SVR(C = 0.1, epsilon=0.2)),
	("lasso", Lasso(random_state = SEED)),
	("elastic", ElasticNet(random_state = SEED)),
	("rf", RandomForestRegressor(n_estimators = 30, random_state = SEED, n_jobs = -1)),
	("extra", ExtraTreesRegressor(n_estimators = 30, random_state = SEED, n_jobs = -1)),
	("lgbm", LGBMRegressor(n_estimators = 30, n_jobs = -1, random_state = SEED)),
]


tic = time.time()

train_latitude, train_longitude, train_error = 0, 0, 0
valid_latitude, valid_longitude, valid_error = 0, 0, 0

for label in ["latitude", "longitude"]:
	X_train, y_train = train_df.final_text.values, train_df[label].values
	X_valid, y_valid = valid_df.final_text.values, valid_df[label].values

	tfv = TfidfVectorizer(ngram_range = (1, 7), analyzer = 'char_wb')
	tfv.fit(X_train)

	X_train = tfv.transform(X_train)
	X_valid = tfv.transform(X_valid)

	print(X_train.shape)
	
	model = StackingRegressor(estimators = estimators, final_estimator = XGBRegressor(n_jobs = -1, random_state = SEED, learning_rate = 0.7, n_estimators = 50), n_jobs = 1)
	model.fit(X_train, y_train)

	y_predict_train = model.predict(X_train)
	y_predict_valid = model.predict(X_valid)

	mae_train = mean_absolute_error(y_train, y_predict_train)
	mae_valid = mean_absolute_error(y_valid, y_predict_valid)

	print("MAE Train {}: {}".format(label, mae_train))
	print("MAE Valid {}: {}".format(label, mae_valid))

	if label == "latitude":
		train_latitude, valid_latitude = mae_train, mae_valid
	else:
		train_longitude, valid_longitude = mae_train, mae_valid


train_error = (train_latitude + train_longitude) / 2
valid_error = (valid_latitude + valid_longitude) / 2

print("Train Error: {}".format(train_error))
print("Valid Error: {}".format(valid_error))

test_error = 0

toc = time.time()
print("[training] -> time {}'s".format(toc - tic))

if save_to_log:
	row = [train_latitude, valid_latitude, train_longitude, valid_longitude, train_error, valid_error, test_error]
	row = [round(row[idx]) for idx in range(len(row))]
	row = [index] + row + [description]
	logger.loc[len(logger)] = row
	logger.to_csv("logger.csv", index = False)



data = pd.concat([train_df, valid_df], axis = 0)
data = data.sample(frac = 1).reset_index(drop=True)

for label in ["latitude", "longitude"]:
	X_train, y_train = data.final_text.values, data[label].values
	X_test = test_df.final_text.values

	tfv = TfidfVectorizer(ngram_range = (1, 7), analyzer = 'char_wb')
	tfv.fit(X_train)

	X_train = tfv.transform(X_train)
	X_test  = tfv.transform(X_test)
	
	print("works...")
	model = StackingRegressor(estimators = estimators, final_estimator = XGBRegressor(n_jobs = -1, random_state = SEED, learning_rate = 0.7, n_estimators = 50), n_jobs = 1)
	model.fit(X_train, y_train)

	if label == "latitude":
		predict_to_submission_file(model, X_test, "lat")
	else:
		predict_to_submission_file(model, X_test, "long", path = submission_name, save_file = True)

