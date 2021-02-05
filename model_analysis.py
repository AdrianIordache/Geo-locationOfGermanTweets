from utils import *

train_df = copy.deepcopy(preprocessed_train)
valid_df = copy.deepcopy(preprocessed_valid)

ID = model_analyzer.id.values[-1] + 1

TEXT = "original"
MODEL = "Ridge"
OBSERVATION = "TfidfVectorizer -> analyzer = 'char_wb', ngram_range = (1, 3)"

tic = time.time()


train_latitude, train_longitude, train_error = 0, 0, 0
valid_latitude, valid_longitude, valid_error = 0, 0, 0

for label in ["latitude", "longitude"]:
	X_train, y_train = train_df.text.values, train_df[label].values
	X_valid, y_valid = valid_df.text.values, valid_df[label].values

	tfv = TfidfVectorizer(analyzer = "char", ngram_range = (1, 5))
	tfv.fit(X_train)

	X_train = tfv.transform(X_train)
	X_valid = tfv.transform(X_valid)


	model = Ridge(alpha = 5, random_state = SEED)
	print("Train input shape: {}".format(X_train.shape))
	FEATURES = X_train.shape[1]
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

toc = time.time()
print("[training] -> time {}'s".format(toc - tic))

row = [train_latitude, valid_latitude, train_longitude, valid_longitude, train_error, valid_error]
row = [round(row[idx], 5) for idx in range(len(row))]
row = [ID] + [MODEL]+ row + [OBSERVATION] + [FEATURES] + [TEXT]

model_analyzer.loc[len(model_analyzer)] = row
model_analyzer.to_csv(PATH_TO_MODEL_ANALYSIS, index = False)
