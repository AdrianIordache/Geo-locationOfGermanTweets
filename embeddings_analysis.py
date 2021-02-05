from utils import *


display(train_df)
features = ["embedding_{}".format(emb) for emb in range(EMBEDDINGS_SIZE)]


tic = time.time()

train_latitude, train_longitude, train_error = 0, 0, 0
valid_latitude, valid_longitude, valid_error = 0, 0, 0

for label in ["latitude", "longitude"]:
	X_train, y_train = train_df[features].values, train_df[label].values
	X_valid, y_valid = valid_df[features].values, valid_df[label].values


	model = Ridge(alpha = 10)
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