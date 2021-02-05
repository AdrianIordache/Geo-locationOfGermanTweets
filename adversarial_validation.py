from utils import *

display(train_df.head(n = 5))

train_copy = train_df.drop(["id", "latitude", "longitude", "text"], axis = 1, inplace = False)
valid_copy = valid_df.drop(["id", "latitude", "longitude", "text"], axis = 1, inplace = False)
test_copy  = test_df.drop(["id", "text"], axis = 1, inplace = False)

print(len(train_copy))
print(len(valid_copy))
print(len(test_copy))

def do_adversarila_validation(train: pd.DataFrame, test: pd.DataFrame):
	train["Label"] = 0
	test["Label"] = 1

	adversarial_data = pd.concat([train, test], axis = 0)
	adversarial_data = adversarial_data.sample(frac=1).reset_index(drop=True)

	adversarial_train = adversarial_data.drop("Label", axis = 1, inplace = False).values
	adversarial_label = adversarial_data['Label'].values

	train, test, y_train, y_test = train_test_split(adversarial_train, adversarial_label, test_size = 0.25, random_state = SEED, shuffle=True)

	train = lgb.Dataset(train, label = y_train)
	test = lgb.Dataset(test, label = y_test)


	param = {'num_leaves': 30,
		'min_data_in_leaf': 30, 
		'objective':'binary',
		'max_depth': 5,
		'learning_rate': 0.05,
		"min_child_samples": 20,
		"boosting": "gbdt",
		"feature_fraction": 0.9,
		"bagging_freq": 1,
		"bagging_fraction": 0.9 ,
		"bagging_seed": SEED,
		"seed": SEED,
		"metric": 'auc',
		"verbosity": -1}


	num_round = 2000
	clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval = 50, early_stopping_rounds = 1000)


print("Adversarial Validation for Train and Valid")
do_adversarila_validation(train_copy, valid_copy)

print("Adversarial Validation for Train and Test")
do_adversarila_validation(train_copy, test_copy)

print("Adversarial Validation for Valid and Test")
do_adversarila_validation(valid_copy, test_copy)