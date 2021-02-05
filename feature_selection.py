from utils import *

def feature_selection(data: pd.DataFrame, labels: pd.Series, keep_percentage: float = 0.5) -> [str]:
    features = data.columns.tolist()
    correlations = data.corr()
    metrics = []
    for feature in features:
        oof    = data[feature].values
        
        error = mean_absolute_error(labels, oof)
        mean_correlation = np.mean(correlations[feature])

        metric = 0.5 * error + 0.5 * mean_correlation

        metrics.append((feature, metric))

    sorted_features = sorted(metrics, key = lambda x: x[1])
    final_features = [feature for (idx, (feature, metric)) in enumerate(sorted_features) if idx <= int(keep_percentage * len(features))]

    return final_features

def test_features_optimization(path_to_file: str = None) -> None:
    test_features = pd.read_csv(PATH_TO_TEST_FEATURES)

    latitude_names  = [column for column in test_features.columns.tolist() if "latitude" in column]
    longitude_names = [column for column in test_features.columns.tolist() if "longitude" in column]

    latitude  = test_features[latitude_names]
    longitude = test_features[longitude_names]

    # display(latitude.corr())
    # display(longitude.corr())

    # print(np.min(latitude.corr().values[0]))
    # print(np.min(longitude.corr().values[0]))

    blending = copy.deepcopy(submission)
    blending["lat"]  = np.mean(latitude.values,  axis = 1)
    blending["long"] = np.mean(longitude.values, axis = 1)

    blending.to_csv("submission_id_16.txt", index = False)

if __name__ == "__main__":
    
    PATH_TO_TEST_FEATURES = "data/optimization/stage-2/level_two_test.csv"

    test_features_optimization(PATH_TO_TEST_FEATURES)