from tsfresh import extract_relevant_features, extract_features
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction import FeatureExtractionSettings
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import time

def read_data(features, labels):
    colnames = ['time_index', 'value', 'id']
    label_colnames = ['value', 'id']
    features = pd.read_csv(features, header=None, names=colnames)
    y = pd.read_csv(labels, header = None, names = label_colnames).sort_values(by = 'id')
    labels = pd.Series(y['value'].values)
    index = y['id']
    labels.index = index
    return (features, labels)

X_train, y_train = read_data('data/train_features.csv', 'data/train_labels.csv')


print ">>>>>>>>>> RUNNING FEATURES EXTRACTION"
start_time1 = time.time()
X_train_filtered = extract_relevant_features(timeseries_container=X_train, y = y_train, column_id='id',
                                             column_sort='time_index', column_value='value')
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('regressor', GradientBoostingRegressor())])
print("--- %s seconds ---" % (time.time() - start_time1))

print ">>>>>>>>>> FITTING PIPELINE"
start_time2 = time.time()
pipeline.fit(X_train_filtered, y_train)
print("--- %s seconds ---" % (time.time() - start_time2))


print ">>>>>>>>>>> RUNNING PREDICTION"
X_test, y_test = read_data('data/test_features.csv', 'data/test_labels.csv')
print "X_test.columns = {}".format(str(X_test.columns))
print "X_train.columns = {}".format(str(X_train.columns))

settings = FeatureExtractionSettings.from_columns(X_train_filtered.columns)
X_test_filtered = extract_features(timeseries_container=X_test, column_id='id', column_sort='time_index', column_value='value',
                                   feature_extraction_settings=settings)
print "X_train_filtered.columns = {}".format(str(X_train_filtered.columns))
print "X_test_filtered_columns = {}".format(str(X_test_filtered.columns))
print "X_train_filtered.numcols = {}".format(str(len(X_train_filtered.columns)))
print "X_test_filtered.numcols = {}".format(str(len(X_test_filtered.columns)))
y_predict = pipeline.predict(X_test_filtered)
result = pd.DataFrame({'y_test': y_test.values, 'y_predict': y_predict})
result.to_csv('prediction_result.csv')
print "mean_squared_error = {}".format(str(mean_squared_error(y_test, y_predict)))
print "mean_absolute_error = {}".format(str(mean_absolute_error(y_test, y_predict)))

