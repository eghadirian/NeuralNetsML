import os
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Fetching Data from Online Resources
download_root = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
housing_path = os.path.join('datasets','housing')
housing_url = download_root+'datasets/housing/housing.tgz'
def fetch_housing_data(housing_url=housing_url, housing_path=housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
def load_housing_data(housing_path=housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)
fetch_housing_data()
housing = load_housing_data()
# PreProcess
# Bias in the train/test bc of small data set: stratification
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']>5, 5, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# Extract train and test sets
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()
# Plotting
plt.scatter(housing['longitude'], housing['latitude'], alpha=0.4, s=housing['population']/100,
            label='population', cmap='jet', c=housing_labels)
# user defined objects: for numerical and categorical treatments
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_per_room=True):
        self.add_bedroom_per_room=add_bedroom_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedroom_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values
# pipeline definition
num_attributes = list(housing.drop('ocean_proximity', axis=1))
cat_attributes = ['ocean_proximity']
num_pipeline = Pipeline([('selector', DataFrameSelector(num_attributes)),
                         ('imputer', SimpleImputer(strategy='median')),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scal', StandardScaler())])
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attributes)),
                         ('encoder', OneHotEncoder())])
feature_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline), ('cat_pipeline', cat_pipeline)])
# grid search
pipe = Pipeline([('preprocess', feature_pipeline), ('model', SVR())])
param_grid = {'model__C': [10**(i-3) for i in range(6)]}
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5)
# fitting and evaluation
grid_search.fit(housing, housing_labels)
print('Test set score: {}'.format(grid_search.score(X_test, y_test)))
plt.show()
