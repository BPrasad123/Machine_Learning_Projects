import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import catboost

import imp
import sys


# Manually load the model incase it does now work with import
imp.load_package('catboost', 'C:\python\Lib\site-packages\catboost')
import catboost as cb
print(sys.path)
sys.path.append('C:\python\Lib\site-packages\mypackages')







from catboost import CatBoostRegressor
import pandas as pd
import sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier




ds = pd.read_csv(r'file_name.csv')

cols = ds.columns.tolist()
n = int(cols.index('Y_Col'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
ds = ds[cols]

ds.info()

X = ds.iloc[:, :-1]
y = ds.iloc[:, ds.shape[1]-1]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X.columns
X.info()

categorical_features_indices = np.where(X.dtypes == 'object')[0]

#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test))



result = pd.DataFrame()
result['actual'] = y_test.copy()
result['output'] = model.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt

result['actual'].mean()
rms = sqrt(mean_squared_error(result['actual'], result['output']))

#*****************
# Optimization
#*****************

X_train_C = X_train.copy()
X_test_C = X_test.copy()
y_train_C = y_train.copy()
y_test_C = y_test.copy()


for col in categorical_features_indices:
    X_train_C[X_train_C.columns[col]] = X_train_C[X_train_C.columns[col]].astype('category').cat.codes
    X_test_C[X_test_C.columns[col]] = X_test_C[X_test_C.columns[col]].astype('category').cat.codes



y_train_C.iloc[:] = y_train_C.iloc[:].astype('category').cat.codes
y_test_C.iloc[:] = y_test_C.iloc[:].astype('category').cat.codes


params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.005], 
          'l2_leaf_reg':[15, 10,20],
          'border_count':[150,50,100,200],
#          'ctr_border_count':[50,5,10,20,100,200],
          'thread_count':4}




train_set = X_train_C.copy()
test_set = X_test_C.copy()
train_label = y_train_C.copy()
test_label = y_test_C.copy()

cat_dims = categorical_features_indices

from paramsearch import paramsearch
from itertools import product,chain
from sklearn.model_selection import KFold


def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):
    kf = KFold(n_splits=n_splits,shuffle=True) 
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = cb.CatBoostRegressor(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test_set)==np.ravel(test_labels)))
    return np.mean(res)



def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):
    ps = paramsearch(params)
    # search 'border_count', 'l2_leaf_reg' etc. individually 
    #   but 'iterations','learning_rate' together
    for prms in chain(ps.grid_search(['border_count']),
#                      ps.grid_search(['ctr_border_count']),
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations','learning_rate']),
                      ps.grid_search(['depth'])):
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        # save the crossvalidation result so that future iterations can reuse the best parameters
        ps.register_result(res,prms)
#        print(res,prms,s'best:', ps.bestscore(),ps.bestparam())
    return ps.bestparam()

bestparams = catboost_param_tune(params,train_set,train_label,cat_dims)
















