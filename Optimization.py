# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


# Importing the dataset
ds = pd.read_csv(r'input_file.csv')

cat_features = np.where(ds.dtypes == 'object')[0]
cat_features = list(cat_features)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

for cat in cat_features:
    ds.iloc[:, cat] = labelencoder.fit_transform(ds.iloc[:, cat])





#ds = pd.get_dummies(ds, drop_first = True)

cols = ds.columns.tolist()
n = int(cols.index('Y_col'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
ds = ds[cols]

X = ds.iloc[:, :-1]
y = ds.iloc[:, ds.shape[1]-1]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import regression_report
from sklearn.grid_search import GridSearchCV



#baseline = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=10)
gbrt=GradientBoostingRegressor(n_estimators=100)
gbrt.fit(X_train, y_train)
y_pred=gbrt.predict(X_test)


print("R-squared for Train: %.2f", gbrt.score(X_train, y_train))
print("R-squared for Test: %.2f", gbrt.score(X_test, y_test))

from sklearn.cross_validation import ShuffleSplit, train_test_split

def GradientBooster(param_grid, n_jobs):
    estimator = GradientBoostingRegressor()
    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    classifier.fit(X_train, y_train)
    print("Best Estimator learned through GridSearch")
    print(classifier.best_estimator_)
    return (cv, classifier.best_estimator_)



param_grid={'n_estimators':[100],
            'learning_rate': [0.05, 0.02, 0.01, 0.005],
            'max_depth':[7,9,11],
            'min_samples_leaf':[8,10,12],
            'max_features':[1.0,0.3,0.1]
            }

n_jobs=3

cv,best_est=GradientBooster(param_grid, n_jobs)





