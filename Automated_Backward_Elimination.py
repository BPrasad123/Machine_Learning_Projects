X['dummy'] = 1
X = X.reindex(columns=['dummy'] + X.columns[:-1].tolist())
X = pd.get_dummies(X, drop_first = True)
yd = df.iloc[:, df.shape[1]-1].copy()

import statsmodels.formula.api as sm
def backwardElimination(x, SL, cols):
    numVars = len(x[0])
    temp = np.zeros((x.shape[0],numVars)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    v = cols.pop(j)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        cols = cols.insert(j, v)
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
    
cols = list(range(0, len(X.columns)))
SL = 0.05
y = yd.iloc[:].values
X_opt = X.iloc[:, cols].values
X_Modeled = backwardElimination(X_opt, SL, cols)