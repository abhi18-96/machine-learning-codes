import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Datasets/RidingMowers.csv")

dum_df = pd.get_dummies(df)
dum_df = dum_df.drop('Response_Not Bought', axis=1)

from sklearn.neighbors import KNeighborsClassifier

X = dum_df.iloc[:,0:2]
y = dum_df.iloc[:,2]

from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'n_neighbors': np.array([1,3,5,7,9])}
print(parameters)

############### Test for single k at a time ###############
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
#kfold = StratifiedKFold(n_splits=5, random_state=42)
#log_reg = KNeighborsClassifier(n_neighbors=9)
#results = cross_val_score(log_reg, X, y, cv=kfold, 
#                          scoring='roc_auc')
#print(results)
#print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

############## Testing all Ks at a time ###################

knn = KNeighborsClassifier()
cv = GridSearchCV(knn, param_grid=parameters,cv=5)

#OR
cv = GridSearchCV(knn, param_grid=parameters,
                  cv=5,scoring='roc_auc')
#OR
cv = GridSearchCV(knn, param_grid=parameters,
                  cv=5,scoring='neg_log_loss')

cv.fit( X , y )

results = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)
