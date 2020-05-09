import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df.iloc[:,1:11], drop_first=True)

X = dum_df
y = df.iloc[:,1]

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
import numpy as np

parameters = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}
# OR
parameters = {'n_neighbors': np.linspace(1,10,10).astype(int)}

print(parameters)

knn = KNeighborsRegressor()

cv = GridSearchCV(knn, param_grid=parameters,
                  cv=5,scoring='neg_mean_absolute_error')

cv.fit( X , y )

pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)
