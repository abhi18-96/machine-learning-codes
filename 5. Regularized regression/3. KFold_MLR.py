import pandas as pd
df = pd.read_csv("G:/Statistics (Python)/Cases/Concrete Strength/Concrete_Data.csv")

X = df.iloc[:,:8]
y = df['Strength']

from sklearn.model_selection import train_test_split 
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

############################################################
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#########################K-Fold CV####################################
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=42)
results = cross_val_score(regressor, X, y, cv=kfold, 
                          scoring='neg_mean_absolute_error')
MAEs = results*(-1)
print(MAEs)
print("MAE: %.2f" % (MAEs.mean()))

##########################################################