import pandas as pd
import numpy as np
df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df.iloc[:,1:11], drop_first=True)

from sklearn.model_selection import train_test_split 

X = dum_df
y = df.iloc[:,0]
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state=2018)

#### Model-1 linear regression  ####
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
pred_lr = model_lr.predict(X_train)
pred_lr [pred_lr <0] = 0

#### Model-2 SVR 'linear'  ######
from sklearn.svm import SVR
model_svrl = SVR(kernel='linear')
model_svrl.fit(X_train,y_train)
pred_svrl=model_svrl.predict(X_train)
pred_svrl[pred_svrl <0] = 0

#### Model-3 SVR 'radial' ######
model_svrr = SVR(kernel='rbf')
model_svrr.fit(X_train,y_train)
pred_svrr=model_svrr.predict(X_train)
pred_svrr[pred_svrr <0] = 0

#### Model-4 Decision Tree Regressor ######
from sklearn.tree import DecisionTreeRegressor
model_dtr= DecisionTreeRegressor()
model_dtr.fit(X_train,y_train)
pred_dtr=model_dtr.predict(X_train)
pred_dtr[pred_dtr<0]=0

###### Combining all the predictions #####
pred_lr=pd.Series(pred_lr)
pred_svrl=pd.Series(pred_svrl)
pred_svrr=pd.Series(pred_svrr)
pred_dtr=pd.Series(pred_dtr)
comb_pred=pd.concat([pred_lr,pred_svrl,pred_svrr,pred_dtr],axis=1)
#(pred_lr,pred_svrl,pred_svrr,pred_svrs,pred_dt
comb_pred.columns=['pred_lr','pred_svrl','pred_svrr','pred_dtr']

###### Now level 2 model RF ############################################################
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=1200)
model.fit(comb_pred,y_train)

########### Predicting on test set####################################
pred_lr = model_lr.predict(X_test)
pred_lr [pred_lr <0] = 0

pred_svrl=model_svrl.predict(X_test)
pred_svrl[pred_svrl <0] = 0

pred_svrr=model_svrr.predict(X_test)
pred_svrr[pred_svrr <0] = 0

pred_dtr=model_dtr.predict(X_test)
pred_dtr[pred_dtr<0]=0

###### Combining all the predictions for test set #####
pred_lr=pd.Series(pred_lr)
pred_svrl=pd.Series(pred_svrl)
pred_svrr=pd.Series(pred_svrr)
pred_dtr=pd.Series(pred_dtr)
comb_pred_test=pd.concat([pred_lr,pred_svrl,pred_svrr,pred_dtr],axis=1)
#(pred_lr,pred_svrl,pred_svrr,pred_svrs,pred_dt
comb_pred_test.columns=['pred_lr','pred_svrl','pred_svrr','pred_dtr']

pred_testdata=model.predict(comb_pred_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, pred_testdata))
print(mean_absolute_error(y_test, pred_testdata))
print(r2_score(y_test, pred_testdata))
