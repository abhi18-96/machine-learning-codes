
import pandas as pd
import numpy as np
df = pd.read_csv("F:/Kaggle/Bike sharing Demand/train.csv",parse_dates=['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday

df['season'] = df['season'].astype('category')
df['weather'] = df['weather'].astype('category')
df.drop(columns=['datetime','casual', 'registered'],inplace=True)

dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop('count',axis=1)
y = dum_df['count']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xscaled=scaler.fit_transform(X)
# Import KMeans
from sklearn.cluster import KMeans

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state = 2019)
    model.fit(Xscaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

######## k = 4
model = KMeans(n_clusters=4,random_state = 2019)

# Fit model to points
model.fit(Xscaled)

# Determine the cluster labels of new_points: labels
labels = pd.Series( model.predict(Xscaled)).rename('ClusterID')

XClustered = pd.concat([X,labels],axis='columns')
SS0 = dum_df[XClustered['ClusterID']==0]
SS1 = dum_df[XClustered['ClusterID']==1]
SS2 = dum_df[XClustered['ClusterID']==2]
SS3 = dum_df[XClustered['ClusterID']==3]

####### Models Building #######
from sklearn.ensemble import RandomForestRegressor

model_0 = RandomForestRegressor(random_state=2019)
X=SS0.drop(['count'],axis='columns')
y=SS0['count']
model_0.fit(X,y)

model_1 = RandomForestRegressor(random_state=2019)
X=SS1.drop(['count'],axis='columns')
y=SS1['count']
model_1.fit(X,y)

model_2 = RandomForestRegressor(random_state=2019)
X=SS2.drop(['count'],axis='columns')
y=SS2['count']
model_2.fit(X,y)

model_3 = RandomForestRegressor(random_state=2019)
X=SS3.drop(['count'],axis='columns')
y=SS3['count']
model_3.fit(X,y)
################################

################Pre-Process on Test set
df2 = pd.read_csv("F:/Kaggle/Bike sharing Demand/test.csv",parse_dates=['datetime'])
df2['year'] = df2['datetime'].dt.year
df2['month'] = df2['datetime'].dt.month
df2['day'] = df2['datetime'].dt.day
df2['hour'] = df2['datetime'].dt.hour
df2['weekday']=df2['datetime'].dt.weekday

df2['season'] = df2['season'].astype('category')
df2['weather'] = df2['weather'].astype('category')
datetimeOriginal = df2['datetime']
df2.drop(columns=['datetime'],inplace=True)

dum_df2 = pd.get_dummies(df2, drop_first=True)

X2 = dum_df2
X2scaled=scaler.fit_transform(X2)
labels2 = pd.Series( model.predict(X2scaled)).rename('ClusterID')

dum_df2 = pd.concat([X2,labels2,datetimeOriginal],axis='columns')
SS20 = dum_df2[dum_df2['ClusterID']==0]
SS21 = dum_df2[dum_df2['ClusterID']==1]
SS22 = dum_df2[dum_df2['ClusterID']==2]
SS23 = dum_df2[dum_df2['ClusterID']==3]

### For preserving the datetime sequences
SS20_datetimeOriginal = SS20['datetime']
SS21_datetimeOriginal = SS21['datetime']
SS22_datetimeOriginal = SS22['datetime']
SS23_datetimeOriginal = SS23['datetime']

SS20 = SS20.drop(['ClusterID','datetime'],axis='columns')
SS21 = SS21.drop(['ClusterID','datetime'],axis='columns')
SS22 = SS22.drop(['ClusterID','datetime'],axis='columns')
SS23 = SS23.drop(['ClusterID','datetime'],axis='columns')

##### Apply the models
pred0 = model_0.predict(SS20)
pred1 = model_1.predict(SS21)
pred2 = model_2.predict(SS22)
pred3 = model_3.predict(SS23)

clust0 = pd.DataFrame({'datetime':SS20_datetimeOriginal, 'count':pred0})
clust1 = pd.DataFrame({'datetime':SS21_datetimeOriginal, 'count':pred1})
clust2 = pd.DataFrame({'datetime':SS22_datetimeOriginal, 'count':pred2})
clust3 = pd.DataFrame({'datetime':SS23_datetimeOriginal, 'count':pred3})

submit = pd.concat([clust0,clust1,clust2,clust3],ignore_index=True)
submit2 = submit.sort_values(by=['datetime'])
submit2.to_csv('F:/Kaggle/Bike sharing Demand/submit_clustered_rf.csv',
               index=False)
