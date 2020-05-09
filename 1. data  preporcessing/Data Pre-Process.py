import pandas as pd
import numpy as np
cars = pd.read_csv("F:/Python Material/ML with Python/Datasets/Cars93.csv")

cars.head(n=10)

dum_cars = pd.get_dummies(cars, drop_first=True)

dum_cars.head(n=10)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
y = ['a','b','a','a','c','a','b','b','a','c','a']
trny = lbcode.fit_transform(y)
print(trny)

carsMissing = pd.read_csv("F:/Python Material/ML with Python/Datasets/Cars93Missing.csv")
carsMissing.shape

carsDropNA = carsMissing.dropna()
carsDropNA.shape

# Dummying the data
dum_cars_miss = pd.get_dummies(carsMissing, drop_first=True)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
carsImputed = imp.fit_transform(dum_cars_miss)

df_carsImputed = pd.DataFrame(carsImputed,
                              columns= dum_cars_miss.columns)

dum_cars_miss.shape
carsImputed.shape
df_carsImputed.shape

# Categorical Imputing
from sklearn_pandas import CategoricalImputer
data = np.array(['a', 'b', 'b', np.nan], dtype=object)
imputer = CategoricalImputer()
imputer.fit_transform(data)

from sklearn_pandas import CategoricalImputer
data = np.array(['a', 'b', 'b', np.nan], dtype=object)
imputer = CategoricalImputer(strategy='constant',fill_value="Baby")
imputer.fit_transform(data)


import numpy as np
milk = pd.read_csv("F:/Python Material/Python Course/Datasets/milk.csv",index_col=0)
milk.head()
np.mean(milk), np.std(milk)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(milk)
milkscaled=scaler.transform(milk)
# OR
milkscaled=scaler.fit_transform(milk)

np.mean(milkscaled[:,0]), np.std(milkscaled[:,0])
np.mean(milkscaled[:,1]), np.std(milkscaled[:,1])
np.mean(milkscaled[:,2]), np.std(milkscaled[:,2])
np.mean(milkscaled[:,3]), np.std(milkscaled[:,3])
np.mean(milkscaled[:,4]), np.std(milkscaled[:,4])

# Converting numpy array to pandas
df_milk = pd.DataFrame(milkscaled,columns=milk.columns)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(milk)
minmaxMilk = minmax.transform(milk)
minmaxMilk[1:5,]