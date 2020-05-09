import pandas as pd
pizza = pd.read_csv("G:/Statistics (Python)/Datasets/pizza.csv")

pizza.corr()

import matplotlib.pyplot as plt
plt.scatter(pizza['Promote'],pizza['Sales'])
plt.xlabel("Promotional Expenditure")
plt.ylabel('Sales')
plt.show()

import seaborn as sns
sns.regplot(x='Promote', y='Sales', data=pizza)
plt.show()

X = pizza[['Promote']]
y = pizza['Sales']

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
results=regressor.fit(X, y)
print(results.coef_)
print(results.intercept_)

import numpy as np
ycap = regressor.predict(X)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y, ycap)))
print(mean_absolute_error(y, ycap))
print(r2_score(y, ycap))

#sales = results.intercept_ + results.coef_*120
#sales
#sales = results.intercept_ + results.coef_*121
#sales
