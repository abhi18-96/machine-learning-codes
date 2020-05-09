import numpy as np
import pandas as pd

df_original = pd.read_csv("G:/Statistics (Python)/Cases/Wisconsin/BreastCancer.csv")
df = df_original.iloc[:,:-1]
X = df.drop(['Code'],axis=1)
y = df_original.iloc[:,-1] 

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=100,random_state=2019)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(X)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

xs_B = xs[y=="Benign"]
xs_M = xs[y=="Malignant"]

ys_B = ys[y=="Benign"]
ys_M = ys[y=="Malignant"]

import matplotlib.pyplot as plt

plt.scatter(xs_B,ys_B,c="green",label="Benign")
plt.scatter(xs_M,ys_M,c="red",label="Malignant")
plt.legend()
plt.title("Malignancy")
plt.show()

########################################################
X_tsne = pd.DataFrame(tsne_features,columns = ['C1', 'C2'])
