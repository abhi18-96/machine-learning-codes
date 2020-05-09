import pandas as pd
import numpy as np

milk = pd.read_csv("G:/Statistics (Python)/Datasets/milk.csv",
                   index_col=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,columns=milk.columns)

# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(milkscaled,method='average')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=50,
           leaf_font_size=10,
)

plt.show()

####### Using Mahalonobis Distance Method #############

# Calculate the linkage: mergings
mergings = linkage(milkscaled,method='average',metric='mahalanobis')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=60,
           leaf_font_size=10,
)
plt.show()