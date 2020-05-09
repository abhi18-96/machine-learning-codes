import h2o
h2o.init()

import numpy as np

df = h2o.import_file("F:/Kaggle/Santander Customer/train.csv")
df.summary()
df.col_names

y = 'TARGET'
x = df.col_names
x.remove(y)
x.remove('ID')
print("Response = " + y)
print("Pridictors = " + str(x))

df['TARGET'] = df['TARGET'].asfactor()
df['TARGET'].levels()

train, test = df.split_frame(ratios=[.7])
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)

from h2o.estimators.gbm import H2OGradientBoostingEstimator
h2o_gbm = H2OGradientBoostingEstimator(distribution = "bernoulli")
h2o_gbm.train(x=x, y= y, training_frame=train, 
                   validation_frame=test, model_id="h2o_GBM")

#y_pred = glm_logistic.predict(test_data=test)

print(h2o_gbm.auc())
print(h2o_gbm.confusion_matrix() )

###########################Tuning######################################
from h2o.grid.grid_search import H2OGridSearch

# GBM hyperparameters
glm_params1 = {'learn_rate': np.linspace(0.0001,0.2,30).tolist(),
                'max_depth': np.arange(10,300,50).tolist()}

h2o_glm = H2OGradientBoostingEstimator(distribution = "bernoulli")

# Train and validate a cartesian grid of GBMs
gbm_grid1 = H2OGridSearch(model=h2o_glm,
                          grid_id='gbm_grid1',
                          hyper_params=glm_params1)
gbm_grid1.train(x=x, y=y,
                training_frame=train,
                seed=1)

# Get the grid results, sorted by validation AUC
gbm_gridperf1 = gbm_grid1.get_grid(sort_by="auc",decreasing=False)
gbm_gridperf1

# Grab the top GBM model, chosen by validation AUC
best_glm1 = gbm_gridperf1.models[0]

h2o.cluster().shutdown()
