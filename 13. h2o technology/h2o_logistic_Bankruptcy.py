import h2o
h2o.init()

df = h2o.import_file("G:/Statistics (Python)/Cases/Bankruptcy/Bankruptcy.csv")
#df.summary()
df.col_names

y = 'D'
x = df.col_names[3:]
#x.remove(y)
#x.remove('ID')
print("Response = " + y)
print("Pridictors = " + str(x))

df['D'] = df['D'].asfactor()
df['D'].levels()

train,  test = df.split_frame(ratios=[.8])
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")
glm_logistic.train(x=x, y= y, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()
#from h2o.model.metrics_base import H2OBinomialModelMetrics  
#print(glm_logistic.model_performance())

print(glm_logistic.auc() )
print(glm_logistic.confusion_matrix() )
h2o.cluster().shutdown()
