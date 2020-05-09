# Import dataset and display summary

library(h2o)
h2o.init()
cancer.hex = h2o.importFile("F:\\h2o Technology\\BreastCancer.csv",parse = TRUE)

summary(cancer.hex)


cancer.split = h2o.splitFrame(data = cancer.hex,ratios = 0.70)
cancer.train = cancer.split[[1]]
cancer.test = cancer.split[[2]]


# Display a summary using table-like functions

h2o.table(cancer.train$Class)
h2o.table(cancer.test$Class)


# Set predictor and response variables
Y = "Class"
X = c("Clump", "UniCell_Size", "Uni_CellShape", "MargAdh", "SEpith",
          "BareN", "BChromatin", "NoemN", "Mitoses" )

# Define the data for the model and display the results

cancer.glm <- h2o.glm(training_frame =cancer.train, x=X, y=Y, family = "binomial", alpha = 0.5)

# View model information: training statistics,performance, important variables
summary(cancer.glm)

# Predict using GLM model
pred = h2o.predict(object = cancer.glm, newdata = cancer.test)

pred
df_pred <- as.data.frame(pred)
h2o.shutdown()
