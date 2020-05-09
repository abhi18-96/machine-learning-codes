concrete <- read.csv("F:/Statistics/Cases/Concrete Strength/Concrete_Data.csv")

library(caret)

set.seed(2018)
intrain <- createDataPartition(y=concrete$Strength,p=0.7,list = F)

training <- concrete[intrain,]
validation <- concrete[-intrain,]

library(h2o)
h2o.init()

train.hex <- as.h2o(training)
valid.hex <- as.h2o(validation)

model.rf <- h2o.randomForest(x=1:8,y=9,
                             training_frame = train.hex)
pred.RF <- h2o.predict(model.rf,newdata =valid.hex)

pred.rf.df <- as.data.frame(pred.RF)
postResample(pred = pred.rf.df$predict,obs = validation$Strength)
# OR
h2o.performance(model.rf,newdata = valid.hex)

h2o.shutdown()
