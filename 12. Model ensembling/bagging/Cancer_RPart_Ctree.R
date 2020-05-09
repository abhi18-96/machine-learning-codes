breastAll <- read.csv("F:\\Statistics\\Cases\\Wisconsin\\BreastCancer.csv")

breast <- breastAll[,-1]

library(caret)
set.seed(1992)
intrain<-createDataPartition(y=breast$Class,p=0.7,list=FALSE)

training   <- breast[ intrain , ]
validation <- breast[-intrain , ]

library(rpart)

dtree <- rpart(Class ~ ., data=training,
               method="class")

library(rpart.plot)

rpart.plot(dtree , type=5, extra=3)

dtree.pred <- predict(dtree, newdata=validation, type="class")
tbl <- table(dtree.pred, validation$Class,
                    dnn=c("Predicted", "Actual"))
confusionMatrix(tbl,positive = "Malignant")

##### ROC #####
dtree.pred.prob <- predict(dtree, newdata=validation,
                           type="prob")
library(pROC)
plot.roc(validation$Class, dtree.pred.prob[,2], print.auc=TRUE ,
         col="red", main="RPART",legacy.axes=TRUE)

##### Conditional Inference Tree ########

library(partykit)
fit.ctree <- ctree(Class~., data=training)

plot(fit.ctree, main="Conditional Inference Tree",
     type="simple")

plot(fit.ctree, main="Conditional Inference Tree",
     type="extended")

ctree.pred <- predict(fit.ctree, validation, type="response")
ctree.perf <- table( ctree.pred,validation$Class ,
                    dnn=c("Predicted", "Actual"))

confusionMatrix(ctree.perf,positive = "Malignant")

##### ROC ####

ctree.prob <- predict(fit.ctree, validation,
                      type= "prob")
plot.roc(validation$Class, ctree.prob[,2] ,
         print.auc=TRUE , col="green" , main="CTREE",legacy.axes=TRUE)

############ Tuning ###############
mygrid = expand.grid(maxdepth=2:15)
model_cv <- train(Class ~ .,
                  data =breast,method="rpart2",
                  tuneGrid = mygrid,
                  trControl = trainControl(method = "cv",
                                           number = 5))

model_cv
plot(model_cv)
