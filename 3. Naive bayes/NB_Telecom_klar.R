telecom <- read.csv("F:\\Statistics\\Cases\\Telecom\\Telecom.csv")

library(caret)

set.seed(333)
intrain <- createDataPartition(y=telecom$Response,
                               p=0.7,list = FALSE)

training <- telecom[intrain,   ]
validation <- telecom[-intrain,]

library(klaR)
# Apriori Prob from training set
classifier <- NaiveBayes(Response ~ . ,
                         data = training)

# Posterior Prob of every observation
PredY <- predict(classifier, newdata=validation)

tbl <- table(PredY$class, validation$Response,
            dnn=list('predicted','actual'))
#confusionMatrix(tbl)

# OR
confusionMatrix(tbl,positive = "Y")

##ROC
library(pROC)
plot.roc(validation$Response,PredY$posterior[,2],
         legacy.axes=TRUE,print.auc=TRUE )


### Cross-Entropy
library(MLmetrics)
valresp <- ifelse(validation$Response=="Y",1,0)
LogLoss(PredY$posterior[,2], valresp)

# Predicting
tsttel <- read.csv("F:\\Statistics\\Cases\\Telecom\\testTelecom.csv")
PredY <- predict(classifier, newdata=tsttel)
# #
predt <- data.frame(tsttel,Predicted=PredY$class)
