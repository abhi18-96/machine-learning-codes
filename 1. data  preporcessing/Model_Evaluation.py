
import pandas as pd
import numpy as np
predicted = np.array(["Y","Y","N","N","Y","N","Y","Y","N","N","N","N","Y","N"], 
                     dtype=object)

existing = np.array(["Y","N","N","N","Y","N","Y","N","Y","Y","Y","N","Y","N"], 
                     dtype=object)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(existing, predicted))
print(classification_report(existing, predicted))
print(accuracy_score(existing,predicted))

from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: Model 1
y_pred_prob1 = np.array([0.82,0.7,0.18,0.55,0.26,0.74,0.85,0.08,
                        0.07,0.38,0.15,0.44,0.92,0.11])

y_test = pd.get_dummies(existing, drop_first=True)
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob1)

############ Plot ROC curve ############
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
########################################
roc_auc_score(y_test, y_pred_prob1)

# Compute predicted probabilities: Model 2
y_pred_prob2 = np.array([0.82,0.6,0.03,0.45,0.76,0.54,0.85,0.08,
                        0.57,0.68,0.45,0.44,0.72,0.11])

y_test = pd.get_dummies(existing, drop_first=True)
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob2)

############ Plot ROC curve ############
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
########################################
roc_auc_score(y_test, y_pred_prob2)


######################## Log Loss ########################
from sklearn.metrics import log_loss
log_loss(y_test, y_pred_prob1)
log_loss(y_test, y_pred_prob2)

#################################################################
y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)  


y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)  

y_pred = np.array([13.4,45.4,89.3,90.4,87.3,45.9,16.5])
y_true = np.array([12.3,46.4,90,100.4,86.3,46,17])
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)  
