import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

#create x and y
x = df.drop("condition", axis = 1)
y = df["condition"]

#split data into train and test
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

#create a random forest model
rf_model = RandomForestClassifier(
    n_estimators = 900,
    random_state = 42
)

#train the model
rf_model.fit(x_train, y_train)

#make predictions
y_pred = rf_model.predict(x_test)

#print some predictions
print("y predictions: ", y_pred[:10])

#print metrics
print("Accuracy: ", accuracy_score(y_test,y_pred))
#Accuracy:  0.8333333333333334

precision = precision_score(y_test, y_pred)
print("Precision: ",precision)
#Precision:  0.875

print("Recall: ", recall_score(y_test,y_pred))
#Recall:  0.75

print("confusion matrix: ", confusion_matrix(y_test, y_pred))
'''[[29  3]
 [ 7 21]]
 '''



'''
 output:
 y predictions:  [0 1 0 0 0 0 1 1 0 0]
Accuracy:  0.8333333333333334
Precision:  0.875
Recall:  0.75
confusion matrix:  [[29  3]
 [ 7 21]]
'''