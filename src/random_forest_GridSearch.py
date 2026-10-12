import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

#load dataset
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

#seperate x and y
x = df.drop("condition", axis = 1)
y = df["condition"]

#create train test split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

#create a rf model
rf_model = RandomForestClassifier(random_state = 42)

#hyperparameters
param_grid = {
    "n_estimators" : [100,300],
    "max_depth" : [None, 5, 10],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf" : [1,2,4],
    "max_features" : ["sqrt", "log2"],
    "class_weight" : [None, "balanced"]
}

#grid search
grid_search = GridSearchCV(
    estimator = rf_model,
    param_grid = param_grid,
    cv = 5,
    scoring = "recall",
    n_jobs = -1
)

grid_search.fit(x_train,y_train)

#print best param
print("best param : ", grid_search.best_params_)
print("best score: ", grid_search.best_score_)

#select the best estimator
best_model = grid_search.best_estimator_

#cal y predict
y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Confusion Matrix:", cm)