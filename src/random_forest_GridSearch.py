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
    test_split = 0.2,
    random_state = 42,
    stratify = y
)

#create a rf model
rf_model = RandomForestClassifier(random_state = 42)

