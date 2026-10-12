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

