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

