import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#load csv
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

#create x and y
x = df.drop("condition", axis=1)
y = df["condition"]

#create a split
x_train, y_train, x_test, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#list all the numerical features — StandarScalar
numerical_features=[
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"]

#list all categorical features — OneHotEncoder
categorical_features=[
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

#numerical features
