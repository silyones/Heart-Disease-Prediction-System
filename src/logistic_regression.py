import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

#load csv
df = pd.read_csv("data/processed/heart_disease_cleaned.csv")

#create x and y
x = df.drop("condition", axis=1)
y = df["condition"]

#create a split
x_train, x_test, y_train, y_test = train_test_split(
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

#preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(),categorical_features)
    ]
)

#create logestic regerssion pipeline
logistic_model=Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=1000))
    ]
)

#train model
logistic_model.fit(x_train,y_train)

#after the model is trained, it needs to predict on x_test
y_pred = logistic_model.predict(x_test)

#display some predictions
print("model predictions",y_pred[:10])

#calculate accuracy
accuracy = accuracy_score(y_test,y_pred)
print("accuracy",accuracy)
#result : accuracy 0.9166666666666666

#precision score
precision = precision_score(y_test,y_pred)
print("precision: ",precision)
#value: 1.0

#recall
recall = recall_score(y_test,y_pred)
print("recall: ",recall)
#values:0.8214285714285714

#confusion matrix
cm = confusion_matrix(y_test,y_pred)
print("confusion matrix: ",cm)

'''
model predictions [0 0 0 0 0 0 1 0 1 0]
accuracy 0.9166666666666666
precision:  1.0
recall:  0.8214285714285714
confusion matrix:  [[32  0]
 [ 5 23]]

'''