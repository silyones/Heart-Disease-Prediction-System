import pandas as pd

#load the dataset
df = pd.read_csv("C:\Heart Disease Prediction System/data/raw/heart_cleveland_upload.csv")

#display the first 5 rows
print(df.head())

#display shape
print("shape:", df.shape)

#display info
print("info:", df.info())

#display describe
print("describe:", df.describe())

#check how many are missing
print(df.isnull().sum())
