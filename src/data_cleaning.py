import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


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

#check duplicate
print(df.duplicated().sum)

#drop duplicate
df = df.drop_duplicates()

#check shape after removing duplicate
print("shape: ", df.shape)

#count each column in condition
print(df["condition"].value_counts())

#show in percentage
print(df["condition"].value_counts(normalize=True) *100)

#check datatype
print(df.dtypes)

#check columns
for col in df.columns:
    print(col, df[col].unique())

correlation = df.corr(numeric_only=True)

#select few numerical features
features = ["age","trestbps","chol","thalach"]
df[features].hist(
    figsize=(10,8),
    bins=20,
    edgecolor="black"
)

plt.suptitle("distribution of patients features")
plt.tight_layout()
plt.savefig("plots/distribution of patients features.png",dpi=300)
plt.show()

#plot condition
sns.countplot(data=df, x="condition")
plt.title("heart disease distribution")
plt.xlabel("condition (0 = no disease, 1 = disease)")
plt.xticks(rotation=30)
plt.ylabel("number of patients")
plt.savefig("plots/heart disease distribution.png",dpi=300)
plt.show()

#plot boxplot
sns.boxplot(data =df[features])
plt.title("potential outlier")
plt.xticks(rotation=20)
plt.savefig("plots/potential outlier.png",dpi=300)
plt.show()

#heatmap
sns.heatmap(correlation,
annot=True,
cmap="coolwarm",
fmt=".2f")
plt.title("feature correlation matrix")
plt.show()

#save cleaned dataset
df.to_csv("data/processed/heart_disease_cleaned.csv")
print(df.shape)

#seperate feature  (X) and targe (Y)
x = df.drop("condition", axis=1)
y = df["condition"]

#print their shape
print("x features shape: ", x.shape)
print("y label shape: ", y.shape)


x_train, x_test, y_train, y_test = train_test_split(
   x,
   y,
   test_size=0.2,
   random_state=42,
   stratify=y
)

print("training data:", x_train.shape)
print("testing data:", x_test.shape)