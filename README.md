# Heart Disease Prediction System

This project predicts heart disease (`condition`: 0 = no disease, 1 = disease) using the Cleveland heart disease dataset (**297 rows, 14 columns**).

The main model is **logistic regression**. Random forest and a hyperparameter-tuned random forest were also tested, but both performed worse on this dataset.

## Dataset

- Source: [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) (`data/raw/heart_cleveland_upload.csv`)
- After cleaning: `data/processed/heart_disease_cleaned.csv`
- Target: `condition`
- Class balance: 160 patients with no disease (0), 137 with disease (1)

## Exploratory analysis

### Heart disease distribution

![Heart disease distribution](plots/heart%20disease%20distribution.png)

The dataset is fairly balanced between patients with and without heart disease.

### Distribution of patient features

![Distribution of patient features](plots/distribution%20of%20patients%20features.png)

Histograms of `age`, `trestbps` (resting blood pressure), `chol` (cholesterol), and `thalach` (maximum heart rate).

### Potential outliers

![Potential outliers](plots/potential%20outlier.png)

Boxplots of the same numeric features. Cholesterol has the most extreme outliers.

## How the models were trained

- Train/test split: 80% / 20%
- `random_state=42`
- Stratified by `condition`

Logistic regression used a pipeline:

- `StandardScaler` on numeric features: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- `OneHotEncoder` on categorical features: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`

## Results — logistic regression (main model)

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 91.67%  |
| Precision | 100%    |
| Recall    | 82.14%  |

Confusion matrix:

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| Actual 0       | 32          | 0           |
| Actual 1       | 5           | 23          |

Precision of 1.0 means every predicted disease case was correct (no false positives). Recall of 0.82 means 5 disease cases were missed.

## Other models tried

Random forest and GridSearchCV-tuned random forest were trained on the same split, but both had **lower accuracy**.

| Model                         | Accuracy | Precision | Recall |
|-------------------------------|----------|-----------|--------|
| Logistic regression           | 91.67%   | 100%      | 82.14% |
| Random forest                 | 83.33%   | 87.50%    | 75.00% |
| Random forest + GridSearchCV  | 81.67%   | 86.96%    | 71.43% |

Tuned random forest best parameters (`scoring="recall"`, 5-fold CV):

- `n_estimators=100`
- `max_depth=5`
- `max_features=sqrt`
- `min_samples_split=10`
- `min_samples_leaf=4`
- `class_weight=balanced`

### Why random forest did worse

1. **Small dataset (297 rows, 14 columns)** — tree ensembles usually need more data to beat a simple linear model. With this size, they overfit the training fold and do not generalize as well.
2. **The signal is probably close to linear** — if the relationship between features and `condition` is roughly linear, logistic regression is a better fit. Extra tree complexity does not add useful signal and can add noise.

## Project structure

```
data/raw/              raw Cleveland CSV
data/processed/        cleaned CSV
plots/                 EDA figures
src/data_cleaning.py
src/logistic_regression.py
src/random_forest.py
src/random_forest_GridSearch.py
```

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
