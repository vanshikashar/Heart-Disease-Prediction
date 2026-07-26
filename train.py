# ==============================
# Heart Disease Prediction
# train.py
# ==============================

import os
import joblib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")

# ==============================
# Create Required Folders
# ==============================

os.makedirs("Images", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("Dataset/heart.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# ==============================
# Dataset Information
# ==============================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# ==============================
# Exploratory Data Analysis
# ==============================

sns.set_style("whitegrid")

# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.savefig("Images/target_distribution.png")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig("Images/age_distribution.png")
plt.show()

# Cholesterol Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["chol"], bins=20, kde=True)
plt.title("Cholesterol Distribution")
plt.savefig("Images/chol_distribution.png")
plt.show()

# Heart Rate Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["thalach"], bins=20, kde=True)
plt.title("Maximum Heart Rate")
plt.savefig("Images/thalach_distribution.png")
plt.show()

# Chest Pain vs Target
plt.figure(figsize=(6,4))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.savefig("Images/chest_pain.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("Images/correlation_heatmap.png")
plt.show()

# ==============================
# Feature Engineering
# ==============================

categorical_columns = ["cp", "restecg", "slope", "thal"]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# ==============================
# Split Features and Target
# ==============================

X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==============================
# Feature Scaling
# ==============================

continuous_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

scaler = StandardScaler()

X_train[continuous_columns] = scaler.fit_transform(
    X_train[continuous_columns]
)

X_test[continuous_columns] = scaler.transform(
    X_test[continuous_columns]
)

# ==============================
# Logistic Regression Model
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# ==============================
# Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n")
print("="*50)
print("MODEL PERFORMANCE")
print("="*50)

print("Accuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))
print("ROC AUC  :", round(roc,4))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease","Disease"],
    yticklabels=["No Disease","Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Images/confusion_matrix.png")
plt.show()

# ==============================
# ROC Curve
# ==============================

RocCurveDisplay.from_estimator(model, X_test, y_test)

plt.title("ROC Curve")
plt.savefig("Images/roc_curve.png")
plt.show()

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

print("\nTop Health Indicators\n")
print(importance[["Feature","Coefficient"]].head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Coefficient",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("Images/feature_importance.png")
plt.show()

# ==============================
# Save Model
# ==============================

joblib.dump(model, "Model/heart_model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")
joblib.dump(X.columns.tolist(), "Model/features.pkl")

print("\n")
print("="*50)
print("Model Saved Successfully!")
print("="*50)
print("Model -> Model/heart_model.pkl")
print("Scaler -> Model/scaler.pkl")
print("Features -> Model/features.pkl")