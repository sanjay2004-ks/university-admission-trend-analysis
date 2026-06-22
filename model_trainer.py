"""
model_trainer.py
Trains Logistic Regression, Random Forest, Linear Regression, and KMeans
on admissions_data.csv and saves models as .pkl files.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("admissions_data.csv")

FEATURES = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]
X = df[FEATURES]
y_cls = df["Admitted"]
y_reg = df["Chance_of_Admit"]

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

# ── Logistic Regression ────────────────────────────────────────────────────
print("=" * 60)
print("Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_cls_train)
lr_pred = lr.predict(X_test)
print(f"  Accuracy : {accuracy_score(y_cls_test, lr_pred):.4f}")
print(classification_report(y_cls_test, lr_pred, target_names=["Not Admitted", "Admitted"]))
joblib.dump(lr, "logistic_regression.pkl")
print("  Saved: logistic_regression.pkl")

# ── Random Forest ──────────────────────────────────────────────────────────
print("=" * 60)
print("Random Forest Classifier")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_cls_train)
rf_pred = rf.predict(X_test)
print(f"  Accuracy : {accuracy_score(y_cls_test, rf_pred):.4f}")
print(classification_report(y_cls_test, rf_pred, target_names=["Not Admitted", "Admitted"]))
print("  Feature importances:")
for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:20s}: {imp:.4f}")
joblib.dump(rf, "random_forest.pkl")
print("  Saved: random_forest.pkl")

# ── Linear Regression ──────────────────────────────────────────────────────
print("=" * 60)
print("Linear Regression (Chance_of_Admit)")
linreg = LinearRegression()
linreg.fit(X_train, y_reg_train)
linreg_pred = linreg.predict(X_test)
print(f"  R²  : {r2_score(y_reg_test, linreg_pred):.4f}")
print(f"  MAE : {mean_absolute_error(y_reg_test, linreg_pred):.4f}")
joblib.dump(linreg, "linear_regression.pkl")
print("  Saved: linear_regression.pkl")

# ── KMeans Clustering ──────────────────────────────────────────────────────
print("=" * 60)
print("KMeans Clustering (k=3)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
# Label clusters by mean CGPA so ordering is stable across runs
cluster_cgpa = {c: df.loc[km.labels_ == c, "CGPA"].mean() for c in range(3)}
sorted_clusters = sorted(cluster_cgpa, key=cluster_cgpa.get)
label_map = {sorted_clusters[0]: "Low Profile", sorted_clusters[1]: "Mid Profile", sorted_clusters[2]: "High Profile"}
df["Cluster"] = [label_map[c] for c in km.labels_]
print("  Cluster distribution:")
print(df["Cluster"].value_counts().to_string())
joblib.dump({"model": km, "scaler": scaler, "label_map": label_map}, "kmeans.pkl")
print("  Saved: kmeans.pkl")

# Save labelled data for dashboard use
df.to_csv("admissions_data.csv", index=False)
print("=" * 60)
print("All models trained and saved successfully.")
