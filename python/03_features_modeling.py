#   - split data by time (first 80% train, last 20% test)
#   - train: logistic regression + random forest (+ xgboost if available)
#   - compute: accuracy, f1, roc-auc
#   - export: reports/figures/model_performance_table.png

from google.colab import files
uploaded = files.upload()  # after you run this, Colab shows "Choose files"

import pandas as pd

df = pd.read_csv("modeling_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Loaded:", df.shape)
df.head()

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# target check
if "target_high_volatility" not in df.columns:
    raise ValueError("target_high_volatility is missing in the dataset.")

df["target_high_volatility"] = df["target_high_volatility"].astype(int)

# safe default
if "event_window" not in df.columns:
    df["event_window"] = 0

# lag features (past only)
for lag in [1, 2, 3]:
    df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)
    df[f"epu_lag{lag}"] = df["epu_index"].shift(lag)

df = df.dropna().reset_index(drop=True)

feature_cols = [
    "log_return_lag1", "log_return_lag2", "log_return_lag3",
    "epu_lag1", "epu_lag2", "epu_lag3",
    "event_window"
]

# add extra features if they exist
for c in ["garch_sigma", "vol_7d", "vol_14d", "vol_30d"]:
    if c in df.columns:
        feature_cols.append(c)

X = df[feature_cols].copy()
y = df["target_high_volatility"].copy()

print("features:", feature_cols)
print("rows after lags:", len(df))

split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("train:", len(X_train), "test:", len(X_test))
print("test period:", df["date"].iloc[split_idx].date(), "->", df["date"].iloc[-1].date())

models = {}

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
models["logistic_regression"] = logreg

rf = RandomForestClassifier(n_estimators=400, random_state=42)
rf.fit(X_train, y_train)
models["random_forest"] = rf

# optional xgboost
try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb
except Exception:
    print("xgboost not available, skipping it.")

print("trained:", list(models.keys()))

rows = []

for name, model in models.items():
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, probs)
    else:
        roc = np.nan

    rows.append([name, acc, f1, roc])

results = pd.DataFrame(rows, columns=["model", "accuracy", "f1", "roc_auc"])
results = results.sort_values("f1", ascending=False).reset_index(drop=True)

results

os.makedirs("reports/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis("off")

vals = np.round(results[["accuracy","f1","roc_auc"]].values, 3)
row_labels = results["model"].tolist()

ax.table(
    cellText=vals,
    colLabels=["accuracy","f1","roc_auc"],
    rowLabels=row_labels,
    loc="center"
)

plt.tight_layout()
plt.savefig("reports/figures/model_performance_table.png", dpi=200)
plt.show()

print("saved: reports/figures/model_performance_table.png")

best_model_name = results.iloc[0]["model"]
best_model = models[best_model_name]

print("best model:", best_model_name)

import shap
import matplotlib.pyplot as plt
import os

os.makedirs("reports/figures", exist_ok=True)

if best_model_name in ["random_forest", "xgboost"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # binary: class 1 = target_high_volatility
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=200)
    plt.show()

elif best_model_name == "logistic_regression":
    explainer = shap.LinearExplainer(best_model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=200)
    plt.show()

print("saved: reports/figures/shap_summary.png")
