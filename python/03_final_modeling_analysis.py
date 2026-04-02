import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("modeling_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Date încărcate: {df.shape[0]} rânduri.")

# 1. Transformăm în procente ca să vedem variația (0.0001 devine 0.01)
cols_to_fix = ['log_return', 'vol_7d', 'vol_14d', 'vol_30d']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col] * 100

# 2. Eliminăm rândurile cu NA (cele de la început care nu au medii mobile)
df = df.dropna().reset_index(drop=True)

# 3. Creăm Lag-uri (informația de ieri pentru a prezice azi)
for lag in [1, 2, 3]:
    df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)
    df[f"epu_lag{lag}"] = df["epu_index"].shift(lag)

df = df.dropna().reset_index(drop=True)

# 4. Ajustăm Target-ul: Dacă e prea strict, modelul nu vede crizele
# Setăm pragul la percentila 75 (cele mai agitate 25% din zile)
threshold = df['vol_7d'].quantile(0.75)
df['target_high_volatility'] = (df['vol_7d'] > threshold).astype(int)

print("Curățare completă. NA eliminate. Date scalate x100.")
# Pregătire date pentru ML
features = [col for col in df.columns if 'lag' in col or col in ['event_window', 'epu_index']]
X = df[features]
y = df["target_high_volatility"]

# Split 80% Train / 20% Test (fără amestecare, păstrăm ordinea timpului)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Calculăm greutatea claselor pentru XGBoost
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss')
}

perf_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    perf_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs)
    })

performance_df = pd.DataFrame(perf_results)
print(performance_df.round(3))
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['log_return'], label='Randament EUR/RON (%)', color='steelblue', alpha=0.8)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Evidențiem zonele de volatilitate ridicată
plt.fill_between(df['date'], df['log_return'].min(), df['log_return'].max(), 
                 where=df['target_high_volatility']==1, color='red', alpha=0.2, label='Regim Volatilitate Ridicată')

plt.title("Analiza EUR/RON: Randamente și Perioade de Risc Politic Detectate", fontsize=14)
plt.ylabel("Variație Procentuală (%)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
# Folosim XGBoost pentru SHAP
explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.title("Impactul Factorilor Politici asupra Volatilității (SHAP Values)")
plt.show()

import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Heatmap de Corelație - Esențial pentru a vedea legătura EPU -> Volatilitate
plt.figure(figsize=(10, 8))
corr_matrix = df[['log_return', 'vol_7d', 'epu_index', 'event_window']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
plt.title("Matricea de Corelație: Analiza legăturii Politic-Economic")
plt.show()

# VIF - Să vedem dacă lag-urile sunt prea corelate (Multicoliniaritate)
X_vif = X_train.assign(const=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print("\n--- Analiza Multicoliniarității (VIF) ---")
print(vif_data)

from sklearn.metrics import roc_curve, auc
best_model = models["XGBoost"] 
# Calculăm probabilitățile și curba
probs = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Alarme False)')
plt.ylabel('True Positive Rate (Crize Detectate)')
plt.title('Performanța Modelului în Detecția Instabilității (Curba ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Creăm un DataFrame de comparație pentru perioada de test
comparison_df = pd.DataFrame({
    'Data': df.iloc[split:]['date'],
    'Real': y_test.values,
    'Probabilitate_Model': best_model.predict_proba(X_test)[:, 1]
})

plt.figure(figsize=(15, 5))
plt.plot(comparison_df['Data'], comparison_df['Probabilitate_Model'], label='Probabilitate de Criză (Estimată)', color='purple')
plt.scatter(comparison_df[comparison_df['Real']==1]['Data'], [1]*len(comparison_df[comparison_df['Real']==1]), 
            color='red', marker='|', label='Eveniment Volatilitate Real')
plt.axhline(0.5, color='gray', linestyle='--')
plt.title("Monitorizarea Riscului: Probabilitate Estimată vs. Realitate")
plt.legend()
plt.show()
