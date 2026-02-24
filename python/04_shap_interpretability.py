#   - compute SHAP explanations for the best model
#   - export: reports/figures/shap_summary.png

import os
import shap
import matplotlib.pyplot as plt

# make sure output folder exists
os.makedirs("reports/figures", exist_ok=True)

#  SHAP depends on the model type

# tree models (random forest / xgboost)
if best_model_name in ["random_forest", "xgboost"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # binary classification: class 1 = "high volatility"
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=200)
    plt.show()

# linear model (logistic regression)
elif best_model_name == "logistic_regression":
    explainer = shap.LinearExplainer(
        best_model,
        X_train,
        feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=200)
    plt.show()

else:
    raise ValueError(f"Unsupported model for SHAP: {best_model_name}")

print("saved: reports/figures/shap_summary.png")
