1. volatility_regimes.png
Interpretation: This time series plot identifies historical clusters of high volatility in the EUR/RON exchange rate. The red-shaded areas represent "stress regimes" (thresholded at the 75th percentile), successfully capturing major shocks like the 2008 financial crisis and localized political instability.

2. vif_correlation.png
Interpretation: The heatmap confirms a moderate correlation between political uncertainty (EPU) and event windows. Crucially, the Variance Inflation Factor (VIF) values are all below the critical threshold of 5, proving that the model is free from significant multicollinearity and that the features are statistically independent.

3. roc_curve.png
Interpretation: The model achieves an AUC of 0.65, indicating a solid ability to distinguish between stable and volatile periods. This performance significantly outperforms a random baseline (0.5), validating the inclusion of political indices in the prediction of financial risk.

4. shap_summary.png
Interpretation: The SHAP summary plot reveals that while historical return lags are the primary drivers, the Economic Policy Uncertainty (EPU) index has a clear marginal impact. High EPU values (red dots) consistently push the model’s prediction toward a high-volatility regime.

5. risk_monitoring.png
Interpretation: This chart compares the model's estimated crisis probabilities against real-world events. A major spike in 2022 shows the model correctly signaling high-stress periods before they occurred, functioning as an effective Early Warning System (EWS) for currency traders and policymakers.
