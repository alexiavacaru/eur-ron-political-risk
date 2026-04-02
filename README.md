## project purpose
- analyze the relationship between political risk and eur/ron exchange rate volatility
- assess whether political uncertainty is associated with currency instability
- combine classical econometrics with machine learning
- evaluate whether volatility regimes can be predicted

## what is analyzed
- eur/ron exchange rate time series (monthly frequency)
- log returns
- conditional volatility (garch(1,1))
- major political event windows
- economic policy uncertainty (epu index)
- classification of high-volatility regimes

## research questions
- are political events associated with increases in eur/ron volatility?
- are structural breaks linked to political shocks?
- can high-volatility periods be predicted?

## data sources
- eur/ron exchange rate (bnr)
- economic policy uncertainty index (epu)
- manually curated political event dates
- analysis period: 2005–present

## constructed variables
- log_return
- rolling_volatility_7d / 14d / 30d
- lag_1 / lag_2 / lag_3
- event_window_dummy (0/1)
- garch_sigma
- target_high_volatility (0/1)

## methods used in r
- time series preprocessing
- adf stationarity test
- structural break detection (bai-perron)
- garch(1,1) modeling
- conditional volatility estimation
- figure export

## methods used in python
- Feature Engineering: Data scaling (x100), lagged variables (L1-L3), and volatility thresholding (75th percentile).
- Time-Aware Split: Chronological 80/20 train/test division to prevent data leakage.
- Statistical Validation: Multicollinearity checks via VIF (Variance Inflation Factor).
- Machine Learning Models: Comparative analysis using Logistic Regression, Random Forest, and XGBoost.
- Imbalance Handling: Implementation of scale_pos_weight to improve detection of rare volatility events.
- Evaluation Metrics: Performance measured via Accuracy, F1-Score, and ROC-AUC.
- Explainable AI (SHAP): Model interpretability using SHAP Values to quantify political impact (EPU) on FX risk.
- Risk Monitoring: Visualization of predicted probabilities as an Early Warning System (EWS).

## generated outputs
all figures are exported to the /reports directory:
- volatility_regimes.png - Returns time series plot
- risk_monitoring.png - GARCH conditional volatility plot
- vif_correlation.png - Structural breaks plot
- roc_curve.png - Model performance table
- shap_summary.png - SHAP summary plot
- interpretations for each figure 

## project contribution
- integration of econometrics and machine learning
- applied analysis on the romanian foreign exchange market
- fully reproducible analytical pipeline
- research-oriented repository structure

## repository structure
- data_raw/ → sources and download scripts
- data_processed/ → cleaned datasets
- r/ → econometric analysis
- python/ → modeling and interpretability
- reports/ → figures and final outputs

## reproduction steps
- download raw data (bnr script + manual epu download)
- execute 01_cleaning.R
- execute 02_breaks_garch.R
- generate modeling_dataset.csv
- run python notebooks
