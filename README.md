## project purpose
- analyze the relationship between political risk and eur/ron exchange rate volatility
- test whether political uncertainty increases currency instability
- combine classical econometrics with machine learning
- evaluate whether volatility regimes can be predicted

## what is analyzed
- daily eur/ron exchange rate series
- log returns
- conditional volatility (garch)
- major political event windows
- economic policy uncertainty (epu index)
- classification of high-volatility regimes

## research questions
- do political events increase eur/ron volatility?
- are structural breaks associated with political shocks?
- can high-volatility periods be predicted?

## data sources
- eur/ron exchange rate (bnr)
- economic policy uncertainty index (epu)
- manually curated political event dates
- analysis period: [fill in]

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
- structural break tests (chow / bai-perron)
- arma-garch(1,1) modeling
- conditional volatility estimation
- figure export

## methods used in python
- feature engineering
- time-aware train/test split
- logistic regression
- random forest
- xgboost (optional)
- shap values for interpretability
- evaluation metrics: accuracy, f1-score, roc-auc

## generated outputs
- returns time series plot
- garch conditional volatility plot
- structural breaks plot
- model performance table
- shap summary plot

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
- reports/ → figures and final report

## reproduction steps
- run data download script
- execute 01_cleaning.R
- execute 02_breaks_garch.R
- run python notebooks

