# 🇨🇦 Canadian Macroeconomic Forecasting with XGBoost

Forecasting key Bank of Canada macroeconomic indicators using gradient boosting and time series machine learning techniques, developed as part of **Carleton University's team at the Bank of Canada Governor's Challenge 2025**.

---

## 📌 Overview

This project builds machine learning models to forecast three core Canadian economic indicators on a year-over-year (YoY) basis:

| Notebook | Target Variable | Frequency |
|---|---|---|
| `Latest_New_GDP_YoY.ipynb` | GDP Growth (YoY %) | Quarterly |
| `Latest_New_CPI_YoY.ipynb` | Total CPI (YoY %) | Monthly |
| `Latest_New_CPI_Core_YoY.ipynb` | Core CPI (YoY %) | Monthly |

Each model uses **XGBoost** with lag-based feature engineering and a **multi-output autoregressive** forecasting pipeline to generate multi-step ahead predictions.

---

## 🧠 Methodology

### Feature Engineering
- **Lagged features**: Up to 12 lags (monthly) or 4 lags (quarterly) for all input variables
- **Time features**: Month/quarter number and a linear time index for trend capture
- **YoY transformation**: Key variables are converted to year-over-year percentage changes

### Input Features
Exogenous predictors used across models include:

- Bank of Canada Target Rate
- WTI & Brent Crude Oil Prices
- Real Exchange Rate (CAD/USD)
- US Real GDP Growth
- Canadian Unemployment Rate
- Population Growth
- Final Private Consumption, Government Spending, Gross Investment, Exports, Imports

### Model
- **Algorithm**: XGBoost Regressor (`n_estimators=1000`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.7`)
- **Validation**: 5-fold `TimeSeriesSplit` cross-validation (respects temporal ordering)
- **Evaluation metrics**: MAE, MSE, RMSE, R²

### Forecasting Pipeline
A **multi-output autoregressive** approach is used for future forecasting:
- All target variables are predicted simultaneously using `MultiOutputRegressor`
- Exogenous assumptions (e.g., BoC rate, oil prices, exchange rate) are held constant or set to user-defined values
- Predictions are rolled forward step-by-step to generate 8–24 month/quarter ahead forecasts

---

## 📁 Project Structure

```
├── GDP_YoY.ipynb                   # Quarterly GDP forecasting model
├── CPI_YoY.ipynb                   # Monthly total CPI forecasting model
├── CPI_Core_YoY.ipynb              # Monthly core CPI forecasting model
├── Quarterly_data_new.csv          # Quarterly input data
├── Monthly_data_new.csv            # Monthly input data
├── GDP_Results.csv                 # GDP forecast output
├── CPI_Results.csv                 # CPI forecast output
└── Core_CPI_Results.csv            # Core CPI forecast output
```

---

## 🛠️ Tech Stack

- **Python 3**
- `pandas`, `numpy` — data processing
- `xgboost` — gradient boosting model
- `scikit-learn` — cross-validation, metrics, MultiOutputRegressor
- `matplotlib`, `seaborn` — visualization

---

## 📊 Output

Each model produces:
- **Backtest plots**: Actual vs. predicted values across TimeSeriesSplit folds
- **Fold-level metrics**: MAE, RMSE, R² per fold and averaged
- **Forward forecast CSV**: Multi-step ahead predictions for the target variable

---

## 📝 Notes & Assumptions

For the forward forecasting pipeline, the following assumptions are applied by default:
- **BoC Target Rate**: Reduced by 25 bps from last observed value (rate cut scenario)
- **WTI Oil**: Fixed at $60/barrel
- **Brent Oil**: Fixed at $65/barrel
- **Real Exchange Rate**: Held constant at last observed value
- **Population Growth**: Held constant at 0.5% YoY

These can be adjusted directly in the `forecast_multivar_single_model()` function.

---

## 📄 Data Sources

Data sourced from publicly available Bank of Canada, Statistics Canada, and financial market datasets.

---

*Developed as part of Carleton University's forecasting team for the Bank of Canada Governor's Challenge 2025. XGBoost model outputs were ensembled with VAR models to produce final policy recommendations presented to a panel of judges.*
